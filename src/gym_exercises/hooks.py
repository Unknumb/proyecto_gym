"""
Hooks de Kedro — seguimiento de experimentos con MLflow (README §8.8).

Cada ``kedro run`` que incluya ``modelado_supervisado`` o ``grafo_biomecanico``
abre un *run* de MLflow y registra:

- **Parámetros**: los bloques ``modelado_supervisado`` y ``grafo_biomecanico`` de
  ``parameters.yml`` (con los ``--params`` de la línea de comandos ya aplicados).
- **Etiquetas**: pipeline, commit y rama de git, y si había cambios sin commitear.
- **Dataset**: ``tabla_modelado`` con su huella, para saber con qué datos se
  entrenó cada versión.
- **Métricas**: F1 macro, *accuracy*, Brier, etc. por objetivo, modelo y conjunto;
  el contraste pareado; las métricas de los agrupamientos; los barridos de *k* y
  de resolución como curvas; y el KPI de cobertura de sustitución.
- **Modelos**: el ganador de cada objetivo, con firma y ejemplo de entrada, y su
  versión en el Model Registry.
- **Artefactos**: los CSV de ``data/08_reporting`` que produjo la corrida.

El hook solo observa: no cambia ningún nodo, modelo ni salida. Si MLflow no está
instalado, si ``conf/*/mlflow.yml`` dice ``activo: false`` o si el registro falla,
el pipeline corre igual y solo queda un aviso en el log.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from kedro.framework.hooks import hook_impl

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
try:
    import mlflow
    from mlflow import MlflowClient
except ImportError:  # el pipeline no depende de MLflow
    mlflow = None

logger = logging.getLogger(__name__)

#: Bloques de ``parameters.yml`` que se registran si el pipeline los usa.
GRUPOS_PARAMETROS = ["modelado_supervisado", "grafo_biomecanico"]

#: Métricas por fila de ``modelos_comparacion``.
METRICAS_MODELOS = ["f1_macro", "accuracy", "balanced_accuracy", "f1_ic95_inf", "f1_ic95_sup",
                    "brier", "pr_auc_macro"]

#: Métricas por objetivo de ``modelos_contraste_pareado`` (complejo − simple).
METRICAS_CONTRASTE = ["diferencia_media", "pct_pliegues_complejo_mejor", "t_nadeau_bengio",
                      "p_valor"]

#: Métricas por método de ``clustering_comparacion``.
METRICAS_CLUSTERING = ["n_grupos", "silhouette_embedding", "davies_bouldin_embedding",
                       "modularidad_proyeccion", "ami_fuente", "ami_agonista_principal",
                       "ami_fuerza", "ari_entre_metodos"]

#: Métricas por método de ``sustitucion_cobertura`` (fila ``TODAS``).
METRICAS_SUSTITUCION = ["cobertura_misma_familia", "cobertura_total", "misma_fuerza_top3",
                        "con_sin_equipamiento_despues"]

#: Los modelos se guardan con skops, que a diferencia de pickle no ejecuta código al
#: cargarlos. Solo admite los tipos que se declaran de confianza: el árbol interno
#: del random forest es el único que no trae aprobado por defecto.
TIPOS_CONFIABLES_SKOPS = ["sklearn.tree._tree.Tree"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Conversión de las salidas a métricas de MLflow
# ─────────────────────────────────────────────────────────────────────────────
def aplanar(diccionario: dict[str, Any], prefijo: str = "") -> dict[str, str]:
    """``{"a": {"b": 1}}`` → ``{"a.b": "1"}``. MLflow acepta hasta 6000 caracteres
    por valor y, en los nombres, solo letras, dígitos, ``_ - . /`` y espacios:
    ``columna (lumbar)`` queda ``columna _lumbar_``."""
    plano: dict[str, str] = {}
    for clave, valor in diccionario.items():
        nombre = prefijo + re.sub(r"[^\w\-. /]", "_", str(clave))
        if isinstance(valor, dict):
            plano.update(aplanar(valor, f"{nombre}."))
        else:
            plano[nombre] = str(valor)[:6000]
    return plano


def _filas_a_metricas(df: pd.DataFrame, claves: list[str], columnas: list[str],
                      prefijo: str = "") -> dict[str, float]:
    metricas = {}
    for _, fila in df.iterrows():
        ruta = "/".join([prefijo, *(str(fila[c]) for c in claves)]).strip("/")
        for columna in columnas:
            if columna in fila and pd.notna(fila[columna]):
                metricas[f"{ruta}/{columna}"] = float(fila[columna])
    return metricas


def metricas_modelos(comparacion: pd.DataFrame) -> dict[str, float]:
    """``mecanica/regresion_logistica/test/f1_macro`` y ``.../cv_f1_macro``."""
    metricas = _filas_a_metricas(comparacion, ["objetivo", "modelo", "conjunto"],
                                 METRICAS_MODELOS)
    cv = comparacion.dropna(subset=["cv_f1_macro"]).drop_duplicates(["objetivo", "modelo"])
    metricas.update(_filas_a_metricas(cv, ["objetivo", "modelo"], ["cv_f1_macro", "cv_f1_std"]))
    return metricas


def metricas_contraste(contraste: pd.DataFrame) -> dict[str, float]:
    return _filas_a_metricas(contraste, ["objetivo"], METRICAS_CONTRASTE, "contraste")


def metricas_clustering(comparacion: pd.DataFrame) -> dict[str, float]:
    return _filas_a_metricas(comparacion, ["metodo"], METRICAS_CLUSTERING, "familias")


def metricas_sustitucion(cobertura: pd.DataFrame, metodo_elegido: str) -> dict[str, float]:
    """Resumen por método y, aparte, el KPI del método elegido (README §1.3)."""
    todas = cobertura[cobertura["articulacion"] == "TODAS"]
    metricas = _filas_a_metricas(todas, ["metodo"], METRICAS_SUSTITUCION, "sustitucion")
    elegido = todas[todas["metodo"] == metodo_elegido]
    if not elegido.empty:
        metricas["kpi/cobertura_sustitucion"] = float(elegido["cobertura_misma_familia"].iloc[0])
    return metricas


def curva_barrido(barrido: pd.DataFrame, eje: str, escala: float = 1) -> list[tuple[str, float, int]]:
    """Un barrido como serie ``(métrica, valor, paso)``: MLflow la grafica como curva.

    El paso debe ser entero: la resolución de Louvain se multiplica por 10.
    """
    serie = []
    for _, fila in barrido.iterrows():
        paso = int(round(fila[eje] * escala))
        for columna in barrido.columns.drop(eje):
            if pd.notna(fila[columna]):
                serie.append((columna, float(fila[columna]), paso))
    return serie


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hook
# ─────────────────────────────────────────────────────────────────────────────
def _git(*args: str, raiz: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=raiz, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


class MlflowHook:
    """Registra en MLflow cada corrida de los pipelines de modelado."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._raiz = Path.cwd()
        self._parametros: dict[str, Any] = {}
        self._rutas_catalogo: dict[str, str] = {}
        self._run_activo = False

    @contextmanager
    def _sin_romper(self, que: str):
        """Un fallo de registro avisa en el log, pero nunca detiene el pipeline."""
        try:
            with warnings.catch_warnings():
                # Las columnas anatómicas son conteos sin nulos: el aviso de MLflow
                # sobre enteros con valores faltantes no aplica.
                warnings.filterwarnings("ignore", message=".*Inferred schema contains integer")
                yield
        except Exception as error:  # noqa: BLE001
            logger.warning("MLflow: no se pudo registrar %s (%s).", que, error)

    @hook_impl
    def after_context_created(self, context) -> None:
        try:
            self._config = context.config_loader["mlflow"]
        except Exception:  # noqa: BLE001  (sin conf/*/mlflow.yml)
            self._config = {}
        self._raiz = Path(context.project_path)
        self._parametros = context.params
        catalogo = context.config_loader["catalog"]
        self._rutas_catalogo = {nombre: cfg["filepath"] for nombre, cfg in catalogo.items()
                                if isinstance(cfg, dict) and "filepath" in cfg}

    def _uri(self, uri: str) -> str:
        """Resuelve ``sqlite:///archivo.db`` relativo a la raíz del proyecto."""
        prefijo = "sqlite:///"
        if uri.startswith(prefijo) and not Path(uri[len(prefijo):]).is_absolute():
            return prefijo + str(self._raiz / uri[len(prefijo):])
        return uri

    @hook_impl
    def before_pipeline_run(self, run_params: dict[str, Any], pipeline) -> None:
        usados = [g for g in GRUPOS_PARAMETROS if f"params:{g}" in pipeline.inputs()]
        if not usados or not self._config.get("activo", False):
            return
        if mlflow is None:
            logger.warning("MLflow no está instalado: la corrida no se registra.")
            return

        with self._sin_romper("el inicio de la corrida"):
            mlflow.set_tracking_uri(self._uri(self._config["tracking_uri"]))
            nombre = self._config.get("experimento", "gym_exercises")
            if mlflow.get_experiment_by_name(nombre) is None:
                artefactos = self._raiz / self._config.get("carpeta_artefactos", "mlruns")
                mlflow.create_experiment(nombre, artifact_location=artefactos.as_uri())
            mlflow.set_experiment(nombre)

            pipelines = run_params.get("pipeline_names") or ["__default__"]
            sucio = _git("status", "--porcelain", "--untracked-files=no", raiz=self._raiz)
            mlflow.start_run(run_name="+".join(pipelines), nested=mlflow.active_run() is not None,
                             tags={"kedro.pipeline": ",".join(pipelines),
                                   "kedro.run_id": str(run_params.get("run_id", "")),
                                   "kedro.env": str(run_params.get("env") or "local"),
                                   "mlflow.source.git.commit": _git("rev-parse", "HEAD",
                                                                    raiz=self._raiz),
                                   "git.rama": _git("branch", "--show-current", raiz=self._raiz),
                                   "git.cambios_sin_commit": str(bool(sucio))})
            self._run_activo = True
            for grupo in usados:
                mlflow.log_params(aplanar(self._parametros.get(grupo, {}), f"{grupo}."))
            logger.info("MLflow: corrida %s en el experimento '%s'.",
                        mlflow.active_run().info.run_id, nombre)

    @hook_impl
    def after_node_run(self, node, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        if not self._run_activo:
            return
        metodo = self._parametros.get("grafo_biomecanico", {}).get("metodo_familias", "")

        if "model_input_tabla_modelado" in outputs:
            with self._sin_romper("la tabla de modelado"):
                tabla = outputs["model_input_tabla_modelado"]
                mlflow.log_input(mlflow.data.from_pandas(tabla, name="tabla_modelado"),
                                 context="entrenamiento")
        if "modelos_comparacion" in outputs:
            with self._sin_romper("las métricas de los modelos"):
                mlflow.log_metrics(metricas_modelos(outputs["modelos_comparacion"]))
        if "modelos_supervisados" in outputs:
            with self._sin_romper("los modelos supervisados"):
                self._registrar_modelos(outputs["modelos_supervisados"],
                                        inputs["model_input_tabla_modelado"])
        if "modelos_contraste_pareado" in outputs:
            with self._sin_romper("el contraste pareado"):
                mlflow.log_metrics(metricas_contraste(outputs["modelos_contraste_pareado"]))
        if "clustering_comparacion" in outputs:
            with self._sin_romper("las métricas de agrupamiento"):
                mlflow.log_metrics(metricas_clustering(outputs["clustering_comparacion"]))
        if "sustitucion_cobertura" in outputs:
            with self._sin_romper("la cobertura de sustitución"):
                mlflow.log_metrics(metricas_sustitucion(outputs["sustitucion_cobertura"], metodo))
        for salida, eje, escala in [("kmeans_barrido", "k", 1),
                                    ("louvain_barrido", "resolucion", 10)]:
            if salida in outputs:
                with self._sin_romper(f"el barrido {salida}"):
                    for metrica, valor, paso in curva_barrido(outputs[salida], eje, escala):
                        mlflow.log_metric(f"{salida}/{metrica}", valor, step=paso)

    def _registrar_modelos(self, modelos: dict[str, Any], tabla: pd.DataFrame) -> None:
        """El ganador de cada objetivo, con firma, ejemplo y versión registrada."""
        registrar = self._config.get("registrar_modelos", False)
        alias = self._config.get("alias_modelo")
        cliente = MlflowClient()
        for objetivo, modelo in modelos.items():
            estimador = modelo["estimador"]
            ejemplo = tabla[list(estimador.feature_names_in_)].head(5)
            mlflow.log_params({f"ganador.{objetivo}": modelo["modelo"],
                               f"umbral_abstencion.{objetivo}": round(modelo["umbral"], 4)})
            info = mlflow.sklearn.log_model(
                estimador, name=f"perfil_{objetivo}", input_example=ejemplo,
                skops_trusted_types=TIPOS_CONFIABLES_SKOPS,
                registered_model_name=f"gym_perfil_{objetivo}" if registrar else None,
                metadata={"modelo": modelo["modelo"], "umbral_abstencion": modelo["umbral"],
                          "clases": [str(c) for c in modelo["clases"]]})
            if registrar and alias and info.registered_model_version:
                cliente.set_registered_model_alias(f"gym_perfil_{objetivo}", alias,
                                                   info.registered_model_version)

    @hook_impl
    def after_pipeline_run(self, pipeline) -> None:
        if not self._run_activo:
            return
        with self._sin_romper("los reportes"):
            for salida in sorted(pipeline.all_outputs()):
                ruta = self._rutas_catalogo.get(salida, "")
                if ruta.startswith("data/08_reporting/") and (self._raiz / ruta).is_file():
                    mlflow.log_artifact(str(self._raiz / ruta), artifact_path="reportes")
        mlflow.end_run()
        self._run_activo = False

    @hook_impl
    def on_pipeline_error(self) -> None:
        if self._run_activo:
            mlflow.end_run(status="FAILED")
            self._run_activo = False
