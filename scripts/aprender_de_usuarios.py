"""El ciclo completo de aprendizaje con datos de usuarios, en un comando (README §18).

1. Importa los aportes (archivo ``.json`` de la página web o carpeta de CSV de la app).
2. Reentrena: ``kedro run --pipeline modelado_supervisado`` (consenso, sin fuga y
   control de no empeorar) y ``grafo_biomecanico`` (familias y recomendaciones).
3. Regenera la página (``docs/index.html``) y la copia de la app de iPhone.
4. Muestra qué aprendió el modelo.

Uso, desde la raíz del repositorio::

    python scripts/aprender_de_usuarios.py ~/Downloads/sustituto-seguro-aportes-2026-11-24.json
    python scripts/aprender_de_usuarios.py <archivo> <otro archivo> <carpeta de la app>
    python scripts/aprender_de_usuarios.py --solo-reentrenar      # con los aportes ya importados
    python scripts/aprender_de_usuarios.py --ejemplo              # 100 aportes de ejemplo (demostración)
    python scripts/aprender_de_usuarios.py --deshacer             # vuelve a la versión sin aportes

Tarda unos 3 minutos. ``--ejemplo`` simula usuarios honestos: etiqueta 100 ejercicios del
catálogo con la etiqueta de su gemelo de free-exercise-db (el mismo criterio de la
simulación, README §18), marcados con el comentario «ejemplo simulado». ``--deshacer``
guarda los aportes en una carpeta de respaldo y reentrena sin ellos: las cifras vuelven
a ser las de la EV2 y la EV3.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import importar_aportes  # noqa: E402

REPORTES = RAIZ / "data/08_reporting"
NOMBRE = {"mecanica": "Mecánica", "fuerza": "Fuerza", "complejidad_tecnica": "Complejidad técnica"}


def paso(titulo: str, comando: list[str]) -> None:
    print(f"\n▶ {titulo}")
    inicio = time.time()
    resultado = subprocess.run(comando, cwd=RAIZ, capture_output=True, text=True)
    if resultado.returncode != 0:
        print(resultado.stdout[-3000:], resultado.stderr[-3000:], sep="\n")
        raise SystemExit(f"Falló: {' '.join(comando)}")
    print(f"  listo en {time.time() - inicio:.0f} s")


def aportes_de_ejemplo(destino: Path) -> Path:
    """100 correcciones honestas simuladas (40 de mecánica, 40 de fuerza y 20 de
    complejidad), sobre ejercicios del catálogo cuyo gemelo quedó en *train*."""
    tabla = pd.read_parquet(RAIZ / "data/05_model_input/tabla_base.parquet")
    perfil = pd.read_parquet(RAIZ / "data/07_model_output/perfil_exigencia.parquet").set_index("ejercicio_id")
    en_train = set(tabla.loc[tabla["particion"] == "train", "ejercicio_id"])
    fecha = datetime.now(timezone.utc).isoformat(timespec="seconds")
    filas = []
    for objetivo, n in [("mecanica", 40), ("fuerza", 40), ("complejidad_tecnica", 20)]:
        pool = tabla[(tabla["rol"] == "prediccion") & tabla[f"{objetivo}_referencia"].notna()
                     & tabla["gemelo_fedb"].isin(en_train)].sample(n, random_state=2026)
        for fila in pool.itertuples():
            e = fila.ejercicio_id
            anterior = perfil.at[e, objetivo]
            filas.append(dict(zip(importar_aportes.COLUMNAS["correcciones_etiquetas"], [
                e, fila.nombre, objetivo, anterior if pd.notna(anterior) else "",
                perfil.at[e, f"{objetivo}_origen"] or "", float(perfil.at[e, f"{objetivo}_confianza"]),
                getattr(fila, f"{objetivo}_referencia"), "ejemplo simulado", fecha])))
    archivo = destino / "aportes_de_ejemplo.json"
    archivo.write_text(json.dumps({"formato": importar_aportes.FORMATO_WEB, "version": 1, "origen": "ejemplo",
                                   "tablas": {"correcciones_etiquetas": filas}}, ensure_ascii=False), encoding="utf-8")
    return archivo


def deshacer() -> None:
    """Mueve los aportes importados a una carpeta de respaldo y deja la tabla vacía."""
    carpeta = importar_aportes.DESTINO
    respaldo = carpeta / f"respaldo-{datetime.now():%Y%m%d-%H%M%S}"
    respaldo.mkdir(parents=True)
    for archivo in carpeta.glob("*.csv"):
        shutil.copy2(archivo, respaldo / archivo.name)
        if archivo.name != "correcciones_etiquetas.csv":
            archivo.unlink()
    pd.DataFrame(columns=importar_aportes.COLUMNAS["correcciones_etiquetas"]).to_csv(
        carpeta / "correcciones_etiquetas.csv", index=False)
    print(f"▶ Aportes guardados en {respaldo.relative_to(RAIZ)}; la tabla de correcciones quedó vacía.")


def resumen() -> None:
    consolidado = pd.read_csv(REPORTES / "aportes_consolidados.csv")
    informe = pd.read_csv(REPORTES / "aprendizaje_usuarios_informe.csv")
    print("\n═══ Qué aprendió el modelo ═══")
    if consolidado.empty:
        print("No hay aportes: el modelo es el de la EV2 y la EV3.")
        return
    estados = consolidado["estado"].value_counts()
    print(f"Etiquetas aportadas: {len(consolidado)} · " + " · ".join(f"{n} {e}" for e, n in estados.items()))
    for fila in informe.itertuples():
        if fila.n_aportes == 0:
            print(f"\n{NOMBRE[fila.objetivo]}: sin aportes")
            continue
        print(f"\n{NOMBRE[fila.objetivo]} ({fila.modelo}): {fila.n_aportes} aportes "
              f"({fila.n_corrigen_fuente} corrigen la fuente, {fila.n_ejercicios_nuevos} ejercicios nuevos)")
        print(f"  F1 en prueba          sin aportes {fila.f1_test_sin_aportes:.3f} → con aportes {fila.f1_test_con_aportes:.3f}")
        print(f"  F1 catálogo externo   sin aportes {fila.f1_externa_sin_aportes:.3f} → con aportes {fila.f1_externa_con_aportes:.3f}")
        print(f"  Coincide con lo que enseñaron los usuarios: antes {fila.coincide_antes:.0%}, "
              f"en ejercicios que no vio {fila.coincide_fuera_de_muestra:.0%}, después {fila.coincide_despues:.0%}")
        print(f"  Decisión: {fila.decision}")
    print("\nAbre docs/index.html: «El modelo» muestra el resultado y cada etiqueta aprendida aparece "
          "como «Aportado por usuarios».")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("origenes", nargs="*", type=Path, help="archivos .json de la web o carpetas de la app")
    ap.add_argument("--solo-reentrenar", action="store_true", help="no importar: reentrenar con lo ya importado")
    ap.add_argument("--ejemplo", action="store_true", help="importar 100 aportes honestos simulados (demostración)")
    ap.add_argument("--deshacer", action="store_true", help="respaldar y quitar los aportes, y reentrenar sin ellos")
    args = ap.parse_args()
    if not (args.origenes or args.solo_reentrenar or args.ejemplo or args.deshacer):
        ap.error("indica al menos un archivo de aportes, o --solo-reentrenar, --ejemplo o --deshacer")
    if args.deshacer:
        deshacer()
    for origen in args.origenes:
        print(f"▶ Importando {origen.name}")
        importar_aportes.importar(origen.expanduser())
    if args.ejemplo:
        with tempfile.TemporaryDirectory() as tmp:
            print("▶ Importando 100 aportes de ejemplo (usuarios honestos simulados)")
            importar_aportes.importar(aportes_de_ejemplo(Path(tmp)))

    kedro = [sys.executable, "-m", "kedro", "run", "--pipeline"]
    paso("Reentrenando los modelos con los aportes (modelado_supervisado)", kedro + ["modelado_supervisado"])
    paso("Recalculando familias y recomendaciones (grafo_biomecanico)", kedro + ["grafo_biomecanico"])
    paso("Regenerando los datos de la página", [sys.executable, "scripts/exportar_datos_web.py"])
    paso("Armando docs/index.html", [sys.executable, "scripts/armar_web.py"])
    paso("Actualizando la copia de la app de iPhone", [sys.executable, "scripts/exportar_modelo_ios.py"])
    resumen()


if __name__ == "__main__":
    main()
