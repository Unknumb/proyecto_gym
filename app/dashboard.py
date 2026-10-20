"""Dashboard de la EV2: recomendador biomecánico de ejercicios.

Sigue el orden de la presentación (10 minutos): KPIs, datos, supervisado, no
supervisado, sustitución en vivo y sesgo. No calcula nada nuevo: lee las salidas
de ``kedro run`` (``data/08_reporting``, ``data/07_model_output`` y ``models/``),
así que sus cifras son las del README. La demo usa el mismo ``Sustituidor`` del
pipeline.

Ejecutar desde la raíz del repositorio::

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from sklearn.metrics import cohen_kappa_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))   # el pickle de modelos referencia módulos del paquete
from gym_exercises.pipelines.grafo_biomecanico.nodes import Sustituidor  # noqa: E402

REPORTE = RAIZ / "data/08_reporting"
FIGURAS = REPORTE / "figures"
META_COBERTURA = 85.0
SENTADILLA = "catalogo:0043"
METODO = "kmeans_node2vec"

# Paleta: misma familia de colores que la guía de estudio.
ACENTO, ROJO, GRIS, AMBAR = "#0D6A73", "#B5332B", "#8A96A3", "#C98A1B"
COLOR_FUENTE = {"Catálogo": "#0D6A73", "free-exercise-db": "#C98A1B", "wger": "#7A5195"}
NOMBRE_FUENTE = {"catalogo": "Catálogo", "free_exercise_db": "free-exercise-db", "wger": "wger"}
NOMBRE_OBJETIVO = {"mecanica": "Mecánica", "fuerza": "Fuerza",
                   "complejidad_tecnica": "Complejidad técnica"}
NOMBRE_MODELO = {"regresion_logistica": "Regresión logística", "random_forest": "Random Forest",
                 "azar_estratificado": "Azar estratificado"}
NOMBRE_METODO = {"kmeans_node2vec": "Node2Vec + K-Means", "louvain": "Louvain",
                 "sin_familias": "Sin familias (solo reglas)"}
NOMBRE_CONTEXTO = {"gimnasio": "Gimnasio", "peso_corporal": "Peso corporal",
                   "rehabilitacion_funcional": "Rehabilitación", "cardio_otros": "Cardio y otros"}
MOTIVOS = {
    "sin_sustituto_anatomico": (
        "No existe un sustituto seguro: todos los músculos principales de este ejercicio "
        "cruzan la articulación lesionada. Consulta a un kinesiólogo."),
    "musculo_con_pocos_ejercicios": (
        "El músculo que habría que conservar tiene menos de 10 ejercicios en el catálogo: "
        "con tan pocas opciones el sistema prefiere no recomendar."),
    "no_carga_la_articulacion": "Este ejercicio no carga esa articulación: no hace falta sustituirlo.",
    "menos_de_3_alternativas": "Hay menos de 3 alternativas seguras: se muestran, pero la lista es corta.",
}

st.set_page_config(page_title="Recomendador biomecánico · EV2", page_icon="🦵", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:3.5rem;max-width:1280px}
h1,h2,h3{letter-spacing:-.01em}
.eyebrow{font-size:.75rem;letter-spacing:.09em;text-transform:uppercase;color:#B5332B;font-weight:600}
.lectura{border-left:3px solid #B5332B;padding:.35rem 0 .35rem .9rem;margin:.4rem 0 1rem;opacity:.92}
.tarjeta{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:14px 16px;height:100%}
.tarjeta .v{font-size:2rem;font-weight:800;line-height:1.05}
.tarjeta .l{font-size:.86rem;opacity:.75;margin-top:4px}
.tarjeta .m{font-size:.78rem;margin-top:8px;font-weight:600}
.ok{color:#23794A}.warn{color:#9A6100}.bad{color:#B5332B}.info{color:#0D6A73}
.chip{display:inline-block;border-radius:999px;padding:1px 10px;font-size:.78rem;font-weight:600;
      background:rgba(13,106,115,.12);color:#0D6A73;margin-right:6px}
.chip.otro{background:rgba(201,138,27,.15);color:#9A6100}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Carga (todo en caché: la demo responde al instante)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def reporte(nombre: str) -> pd.DataFrame:
    return pd.read_csv(REPORTE / f"{nombre}.csv")


@st.cache_data
def parametros() -> dict:
    return yaml.safe_load((RAIZ / "conf/base/parameters.yml").read_text(encoding="utf-8"))


@st.cache_data
def familias() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data/07_model_output/familias_movimiento.parquet")


@st.cache_data
def perfil_con_contexto() -> pd.DataFrame:
    perfil = pd.read_parquet(RAIZ / "data/07_model_output/perfil_exigencia.parquet")
    ejercicios = pd.read_parquet(RAIZ / "data/03_primary/ejercicios.parquet",
                                 columns=["ejercicio_id", "contexto"])
    return perfil.merge(ejercicios, on="ejercicio_id")


@st.cache_data
def niveles_dos_fuentes() -> pd.DataFrame:
    """Ejercicios de free-exercise-db que también tienen nivel en megaGymDataset."""
    e = pd.read_parquet(RAIZ / "data/03_primary/ejercicios.parquet",
                        columns=["nivel", "nivel_megagym", "nivel_origen"])
    return e[(e["nivel_origen"] == "free_exercise_db") & e["nivel_megagym"].notna()]


@st.cache_resource
def ganadores() -> dict[str, dict]:
    """Modelo ganador y umbral de abstención de cada objetivo (sin el estimador)."""
    modelos = joblib.load(RAIZ / "models/modelos_supervisados.pkl")
    return {o: {"modelo": m["modelo"], "umbral": float(m["umbral"])} for o, m in modelos.items()}


@st.cache_resource
def sustituidor() -> Sustituidor:
    return Sustituidor(familias(),
                       pd.read_parquet(RAIZ / "models/embeddings_node2vec.parquet"),
                       pd.read_csv(RAIZ / "data/03_primary/musculo_articulacion.csv"),
                       parametros()["grafo_biomecanico"])


def lectura(texto: str) -> None:
    """Frase de interpretación para el negocio debajo de cada gráfico."""
    st.markdown(f'<div class="lectura">{texto}</div>', unsafe_allow_html=True)


def tarjeta(columna, valor: str, etiqueta: str, estado: str, clase: str) -> None:
    columna.markdown(f'<div class="tarjeta"><div class="v">{valor}</div><div class="l">{etiqueta}</div>'
                     f'<div class="m {clase}">{estado}</div></div>', unsafe_allow_html=True)


def encabezado(eyebrow: str, titulo: str, minuto: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow} · {minuto}</div>', unsafe_allow_html=True)
    st.title(titulo)


def estilo(fig: go.Figure, alto: int = 380) -> go.Figure:
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=86, b=10), font=dict(size=14),
                      title=dict(y=0.98, yanchor="top", x=0, xanchor="left"), separators=",.",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


def pct(x: float) -> str:
    return f"{x:.1f} %".replace(".", ",")


def dec(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}".replace(".", ",")


def acierto_externo(objetivo: str) -> tuple[float, float, float]:
    """Acierto en el catálogo (validación externa) con y sin abstención, y cobertura."""
    g = ganadores()[objetivo]
    p = reporte("modelos_predicciones_test")
    p = p[(p["objetivo"] == objetivo) & (p["modelo"] == g["modelo"]) & (p["conjunto"] == "externa_catalogo")]
    acierta = p["y_real"] == p["y_pred"]
    retenidos = p["confianza"] >= g["umbral"]
    return acierta[retenidos].mean() * 100, acierta.mean() * 100, retenidos.mean() * 100


def cobertura_global() -> pd.Series:
    c = reporte("sustitucion_cobertura")
    return c[(c["metodo"] == METODO) & (c["articulacion"] == "TODAS")].iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Vistas
# ─────────────────────────────────────────────────────────────────────────────
def vista_kpis() -> None:
    encabezado("IE5 · IE8", "Sistema de recomendación biomecánica", "0:00–1:15")
    st.markdown("Las apps de gimnasio recomiendan por nombre o por popularidad: a quien tiene la "
                "**rodilla lesionada** le ofrecen una zancada en lugar de la sentadilla, otro "
                "ejercicio que carga la misma rodilla. Este sistema recomienda alternativas que "
                "trabajan el **mismo músculo** con el **mismo patrón de movimiento** sin cargar la "
                "articulación lesionada, y **se abstiene** cuando no existe una alternativa segura.")

    g = cobertura_global()
    comp = reporte("modelos_comparacion")
    gan = ganadores()

    def f1_test(objetivo: str) -> float:
        fila = comp[(comp["objetivo"] == objetivo) & (comp["modelo"] == gan[objetivo]["modelo"])
                    & (comp["conjunto"] == "test")]
        return float(fila["f1_macro"].iat[0])

    perfil = perfil_con_contexto()
    catalogo = perfil[perfil["fuente"] == "catalogo"]
    ac_mec, ac_fue = acierto_externo("mecanica")[0], acierto_externo("fuerza")[0]

    st.subheader("KPIs de negocio")
    c = st.columns(4)
    tarjeta(c[0], pct(g["cobertura_misma_familia"]),
            "Cobertura de sustitución: ≥ 3 alternativas seguras de la misma familia",
            f"✔ Cumple · meta {META_COBERTURA:.0f} %", "ok")
    tarjeta(c[1], f"{dec(f1_test('mecanica'))} / {dec(f1_test('fuerza'))}",
            "F1 macro en test · mecánica / fuerza", "✔ Cumple · meta 0,85", "ok")
    tarjeta(c[2], f"{ac_mec:.1f} / {ac_fue:.1f}".replace(".", ","),
            "% de acierto de lo etiquetado en el catálogo (~60 casos, ±8 pts)",
            "⚠ En el límite · meta 90 %", "warn")
    tarjeta(c[3], pct(g["sin_sustituto_anatomico"] / g["afectados"] * 100),
            "de los casos no tiene sustituto por anatomía", "→ El sistema se abstiene", "info")

    st.write("")
    c = st.columns(4)
    tarjeta(c[0], f"{pct(catalogo['mecanica'].notna().mean() * 100)}",
            "del catálogo con mecánica (antes 0 %)", "Fuerza: " + pct(catalogo["fuerza"].notna().mean() * 100), "info")
    tarjeta(c[1], f"{g['con_sin_equipamiento_antes']:.0f} → {g['con_sin_equipamiento_despues']:.0f} %",
            "listas con una opción sin equipamiento", "✔ Sesgo mitigado con una cuota", "ok")
    cob = reporte("sustitucion_cobertura")
    bajo = cob[(cob["metodo"] == METODO) & (cob["articulacion"] != "TODAS") & (cob["afectados"] >= 50)
               & (cob["cobertura_misma_familia"] < META_COBERTURA)].sort_values("cobertura_misma_familia")
    tarjeta(c[2], str(len(bajo)), "articulaciones bajo la meta: " +
            ", ".join(f"{a} {pct(v)}" for a, v in zip(bajo["articulacion"], bajo["cobertura_misma_familia"])),
            "✖ No cumple", "bad")
    tarjeta(c[3], "0", "fotogramas enviados a servidores externos", "✔ Cumple por diseño", "ok")

    lectura("La meta principal se cumple, pero no en todas partes: <b>hombro, rodilla y cadera</b>, "
            "las lesiones más comunes, quedan bajo el 85 %. Y en 4 de cada 10 casos la respuesta "
            "correcta es <b>no recomendar</b>: el sistema lo dice en vez de inventar.")


def vista_datos() -> None:
    encabezado("IE5 · CRISP-DM fases 2 y 3", "Datos: cinco fuentes, un vocabulario", "1:15–2:45")
    resumen = reporte("integracion_fuentes_resumen")
    fuentes = resumen[resumen["dimension"] == "fuente"].copy()
    fuentes["fuente"] = fuentes["grupo"].map(NOMBRE_FUENTE)
    total = int(resumen.loc[resumen["dimension"] == "total", "n_ejercicios"].iat[0])

    izq, der = st.columns([1.1, 1])
    with izq:
        fig = go.Figure(go.Waterfall(
            x=[*fuentes["fuente"], "Total"], measure=["relative"] * len(fuentes) + ["total"],
            y=[*fuentes["n_ejercicios"], 0], text=[*fuentes["n_ejercicios"].astype(int), total],
            textposition="outside", connector=dict(line=dict(color=GRIS)),
            increasing=dict(marker=dict(color=ACENTO)), totals=dict(marker=dict(color=ROJO))))
        fig.update_layout(title="Ejercicios únicos tras eliminar duplicados entre fuentes", showlegend=False)
        st.plotly_chart(estilo(fig), use_container_width=True)
        lectura(f"El catálogo es el esqueleto; free-exercise-db y wger suman solo los ejercicios que "
                f"no estaban. Quedan <b>{total:,}</b> ejercicios únicos.".replace(",", "."))
    with der:
        largo = fuentes.melt(id_vars="fuente", value_vars=["pct_con_nivel", "pct_con_mecanica", "pct_con_fuerza"],
                             var_name="etiqueta", value_name="pct")
        largo["etiqueta"] = largo["etiqueta"].map({"pct_con_nivel": "Nivel", "pct_con_mecanica": "Mecánica",
                                                   "pct_con_fuerza": "Fuerza"})
        fig = px.bar(largo, x="etiqueta", y="pct", color="fuente", barmode="group",
                     color_discrete_map=COLOR_FUENTE, text_auto=".0f",
                     labels={"pct": "% con etiqueta en la fuente", "etiqueta": ""})
        fig.update_layout(title="¿Quién trae etiquetas? (antes de modelar)")
        st.plotly_chart(estilo(fig), use_container_width=True)
        lectura("Solo <b>free-exercise-db</b> trae mecánica y fuerza: es la fuente de entrenamiento. "
                "El catálogo y wger casi no las tienen, y eso es lo que completan los modelos.")

    st.subheader("El hallazgo que cambió el objetivo")
    pares = niveles_dos_fuentes()
    kappa = cohen_kappa_score(pares["nivel"], pares["nivel_megagym"])
    acuerdo = (pares["nivel"] == pares["nivel_megagym"]).mean() * 100
    orden = ["beginner", "intermediate", "expert"]
    tabla = pd.crosstab(pares["nivel"], pares["nivel_megagym"]).reindex(index=orden, columns=orden, fill_value=0)
    izq, der = st.columns([1, 1.1])
    with izq:
        fig = px.imshow(tabla, text_auto=True, color_continuous_scale=["#F3F5F7", ACENTO], aspect="auto",
                        labels=dict(x="Nivel en megaGymDataset", y="Nivel en free-exercise-db", color="n"))
        fig.update_layout(title=f"El mismo ejercicio en dos fuentes (n = {len(pares)})", coloraxis_showscale=False)
        st.plotly_chart(estilo(fig, 340), use_container_width=True)
    with der:
        st.write("")
        c = st.columns(2)
        tarjeta(c[0], dec(kappa), "kappa de Cohen entre las dos fuentes", "= igual que al azar", "bad")
        tarjeta(c[1], pct(acuerdo), "de coincidencia (el azar daría ~45 %)", "Dificultad = opinión del curador", "warn")
        st.write("")
        lectura("La <b>dificultad no es una etiqueta objetiva</b>: depende de quién la puso. Por eso "
                "dejamos de predecirla y la reemplazamos por un <b>perfil de exigencia</b> (mecánica, "
                "fuerza y complejidad técnica). Es CRISP-DM iterando: la evaluación de los datos nos "
                "devolvió a redefinir el objetivo de negocio.")

    with st.expander("Sesgo de muestreo del catálogo (EV1)"):
        st.image(str(FIGURAS / "sesgo_muestreo.png"))
        lectura("61 % gimnasio y 9 % rehabilitación; cuatro músculos con menos de 10 ejercicios. "
                "Se mitiga con la cuota sin equipamiento y la abstención por músculo escaso.")


def vista_supervisado() -> None:
    encabezado("IE6 · CRISP-DM fase 4", "Modelos supervisados: completar etiquetas que faltan", "2:45–5:00")
    objetivo = st.radio("Objetivo", list(NOMBRE_OBJETIVO), format_func=NOMBRE_OBJETIVO.get, horizontal=True)
    gan = ganadores()[objetivo]
    comp = reporte("modelos_comparacion")
    comp = comp[comp["objetivo"] == objetivo]
    azar = float(comp.loc[comp["modelo"] == "azar_estratificado", "f1_macro"].iat[0])
    st.caption(f"Ganador: **{NOMBRE_MODELO[gan['modelo']]}** · umbral de abstención {dec(gan['umbral'])} · "
               "train/test separados por grupos de variantes (GroupShuffleSplit + GroupKFold)")

    izq, der = st.columns(2)
    with izq:
        modelos = comp[comp["modelo"] != "azar_estratificado"].copy()
        modelos["Modelo"] = modelos["modelo"].map(NOMBRE_MODELO)
        modelos["Conjunto"] = modelos["conjunto"].map({"test": "Test (free-exercise-db)",
                                                       "externa_catalogo": "Externa (catálogo)"})
        modelos["error"] = modelos["f1_ic95_sup"] - modelos["f1_macro"]
        modelos["error_menos"] = modelos["f1_macro"] - modelos["f1_ic95_inf"]
        fig = px.bar(modelos, x="Modelo", y="f1_macro", color="Conjunto", barmode="group", text_auto=".2f",
                     error_y="error", error_y_minus="error_menos",
                     color_discrete_sequence=[ACENTO, AMBAR], labels={"f1_macro": "F1 macro", "Modelo": ""})
        fig.add_hline(y=azar, line_dash="dash", line_color=ROJO,
                      annotation_text=f"azar {dec(azar)}", annotation_position="top left")
        fig.update_traces(textposition="inside", insidetextanchor="middle")
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(title="F1 macro con intervalo bootstrap 95 %")
        st.plotly_chart(estilo(fig), use_container_width=True)
    with der:
        imp = reporte("modelos_importancia")
        imp = imp[imp["objetivo"] == objetivo].sort_values("caida_media")
        fig = px.bar(imp, x="caida_media", y="familia", orientation="h", error_x="caida_sd",
                     color_discrete_sequence=[ACENTO],
                     labels={"caida_media": "Caída del F1 al desordenar la variable", "familia": ""})
        fig.update_layout(title="¿En qué se fija el modelo? (importancia por permutación)")
        st.plotly_chart(estilo(fig), use_container_width=True)

    textos = {
        "mecanica": "La mecánica se decide por la <b>anatomía</b> (sinergistas y articulaciones): es la "
                    "definición biomecánica de <i>compuesto</i>. El modelo aprendió biomecánica, no palabras. "
                    "Logística y Random Forest empatan dentro de un error estándar: gana la más simple.",
        "fuerza": "La fuerza se decide por las <b>instrucciones</b> (verbos <i>push</i>, <i>pull</i>, <i>hold</i>). "
                  "El Random Forest es el único que no cae en el catálogo; frente a la logística es un "
                  "empate dentro del ruido (CV repetida 5 × 5, p = 0,24).",
        "complejidad_tecnica": "Ninguna variable domina: la complejidad es un <b>juicio editorial</b>. "
                               "El Random Forest sobreajusta el estilo de la fuente (cae a 0,59 en el "
                               "catálogo), por eso gana la logística y casi siempre se abstiene.",
    }
    lectura(textos[objetivo])

    st.subheader("Abstención: solo etiqueta cuando está seguro")
    p = reporte("modelos_predicciones_test")
    p = p[(p["objetivo"] == objetivo) & (p["modelo"] == gan["modelo"]) & (p["conjunto"] == "externa_catalogo")]
    umbral = st.slider("Umbral de confianza", 0.40, 0.99, gan["umbral"], 0.01,
                       help="El umbral elegido es el menor que logra 95 % de acierto fuera de muestra en train.")
    curva = []
    for u in np.round(np.arange(0.40, 1.0, 0.01), 2):
        r = p["confianza"] >= u
        curva.append({"umbral": u, "Cobertura": r.mean() * 100,
                      "Acierto": (p.loc[r, "y_real"] == p.loc[r, "y_pred"]).mean() * 100 if r.any() else np.nan})
    curva = pd.DataFrame(curva).melt(id_vars="umbral", var_name="Serie", value_name="pct")
    izq, der = st.columns([1.6, 1])
    with izq:
        fig = px.line(curva, x="umbral", y="pct", color="Serie", color_discrete_sequence=[GRIS, ACENTO],
                      labels={"umbral": "Umbral de confianza", "pct": "% en el catálogo"})
        fig.add_vline(x=umbral, line_color=ROJO, line_dash="dot")
        fig.add_hline(y=90, line_color=GRIS, line_dash="dash", annotation_text="meta 90 %")
        fig.update_layout(title="Validación externa: cobertura vs. acierto")
        st.plotly_chart(estilo(fig, 340), use_container_width=True)
    with der:
        r = p["confianza"] >= umbral
        acierto = (p.loc[r, "y_real"] == p.loc[r, "y_pred"]).mean() * 100 if r.any() else float("nan")
        sin = (p["y_real"] == p["y_pred"]).mean() * 100
        st.write("")
        c = st.columns(2)
        tarjeta(c[0], pct(acierto), "acierto en lo que etiqueta", f"sin abstención: {pct(sin)}", "info")
        tarjeta(c[1], pct(r.mean() * 100), "de los ejercicios recibe etiqueta", f"n = {len(p)} del catálogo", "info")
        st.write("")
        lectura("Subir el umbral cambia cobertura por acierto. Una etiqueta falsa es el error grave "
                "(a un principiante le llega un ejercicio técnico); quedarse sin etiqueta es el leve.")

    with st.expander("Diagnóstico para las preguntas: fuga, ruido y cambio de dominio"):
        c = st.columns(3)
        fuga = reporte("modelos_control_fuga")
        fuga = fuga[fuga["objetivo"] == objetivo].iloc[0]
        tarjeta(c[0], f"{dec(fuga['f1_real_cv'])} → {dec(fuga['f1_objetivo_barajado'])}",
                "F1 con la etiqueta real → barajada", f"azar {dec(fuga['f1_azar'])}: sin fuga", "ok")
        adv = reporte("validacion_adversarial")
        adv = adv[(adv["fuente_prediccion"] == "catalogo") & (adv["familia"] == "Todas las variables")]
        tarjeta(c[1], dec(float(adv["auc"].iat[0]), 3), "AUC para distinguir catálogo de free-exercise-db",
                "Cambio de dominio: explica la caída", "warn")
        cv = reporte("modelos_cv_repetida")
        cv = cv[cv["objetivo"] == objetivo].pivot_table(index=["repeticion", "pliegue"], columns="modelo",
                                                         values="f1_macro")
        gana_rf = (cv["random_forest"] > cv["regresion_logistica"]).mean() * 100
        tarjeta(c[2], pct(gana_rf), "de 25 pliegues en que gana el Random Forest",
                f"diferencia media {dec((cv['random_forest'] - cv['regresion_logistica']).mean(), 3)}", "info")
        st.image(str(FIGURAS / "modelos_matrices_confusion.png"))


def vista_no_supervisado() -> None:
    encabezado("IE7 · CRISP-DM fase 4", "Grafo biomecánico y familias de movimiento", "5:00–7:15")
    comp = reporte("clustering_comparacion").set_index("metodo")
    cob = reporte("sustitucion_cobertura")
    cob = cob[cob["articulacion"] == "TODAS"].set_index("metodo")["cobertura_misma_familia"]
    tabla = pd.DataFrame({
        "Grupos": comp["n_grupos"].astype(int),
        "Modularidad ↑": comp["modularidad_proyeccion"].round(2),
        "Silhouette ↑": comp["silhouette_embedding"].round(2),
        "Davies-Bouldin ↓": comp["davies_bouldin_embedding"].round(2),
        "AMI con la fuente ↓": comp["ami_fuente"].round(2),
        "Cobertura de sustitución ↑": cob.reindex(comp.index).map(pct),
    }).rename(index=NOMBRE_METODO).rename_axis("Técnica")
    st.subheader("Dos técnicas, cada una gana en su vara")
    st.dataframe(tabla, use_container_width=True)
    izq, der = st.columns([1, 1.25])
    with izq:
        st.write("")
        lectura("Louvain gana en modularidad, K-Means en <i>silhouette</i>. Decidimos por el <b>KPI de "
                f"negocio</b>: {pct(cob[METODO])} contra {pct(cob['louvain'])} de cobertura, y las familias de "
                "Node2Vec mezclan las fuentes (AMI 0,08): aprendió biomecánica, no quién escribió la ficha.")
    with der:
        barrido = reporte("kmeans_barrido")
        fig = go.Figure()
        fig.add_scatter(x=barrido["k"], y=barrido["silhouette"], name="Silhouette", line=dict(color=ACENTO, width=3))
        fig.add_scatter(x=barrido["k"], y=barrido["estabilidad_ari"], name="Estabilidad (ARI)", yaxis="y2",
                        line=dict(color=AMBAR, width=2, dash="dot"))
        k = int(parametros()["grafo_biomecanico"]["kmeans"]["k"])
        fig.add_vline(x=k, line_color=ROJO, line_dash="dash", annotation_text=f"k = {k}")
        fig.update_layout(title="Elección de k: último antes de que caiga la silhouette",
                          xaxis_title="k (número de familias)", yaxis_title="Silhouette",
                          yaxis2=dict(overlaying="y", side="right", title="ARI", showgrid=False))
        st.plotly_chart(estilo(fig, 330), use_container_width=True)

    metodo = st.radio("Familias según", ["kmeans_node2vec", "louvain"], format_func=NOMBRE_METODO.get,
                      horizontal=True)
    fam = familias()
    columna = f"familia_{metodo}"
    izq, der = st.columns([1.35, 1])
    with izq:
        fig = px.scatter(fam.assign(Fuente=fam["fuente"].map(NOMBRE_FUENTE)), x="umap_x", y="umap_y",
                         color=columna, hover_name="nombre",
                         hover_data={"Fuente": True, "umap_x": False, "umap_y": False, columna: False},
                         color_discrete_sequence=px.colors.qualitative.Safe + px.colors.qualitative.Pastel,
                         labels={columna: "Familia"})
        fig.update_traces(marker=dict(size=5, opacity=.8))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title="Mapa UMAP de los embeddings Node2Vec (pasa el mouse por un punto)",
                          legend=dict(orientation="v", y=1, x=1.02, font=dict(size=11)))
        estilo(fig, 520).update_layout(legend=dict(orientation="v", yanchor="top", y=1, x=1.02))
        st.plotly_chart(fig, use_container_width=True)
    with der:
        perf = reporte("familias_perfiles")
        perf = perf[perf["metodo"] == metodo].sort_values("pct_catalogo")
        largo = perf.melt(id_vars="nombre", value_vars=["pct_catalogo", "pct_free_exercise_db", "pct_wger"],
                          var_name="Fuente", value_name="pct")
        largo["Fuente"] = largo["Fuente"].map({"pct_catalogo": "Catálogo",
                                               "pct_free_exercise_db": "free-exercise-db", "pct_wger": "wger"})
        fig = px.bar(largo, x="pct", y="nombre", color="Fuente", orientation="h", color_discrete_map=COLOR_FUENTE,
                     labels={"pct": "% de la familia", "nombre": ""})
        fig.update_layout(title="¿Las familias mezclan las fuentes?")
        st.plotly_chart(estilo(fig, 520), use_container_width=True)
    lectura("Cada punto es un ejercicio. Las familias son reconocibles (cuádriceps-glúteo de empuje, "
            "dorsal, pecho, bíceps, core, estiramientos…) y sus colores de fuente se mezclan. Con Louvain "
            "varias comunidades son casi de una sola fuente.")

    with st.expander("El grafo: vecindad de la sentadilla"):
        st.image(str(FIGURAS / "grafo_ego_sentadilla.png"))
        lectura("El agonista de la sentadilla es el glúteo, pero el cuádriceps (sinergista) cruza la "
                "rodilla. Por eso la carga articular cuenta agonistas <b>y</b> sinergistas.")


def vista_sustitucion() -> None:
    encabezado("IE8 · demo en vivo", "Sustitución con una articulación lesionada", "7:15–9:00")
    st.info("⚕️ Herramienta informativa: no reemplaza la evaluación de un profesional de la salud.")
    fam = familias()
    sust = sustituidor()

    casos = {"Sentadilla · rodilla": (SENTADILLA, "rodilla", "Cualquier usuario"),
             "Sentadilla · rodilla · principiante": (SENTADILLA, "rodilla", "Principiante"),
             "Extensión de piernas · rodilla": ("catalogo:0585", "rodilla", "Cualquier usuario"),
             "Press de banca · codo": ("catalogo:0025", "codo", "Cualquier usuario")}
    st.caption("Casos preparados")
    botones = st.columns(len(casos))
    for boton, (nombre, (ej, art, perfil)) in zip(botones, casos.items()):
        if boton.button(nombre, use_container_width=True):
            st.session_state.update(ejercicio=ej, articulacion=art, perfil=perfil)

    st.session_state.setdefault("ejercicio", SENTADILLA)
    st.session_state.setdefault("perfil", "Cualquier usuario")
    opciones = fam.sort_values("nombre")["ejercicio_id"].tolist()
    nombres = fam.set_index("ejercicio_id")
    c = st.columns([2.2, 1.2, 1.3, 0.9])
    ejercicio = c[0].selectbox("Ejercicio que no puede hacer", opciones, key="ejercicio",
                               format_func=lambda i: f"{nombres.at[i, 'nombre']} · {NOMBRE_FUENTE[nombres.at[i, 'fuente']]}")
    i = int(sust.indice[ejercicio])
    carga = sorted(sust.carga[i])
    if st.session_state.get("articulacion") not in carga:
        st.session_state["articulacion"] = "rodilla" if "rodilla" in carga else carga[0]
    articulacion = c[1].selectbox("Articulación lesionada", carga, key="articulacion")
    perfil = c[2].radio("Perfil", ["Cualquier usuario", "Principiante"], key="perfil", horizontal=True)
    cuota = c[3].toggle("Cuota sin equipo", value=True)

    original = sust.familias.iloc[i]
    st.markdown(f"**{original['nombre']}** · familia <span class='chip'>{original[f'familia_{METODO}']}</span> "
                f"agonistas: {', '.join(original['agonistas'])} · carga: {', '.join(carga)}",
                unsafe_allow_html=True)

    filtro = None
    if perfil == "Principiante":
        f = sust.familias
        filtro = ((f["complejidad_tecnica"] == "baja") & ~f["impacto"].astype(bool)
                  & (f["estabilidad"] != "inestable")).to_numpy()
        st.caption("Principiante = complejidad técnica **baja y conocida**, sin impacto y no inestable.")

    alternativas, motivo = sust.recomendar(ejercicio, articulacion, filtro=filtro, cuota=cuota)
    if alternativas.empty:
        st.warning(f"**El sistema se abstiene.** {MOTIVOS.get(motivo, motivo)}")
    else:
        if motivo:
            st.info(MOTIVOS.get(motivo, motivo))
        vista = pd.DataFrame({
            "Alternativa": alternativas["nombre"],
            "Movimiento": np.where(alternativas["misma_familia"], "✅ Mismo movimiento", "↪ Otro movimiento"),
            "Familia": alternativas[f"familia_{METODO}"],
            "Similitud": alternativas["similitud"],
            "Equipamiento": alternativas["familia_equipamiento"],
            "Fuerza": alternativas["fuerza"].fillna("—"),
            "Complejidad": alternativas["complejidad_tecnica"].fillna("sin etiqueta"),
            "Fuente": alternativas["fuente"].map(NOMBRE_FUENTE),
        })
        st.dataframe(vista, hide_index=True, use_container_width=True,
                     column_config={"Similitud": st.column_config.ProgressColumn(
                         "Similitud", min_value=0, max_value=1, format="%.2f")})
        st.caption("Ninguna alternativa carga la articulación vetada (filtro duro, verificado por un test). "
                   "Orden: primero la misma familia, después similitud de coseno entre embeddings.")

    st.subheader("¿Dónde se cumple la meta?")
    cob = reporte("sustitucion_cobertura")
    cob = cob[(cob["articulacion"] != "TODAS") & (cob["afectados"] >= 50) & (cob["metodo"] != "sin_familias")]
    orden = cob[cob["metodo"] == METODO].sort_values("cobertura_misma_familia")["articulacion"].tolist()
    cob = cob.assign(Método=cob["metodo"].map(NOMBRE_METODO))
    fig = px.bar(cob, x="cobertura_misma_familia", y="articulacion", color="Método", barmode="group",
                 orientation="h", category_orders={"articulacion": orden[::-1]},
                 color_discrete_map={NOMBRE_METODO[METODO]: ACENTO, "Louvain": GRIS},
                 labels={"cobertura_misma_familia": "% de ejercicios sustituibles con ≥ 3 alternativas de su familia",
                         "articulacion": ""})
    fig.add_vline(x=META_COBERTURA, line_dash="dash", line_color=ROJO, annotation_text="meta 85 %")
    st.plotly_chart(estilo(fig, 420), use_container_width=True)
    lectura("Rodilla, hombro y cadera quedan bajo la meta. En la rodilla la alternativa segura casi siempre "
            "<b>cambia de patrón</b> (de sentadilla a puente o bisagra de cadera): por eso la app debe avisar "
            "cuando la alternativa es «otro movimiento».")


def vista_sesgo() -> None:
    encabezado("IE8 · ética", "Sesgo después de modelar y recomendaciones", "9:00–10:00")
    ctx = reporte("sustitucion_contexto")
    ctx = ctx[ctx["metodo"] == METODO].assign(Contexto=lambda d: d["contexto"].map(NOMBRE_CONTEXTO))
    largo = ctx.melt(id_vars="Contexto", value_vars=["con_sin_equipamiento_antes", "con_sin_equipamiento_despues"],
                     var_name="Momento", value_name="pct")
    largo["Momento"] = largo["Momento"].map({"con_sin_equipamiento_antes": "Sin cuota",
                                             "con_sin_equipamiento_despues": "Con cuota"})
    izq, der = st.columns(2)
    with izq:
        fig = px.bar(largo, x="Contexto", y="pct", color="Momento", barmode="group", text_auto=".0f",
                     color_discrete_sequence=[GRIS, ACENTO],
                     labels={"pct": "% de listas con una opción sin equipamiento", "Contexto": ""})
        fig.update_layout(title="Mitigado: cuota de una alternativa sin equipamiento")
        st.plotly_chart(estilo(fig), use_container_width=True)
        lectura("Sin corrección, solo un tercio de las listas del gimnasio traía una opción para entrenar "
                "en casa. Con la cuota, casi todas.")
    with der:
        perfil = perfil_con_contexto()
        cobertura = (perfil.groupby("fuente")[list(NOMBRE_OBJETIVO)].agg(lambda s: s.notna().mean() * 100)
                     .rename(index=NOMBRE_FUENTE, columns=NOMBRE_OBJETIVO).reset_index()
                     .melt(id_vars="fuente", var_name="Etiqueta", value_name="pct"))
        fig = px.bar(cobertura, x="Etiqueta", y="pct", color="fuente", barmode="group", text_auto=".0f",
                     color_discrete_map=COLOR_FUENTE, labels={"pct": "% con etiqueta final", "Etiqueta": "",
                                                              "fuente": "Fuente"})
        fig.update_layout(title="Sesgo nuevo: wger queda peor etiquetada")
        st.plotly_chart(estilo(fig), use_container_width=True)
        rehab = perfil[perfil["contexto"] == "rehabilitacion_funcional"]["complejidad_tecnica"].notna().mean() * 100
        lectura(f"La abstención creó un sesgo: con instrucciones cortas, el modelo se abstiene más en wger y "
                f"el perfil de principiante casi no la recomienda. A cambio, rehabilitación pasó de 6,7 % a "
                f"<b>{pct(rehab)}</b> con complejidad conocida.")

    st.subheader("Recomendaciones de negocio")
    c = st.columns(3)
    for col, (titulo, texto) in zip(c, [
        ("1 · La abstención es una respuesta",
         "«No hay un sustituto seguro, consulta a un kinesiólogo» es parte del producto, no un error."),
        ("2 · Avisar cuando cambia el movimiento",
         "Mostrar la familia y marcar «otro movimiento»; exigir una similitud mínima antes de ofrecerla."),
        ("3 · Validar el mapeo con un profesional",
         "Toda la seguridad descansa en el mapeo músculo → articulación. Es el primer paso antes de producción."),
    ]):
        col.markdown(f'<div class="tarjeta"><b>{titulo}</b><div class="l">{texto}</div></div>',
                     unsafe_allow_html=True)
    st.write("")
    st.success("Todo es reproducible: `kedro run` regenera estas salidas con semillas fijas, y este "
               "dashboard solo las lee.")


VISTAS = {
    "1 · KPIs": vista_kpis,
    "2 · Datos": vista_datos,
    "3 · Supervisado": vista_supervisado,
    "4 · No supervisado": vista_no_supervisado,
    "5 · Sustitución en vivo": vista_sustitucion,
    "6 · Sesgo y cierre": vista_sesgo,
}

with st.sidebar:
    st.markdown("### Recomendador biomecánico")
    st.caption("EV2 · MLY1101 Machine Learning · Duoc UC")
    eleccion = st.radio("Vista", list(VISTAS), label_visibility="collapsed")
    st.divider()
    st.caption("Datos: salidas de `kedro run` en `data/08_reporting`, `data/07_model_output` y `models/`. "
               "Detalle en el README §11–§14.")

VISTAS[eleccion]()
