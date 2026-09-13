"""
Nodos del pipeline Data Enrichment — integración de megaGymDataset (Kaggle).

Aporta al catálogo base variables que su esquema no trae (README §3.4): nivel de
dificultad (``Level``), tipo de entrenamiento (``Type``) y valoración (``Rating``).
Las dos fuentes no comparten llave, así que la unión es por nombre de ejercicio
y **la calidad del emparejamiento se audita, no se asume**.

Origen: rama ``benja`` (commit ``e074742``). Correcciones respecto de esa versión:

1. **Fan-out del merge.** megaGymDataset trae 7 títulos duplicados; el merge sin
   deduplicar multiplicaba 8 ejercicios (1.332 filas en lugar de 1.324). Aquí se
   deduplica antes de unir y el merge valida ``many_to_one``.
2. **Falsos positivos del scorer.** ``WRatio`` con umbral 80 puntúa alto las
   coincidencias *parciales* (``"barbell seated overhead press"`` →
   ``"barbell roll-out"``): reportaba 99,62 % de cobertura, pero una fracción
   grande de los pares apunta a otra región anatómica. Se reemplaza por
   ``token_sort_ratio`` sobre nombres normalizados, solapamiento de tokens y
   coherencia de región muscular y equipamiento.
3. **Jerarquía muscular e id.** La rama reconstruía el catálogo desde el JSON con
   ``musculo_primario = muscle_group`` (el sinergista) y lo pasaba por CSV, que
   corrompe el ``id``: los dos defectos ya corregidos en README §3.3. Aquí se
   parte de ``intermediate_exercise_features``, que ya los tiene resueltos.
4. **Nulos honestos.** Los no emparejados quedan como nulos, no como el texto
   ``"No especificado"``: así ``rating`` sigue siendo numérico y no se confunde
   "sin match" con "match sin valoración".
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Normalización de nombres
# ─────────────────────────────────────────────────────────────────────────────
#: Prefijos de programa de entrenamiento que megaGymDataset antepone al nombre
#: del ejercicio ("30 Arms", "HM", "FYR2", "Holman"...). No describen el
#: movimiento: son la marca del plan del que se extrajo la ficha.
_PREFIJO_CODIGO = re.compile(
    r"^(?:30\s+(?:Arms|Shoulders|Back|Chest|Legs|Abs|Core)|FYR2?|HM|AM|UP|TBS"
    r"|UNS|UN|ACFT|KV|RG|BFR|PJR|AA|JM|CM|LD)\s+"
)
_PREFIJO_PROGRAMA = re.compile(
    r"^(?:holman|paul carter|dumbbell fix|total fitness|king maker|metaburn)\s+",
    re.IGNORECASE,
)
#: Variantes de escritura de un mismo término, llevadas a una forma única.
_SINONIMOS: list[tuple[str, str]] = [
    (r"\bpush[\s-]?ups?\b", "push up"),
    (r"\bpull[\s-]?ups?\b", "pull up"),
    (r"\bchin[\s-]?ups?\b", "chin up"),
    (r"\bsit[\s-]?ups?\b", "sit up"),
    (r"\bstep[\s-]?ups?\b", "step up"),
    (r"\bbody[\s-]?weight\b", "bodyweight"),
    (r"\b(?:e-?z|sz)[\s-]?(?:curl\s)?bar(?:bell)?\b", "ez bar"),
    (r"\b(?:stability|swiss|exercise)\s+ball\b", "exercise ball"),
    (r"\bskull[\s-]?crushers?\b", "skullcrusher"),
    (r"\blever(?:age)?\b", "lever"),
    (r"\bsmith machine\b", "smith"),
    (r"\brear lunge\b", "reverse lunge"),
]
#: Marcas que el catálogo base usa para distinguir fichas del mismo ejercicio
#: (modelo de la animación, ángulo de cámara, número de versión).
_RUIDO = [r"\((?:male|female)\)", r"\((?:back|side) pov\)", r"\bv\.\s*\d+\b",
          r"\b(?:male|female)\b"]


def normalizar_nombre(nombre: str) -> str:
    """Lleva un nombre de ejercicio a una forma canónica comparable.

    Quita prefijos de programa, marcas de variante y puntuación, unifica
    sinónimos de escritura y reduce plurales simples, de modo que
    ``"30 Arms Cable Rope Overhead Triceps Extension"`` y
    ``"cable rope overhead tricep extension"`` queden iguales.
    """
    texto = _PREFIJO_CODIGO.sub("", str(nombre).strip())
    texto = _PREFIJO_PROGRAMA.sub("", texto).lower()
    for patron in _RUIDO:
        texto = re.sub(patron, " ", texto)
    for patron, reemplazo in _SINONIMOS:
        texto = re.sub(patron, reemplazo, texto)
    texto = re.sub(r"[^a-z0-9 ]+", " ", texto)
    tokens = [
        t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
        for t in texto.split()
    ]
    return " ".join(tokens)


def _tokens_equivalentes(a: str, b: str) -> bool:
    """Dos tokens valen lo mismo si uno es prefijo del otro o casi idénticos
    (``clap``/``clapping``, ``scapula``/``scapular``, ``flye``/``fly``)."""
    if a == b:
        return True
    corto, largo = sorted((a, b), key=len)
    return (len(corto) >= 3 and largo.startswith(corto)) or fuzz.ratio(a, b) >= 85


def jaccard_tokens(a: str, b: str) -> float:
    """Solapamiento de tokens entre dos nombres ya normalizados (0 a 1).

    Complementa a ``token_sort_ratio``, que compara caracteres: en nombres
    cortos, cambiar una palabra entera apenas mueve la similitud de caracteres
    (``"smith squat"`` vs ``"sit squat"`` = 90), pero sí el solapamiento de
    tokens (1 de 3).
    """
    tokens_a, tokens_b = a.split(), b.split()
    usados: set[int] = set()
    comunes = 0
    for t in tokens_a:
        for k, u in enumerate(tokens_b):
            if k not in usados and _tokens_equivalentes(t, u):
                usados.add(k)
                comunes += 1
                break
    union = len(set(tokens_a)) + len(set(tokens_b)) - comunes
    return comunes / union if union else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Reglas de coherencia entre fuentes
# ─────────────────────────────────────────────────────────────────────────────
#: Región anatómica de cada ``BodyPart`` de megaGymDataset.
REGION_MEGAGYM: dict[str, str] = {
    "Abdominals": "core", "Lower Back": "espalda_baja",
    "Quadriceps": "pierna", "Hamstrings": "pierna", "Glutes": "pierna",
    "Abductors": "pierna", "Adductors": "pierna", "Calves": "pantorrilla",
    "Chest": "pecho", "Shoulders": "hombro",
    "Lats": "espalda_alta", "Middle Back": "espalda_alta", "Traps": "espalda_alta",
    "Biceps": "biceps", "Triceps": "triceps", "Forearms": "antebrazo",
    "Neck": "cuello",
}
#: Regiones admisibles para cada músculo primario (``target``) del catálogo.
#: Es deliberadamente tolerante entre regiones vecinas, porque las dos fuentes
#: etiquetan distinto el mismo ejercicio (un peso muerto es ``glutes`` en una y
#: ``Hamstrings`` en otra). ``cardiovascular system`` no tiene región: no se
#: contrasta.
REGIONES_POR_MUSCULO: dict[str, set[str]] = {
    "abs": {"core"}, "spine": {"espalda_baja", "core"},
    "quads": {"pierna"}, "hamstrings": {"pierna", "espalda_baja"},
    "glutes": {"pierna", "espalda_baja"}, "abductors": {"pierna"},
    "adductors": {"pierna"}, "calves": {"pantorrilla"},
    "pectorals": {"pecho"}, "serratus anterior": {"pecho", "core", "hombro"},
    "delts": {"hombro"}, "traps": {"espalda_alta", "hombro"},
    "levator scapulae": {"espalda_alta", "cuello"},
    "lats": {"espalda_alta"}, "upper back": {"espalda_alta", "hombro"},
    "biceps": {"biceps", "antebrazo"}, "triceps": {"triceps"},
    "forearms": {"antebrazo"},
}
#: Valores de ``Equipment`` de megaGymDataset compatibles con cada
#: equipamiento del catálogo. Un nulo en megaGymDataset no contradice nada.
EQUIPOS_COMPATIBLES: dict[str, set[str]] = {
    "body weight": {"Body Only", "Other"},
    "weighted": {"Body Only", "Other", "Dumbbell", "Kettlebells"},
    "assisted": {"Machine", "Body Only", "Other", "Bands"},
    "dumbbell": {"Dumbbell"},
    "barbell": {"Barbell"}, "olympic barbell": {"Barbell"},
    "trap bar": {"Barbell", "Other"}, "ez barbell": {"E-Z Curl Bar", "Barbell"},
    "cable": {"Cable"}, "rope": {"Cable", "Other"},
    "leverage machine": {"Machine"}, "sled machine": {"Machine"},
    "smith machine": {"Machine", "Barbell"},
    "band": {"Bands"}, "resistance band": {"Bands"},
    "kettlebell": {"Kettlebells"}, "medicine ball": {"Medicine Ball"},
    "stability ball": {"Exercise Ball", "Body Only", "Other"},
    "bosu ball": {"Exercise Ball", "Body Only", "Other"},
    "roller": {"Foam Roll", "Other"}, "wheel roller": {"Other", "Body Only"},
}
#: Máquinas de cardio e implementos poco frecuentes (``tire``, ``hammer``...).
EQUIPOS_POR_DEFECTO: set[str] = {"Machine", "Other"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Limpieza de megaGymDataset
# ─────────────────────────────────────────────────────────────────────────────
def preparar_megagym(raw_megagym: pd.DataFrame) -> pd.DataFrame:
    """Limpia megaGymDataset y lo deja a grano 1 fila = 1 ejercicio.

    - Descarta ``Desc`` y ``RatingDesc``: la primera es texto extraído de
      terceros (ver README §5.2) y la segunda solo toma el valor ``"Average"``.
    - ``Rating == 0`` es un marcador de "sin valoraciones", no una nota: pasa a
      nulo.
    - ``nivel_curado`` marca las fichas cuyo ``Level`` es informativo. En las
      filas sin ``Rating`` el nivel es casi siempre ``Intermediate`` (valor de
      relleno de la fuente), así que no debe leerse como dificultad real.
    - Deduplica por nombre normalizado. Cuando varias fichas colapsan al mismo
      nombre (``"Cable cross-over"`` y ``"UP Cable Cross-Over"``), se prefiere
      la curada y, entre ellas, la de título más corto (la versión sin prefijo
      de programa).

    Args:
        raw_megagym: CSV crudo de Kaggle.

    Returns:
        DataFrame limpio y deduplicado por ``titulo_norm``.
    """
    df = raw_megagym.drop(columns=["Desc", "RatingDesc"], errors="ignore").copy()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.rename(columns={
        "Title": "titulo", "Type": "tipo", "BodyPart": "parte_cuerpo",
        "Equipment": "equipamiento", "Level": "nivel", "Rating": "rating",
    })
    df["titulo"] = df["titulo"].str.strip()
    df["titulo_norm"] = df["titulo"].map(normalizar_nombre)
    df["nivel_curado"] = df["rating"].notna()
    df["rating"] = df["rating"].where(df["rating"] > 0)

    repetidos = df["titulo"].str.lower()
    n_titulos_repetidos = int(repetidos[repetidos.duplicated()].nunique())
    n_crudo = len(df)
    df = (
        df.assign(_largo=df["titulo"].str.len())
        .sort_values(["nivel_curado", "_largo"], ascending=[False, True], kind="stable")
        .drop_duplicates(subset="titulo_norm")
        .drop(columns="_largo")
        .sort_index()
        .reset_index(drop=True)
    )

    logger.info(
        "megaGymDataset: %d filas crudas, %d títulos repetidos literalmente; "
        "%d fichas únicas tras normalizar.",
        n_crudo, n_titulos_repetidos, len(df),
    )
    return df[["titulo", "titulo_norm", "tipo", "parte_cuerpo", "equipamiento",
               "nivel", "nivel_curado", "rating"]]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Emparejamiento auditado
# ─────────────────────────────────────────────────────────────────────────────
def emparejar_con_megagym(
    exercise_features: pd.DataFrame,
    megagym: pd.DataFrame,
    parametros: dict[str, Any],
) -> pd.DataFrame:
    """Busca, para cada ejercicio del catálogo, su ficha en megaGymDataset.

    Regla de aceptación, en orden:

    1. **Exacto** — el nombre normalizado coincide con un título normalizado.
       Se acepta sin más: dos nombres idénticos son la misma ficha aunque las
       fuentes etiqueten distinto el músculo (``"diamond push-up"`` es
       ``triceps`` en el catálogo y ``Chest`` en megaGymDataset).
    2. **Difuso** — entre las fichas con región muscular y equipamiento
       coherentes, la de mayor ``token_sort_ratio`` con similitud ≥
       ``umbral_similitud`` y solapamiento de tokens > ``umbral_jaccard``.
    3. **Sin match** — ninguna ficha cumple. Se registra igual el mejor
       candidato por similitud, para que el rechazo sea auditable.

    Args:
        exercise_features: Tabla analítica a grano ejercicio.
        megagym: Salida de ``preparar_megagym``.
        parametros: ``umbral_similitud`` (0-100) y ``umbral_jaccard`` (0-1).

    Returns:
        Una fila por ejercicio con el candidato, sus puntajes, las
        verificaciones de coherencia y el método de aceptación.
    """
    umbral_sim = float(parametros["umbral_similitud"])
    umbral_jac = float(parametros["umbral_jaccard"])

    catalogo = exercise_features.reset_index(drop=True)
    nombres = catalogo["name"].map(normalizar_nombre)
    titulos = megagym["titulo_norm"].tolist()
    similitud = process.cdist(nombres.tolist(), titulos,
                              scorer=fuzz.token_sort_ratio, workers=-1)

    region_mega = megagym["parte_cuerpo"].map(REGION_MEGAGYM).to_numpy(dtype=object)
    equipo_mega = megagym["equipamiento"].to_numpy(dtype=object)
    equipo_nulo = megagym["equipamiento"].isna().to_numpy()

    filas = []
    for i, ejercicio in catalogo.iterrows():
        puntajes = similitud[i]
        regiones = REGIONES_POR_MUSCULO.get(ejercicio["musculo_primario"])
        region_ok = (np.ones(len(megagym), dtype=bool) if regiones is None
                     else np.isin(region_mega, list(regiones)))
        equipos = EQUIPOS_COMPATIBLES.get(ejercicio["equipment"], EQUIPOS_POR_DEFECTO)
        equipo_ok = equipo_nulo | np.isin(equipo_mega, list(equipos))

        exactos = np.flatnonzero(puntajes >= 100)
        elegido, metodo, jaccard = None, "sin_match", np.nan
        if exactos.size:
            elegido, metodo, jaccard = int(exactos[0]), "exacto", 1.0
        else:
            candidatos = np.flatnonzero(region_ok & equipo_ok & (puntajes >= umbral_sim))
            for j in candidatos[np.argsort(-puntajes[candidatos], kind="stable")]:
                jac = jaccard_tokens(nombres[i], titulos[j])
                if jac > umbral_jac:
                    elegido, metodo, jaccard = int(j), "difuso", jac
                    break
        if elegido is None:
            elegido = int(puntajes.argmax())
            jaccard = jaccard_tokens(nombres[i], titulos[elegido])
        j = elegido

        filas.append({
            "id": ejercicio["id"],
            "name": ejercicio["name"],
            "nombre_norm": nombres[i],
            "candidato_titulo": megagym.at[j, "titulo"],
            "candidato_titulo_norm": titulos[j],
            "similitud": round(float(puntajes[j]), 2),
            "jaccard": round(jaccard, 3),
            "region_coherente": bool(region_ok[j]) if regiones is not None else pd.NA,
            "equipo_coherente": bool(equipo_ok[j]),
            "metodo": metodo,
            "aceptado": metodo != "sin_match",
        })

    reporte = pd.DataFrame(filas)
    conteo = reporte["metodo"].value_counts()
    logger.info(
        "Emparejamiento con megaGymDataset: %d exactos, %d difusos, %d sin match "
        "(cobertura %.1f %%).",
        conteo.get("exacto", 0), conteo.get("difuso", 0), conteo.get("sin_match", 0),
        reporte["aceptado"].mean() * 100,
    )
    return reporte


# ─────────────────────────────────────────────────────────────────────────────
# 3. Unión a la tabla analítica
# ─────────────────────────────────────────────────────────────────────────────
def enrich_with_external_metadata(
    exercise_features: pd.DataFrame,
    megagym: pd.DataFrame,
    emparejamiento: pd.DataFrame,
) -> pd.DataFrame:
    """Une ``nivel``, ``tipo`` y ``rating`` de megaGymDataset al catálogo.

    Solo se transfieren atributos de los pares aceptados. El grano se verifica
    en ambos extremos: cada ficha de megaGymDataset es única
    (``validate="many_to_one"``) y la salida conserva exactamente una fila por
    ejercicio.

    Args:
        exercise_features: Tabla analítica a grano ejercicio.
        megagym: Salida de ``preparar_megagym``.
        emparejamiento: Salida de ``emparejar_con_megagym``.

    Returns:
        ``exercise_features`` con las columnas ``megagym_titulo``,
        ``match_metodo``, ``match_similitud``, ``nivel``, ``nivel_curado``,
        ``tipo`` y ``rating``.
    """
    aceptados = emparejamiento.loc[
        emparejamiento["aceptado"], ["id", "candidato_titulo_norm", "metodo", "similitud"]
    ]
    atributos = megagym[["titulo_norm", "titulo", "nivel", "nivel_curado", "tipo", "rating"]]
    enlace = aceptados.merge(
        atributos, left_on="candidato_titulo_norm", right_on="titulo_norm",
        how="left", validate="many_to_one",
    )

    enriquecida = exercise_features.merge(
        enlace.drop(columns=["candidato_titulo_norm", "titulo_norm"]).rename(columns={
            "titulo": "megagym_titulo", "metodo": "match_metodo",
            "similitud": "match_similitud",
        }),
        on="id", how="left", validate="one_to_one",
    )
    enriquecida["match_metodo"] = enriquecida["match_metodo"].fillna("sin_match")
    enriquecida["nivel_curado"] = enriquecida["nivel_curado"].astype("boolean").fillna(False)

    if len(enriquecida) != len(exercise_features):
        raise ValueError(
            f"El enriquecimiento cambió el grano: {len(exercise_features)} → "
            f"{len(enriquecida)} filas."
        )

    logger.info(
        "Tabla enriquecida: %d filas; nivel disponible en %d, nivel curado en %d.",
        len(enriquecida), int(enriquecida["nivel"].notna().sum()),
        int(enriquecida["nivel_curado"].sum()),
    )
    return enriquecida
