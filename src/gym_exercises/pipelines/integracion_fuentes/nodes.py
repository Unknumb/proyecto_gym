"""
Nodos del pipeline Integración de Fuentes — catálogo + free-exercise-db + wger.

Prepara la base del grafo biomecánico (README §7) y de los modelos de la EV2.
Resuelve tres problemas que ningún dataset resuelve por separado:

1. **Etiquetas faltantes.** El catálogo no trae mecánica, tipo de fuerza ni una
   dificultad fiable (README §3.4, §3.5). free-exercise-db trae las tres. Se
   transfieren a los ejercicios del catálogo con par, y en los demás quedan como
   objetivo de los modelos supervisados.
2. **Tres vocabularios musculares distintos** (``lats`` / ``latissimus dorsi`` /
   ``Latissimus dorsi``). Si no se unifican, el grafo tendría tres nodos para el
   mismo músculo y dos ejercicios equivalentes nunca compartirían vecindario. Todo
   se lleva al ``musculo_normalizado`` del mapeo articular, que ya tiene
   articulación confirmada.
3. **Ejercicios repetidos entre fuentes.** El mismo *barbell curl* está en las
   tres. Se emparejan con la misma regla estricta que megaGymDataset (§3.5), y
   solo los que no tienen par entran al grafo como nodos nuevos.

**Regiones compuestas** (``back``, ``chest``, ``shoulders``, ``core``...,
decisión pendiente en README §6.2). No se excluyen ni se reparten en sus
componentes: quedan como nodo de región, unido a sus componentes por una relación
*parte de*. Solo reciben arista hacia una articulación cuando **todos** sus
componentes la comparten (``chest`` → hombro). ``back`` mezcla hombro,
escapulotorácica y columna, así que no recibe ninguna. Es la única regla que
conserva el ejercicio en el grafo sin afirmar algo que la fuente no dice.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from gym_exercises.pipelines.data_enrichment.nodes import jaccard_tokens, normalizar_nombre

logger = logging.getLogger(__name__)

#: Columnas del esquema común a las tres fuentes (grano 1 fila = 1 ejercicio).
COLUMNAS_UNIFICADAS = [
    "ejercicio_id", "fuente", "id_fuente", "nombre", "nombre_norm",
    "categoria_fuente", "equipamiento_fuente", "familia_equipamiento", "contexto",
    "musculos_primarios", "musculos_secundarios", "instrucciones_en",
    "nivel", "mecanica", "fuerza", "es_estiramiento", "grupo_variacion",
    "licencia", "autor",
]
#: Si wger lista varios equipos, manda el que más determina el ejercicio.
PRIORIDAD_FAMILIA = ["maquina", "cable", "peso libre", "banda elastica",
                     "balon o rodillo", "asistido", "maquina de cardio", "otro",
                     "peso corporal"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ontología muscular
# ─────────────────────────────────────────────────────────────────────────────
def construir_ontologia_muscular(
    mapeo_catalogo: pd.DataFrame,
    vocabulario_externo: pd.DataFrame,
    articulaciones: pd.DataFrame,
    regiones: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une los vocabularios y deriva las aristas músculo → articulación.

    Args:
        mapeo_catalogo: ``musculo_articulacion_manual.csv`` (catálogo base).
        vocabulario_externo: equivalencias de free-exercise-db y wger.
        articulaciones: articulación(es) canónica(s) de cada músculo.
        regiones: componentes de cada región compuesta.

    Returns:
        ``vocabulario`` — (fuente, musculo_fuente) → musculo canónico, con su
        tipo (``musculo`` o ``region``). Las etiquetas que no son músculo ni
        región (``cardiovascular system``, ``feet``...) no aparecen: se excluyen.

        ``musculo_articulacion`` — aristas (musculo, articulacion, origen), con
        ``origen`` = ``directo`` o ``consenso_region``.
    """
    musculos = set(articulaciones["musculo"])
    faltantes = set(regiones["musculo_componente"]) - musculos
    if faltantes:
        raise ValueError(f"Componentes de región sin articulación canónica: {sorted(faltantes)}")

    catalogo = mapeo_catalogo[["musculo_crudo", "musculo_normalizado"]].rename(
        columns={"musculo_crudo": "musculo_fuente", "musculo_normalizado": "musculo"})
    catalogo.insert(0, "fuente", "catalogo")
    vocabulario = pd.concat(
        [catalogo, vocabulario_externo[["fuente", "musculo_fuente", "musculo"]]],
        ignore_index=True,
    )
    vocabulario["tipo"] = np.select(
        [vocabulario["musculo"].isin(musculos),
         vocabulario["musculo"].isin(set(regiones["region"]))],
        ["musculo", "region"], default="excluido",
    )
    excluidos = vocabulario.loc[vocabulario["tipo"] == "excluido"]
    externos_sin_destino = excluidos.loc[excluidos["fuente"] != "catalogo"]
    if len(externos_sin_destino):
        raise ValueError("Músculos externos sin destino en la ontología: "
                         f"{externos_sin_destino['musculo_fuente'].tolist()}")
    logger.info("Ontología: se excluyen %d etiquetas del catálogo que no son músculo "
                "ni región: %s", len(excluidos), sorted(excluidos["musculo_fuente"]))
    vocabulario = vocabulario.loc[vocabulario["tipo"] != "excluido"].reset_index(drop=True)

    directas = articulaciones[["musculo", "articulacion"]].assign(origen="directo")
    por_region = regiones.merge(articulaciones, left_on="musculo_componente",
                                right_on="musculo")
    n_componentes = regiones.groupby("region")["musculo_componente"].nunique()
    comparten = por_region.groupby(["region", "articulacion"])["musculo_componente"].nunique()
    consenso = (comparten[comparten == comparten.index.get_level_values("region")
                          .map(n_componentes).to_numpy()]
                .reset_index()[["region", "articulacion"]]
                .rename(columns={"region": "musculo"})
                .assign(origen="consenso_region"))
    sin_consenso = sorted(set(n_componentes.index) - set(consenso["musculo"]))
    logger.info("Regiones con articulación por consenso: %s; sin consenso: %s",
                dict(consenso[["musculo", "articulacion"]].values), sin_consenso)

    musculo_articulacion = pd.concat([directas, consenso], ignore_index=True)
    return vocabulario, musculo_articulacion


def _traducir_musculos(listas: pd.Series, fuente: str,
                       vocabulario: pd.DataFrame) -> pd.Series:
    """Lleva cada lista de músculos de una fuente al vocabulario canónico,
    sin duplicados y conservando el orden."""
    equivalencias = dict(
        vocabulario.loc[vocabulario["fuente"] == fuente, ["musculo_fuente", "musculo"]].values)
    return listas.map(lambda musculos: list(dict.fromkeys(
        equivalencias[m] for m in musculos if m in equivalencias)))


def _familias(equipamiento: pd.DataFrame, fuente: str) -> dict[str, tuple[str, str]]:
    tabla = equipamiento.loc[equipamiento["fuente"] == fuente]
    return {fila.equipamiento_fuente: (fila.familia, fila.contexto)
            for fila in tabla.itertuples()}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preparación de cada fuente al esquema común
# ─────────────────────────────────────────────────────────────────────────────
def preparar_catalogo(
    raw_catalogo: pd.DataFrame,
    features_enriquecidas: pd.DataFrame,
    vocabulario: pd.DataFrame,
    equipamiento: pd.DataFrame,
) -> pd.DataFrame:
    """Catálogo base al esquema común.

    Toma ``target`` como agonista (README §3.3) y ``secondary_muscles`` como
    sinergistas; las instrucciones en inglés, para compartir idioma con las
    otras dos fuentes en el modelo de texto; y el nivel de megaGymDataset solo
    si es curado (``nivel_curado``, §3.5).
    """
    familias = _familias(equipamiento, "catalogo")
    catalogo = raw_catalogo.merge(
        features_enriquecidas[["id", "nivel", "nivel_curado"]], on="id",
        how="left", validate="one_to_one")

    primarios = _traducir_musculos(catalogo["target"].map(lambda t: [t]), "catalogo", vocabulario)
    secundarios = _traducir_musculos(catalogo["secondary_muscles"], "catalogo", vocabulario)
    secundarios = [[m for m in s if m not in p] for p, s in zip(primarios, secundarios)]

    unificado = pd.DataFrame({
        "ejercicio_id": "catalogo:" + catalogo["id"],
        "fuente": "catalogo",
        "id_fuente": catalogo["id"],
        "nombre": catalogo["name"],
        "categoria_fuente": catalogo["body_part"],
        "equipamiento_fuente": catalogo["equipment"],
        "familia_equipamiento": catalogo["equipment"].map(lambda e: familias[e][0]),
        "contexto": catalogo["equipment"].map(lambda e: familias[e][1]),
        "musculos_primarios": primarios,
        "musculos_secundarios": secundarios,
        "instrucciones_en": catalogo["instruction_steps"].map(lambda s: " ".join(s["en"])),
        "nivel": catalogo["nivel"].where(catalogo["nivel_curado"].fillna(False)).str.lower(),
        "mecanica": None,
        "fuerza": None,
        "es_estiramiento": catalogo["name"].str.contains(r"\bstretch", case=False),
        "grupo_variacion": None,
        "licencia": "MIT",
        "autor": catalogo["attribution"].map(lambda _: "Hasan Emir Yıldırım (datos)"),
    })
    return _cerrar_esquema(unificado)


def preparar_free_exercise_db(
    raw_fedb: pd.DataFrame,
    vocabulario: pd.DataFrame,
    equipamiento: pd.DataFrame,
) -> pd.DataFrame:
    """free-exercise-db al esquema común. Es la fuente de las etiquetas."""
    familias = _familias(equipamiento, "free_exercise_db")
    equipo = raw_fedb["equipment"].fillna("other")
    unificado = pd.DataFrame({
        "ejercicio_id": "free_exercise_db:" + raw_fedb["id"],
        "fuente": "free_exercise_db",
        "id_fuente": raw_fedb["id"],
        "nombre": raw_fedb["name"].str.strip(),
        "categoria_fuente": raw_fedb["category"],
        "equipamiento_fuente": equipo,
        "familia_equipamiento": equipo.map(lambda e: familias[e][0]),
        "contexto": equipo.map(lambda e: familias[e][1]),
        "musculos_primarios": _traducir_musculos(raw_fedb["primaryMuscles"],
                                                 "free_exercise_db", vocabulario),
        "musculos_secundarios": _traducir_musculos(raw_fedb["secondaryMuscles"],
                                                   "free_exercise_db", vocabulario),
        "instrucciones_en": raw_fedb["instructions"].map(" ".join),
        "nivel": raw_fedb["level"],
        "mecanica": raw_fedb["mechanic"],
        "fuerza": raw_fedb["force"],
        "es_estiramiento": raw_fedb["category"] == "stretching",
        "grupo_variacion": None,
        "licencia": "The Unlicense",
        "autor": "yuhonas/free-exercise-db",
    })
    return _cerrar_esquema(unificado)


def preparar_wger(
    raw_wger: pd.DataFrame,
    vocabulario: pd.DataFrame,
    equipamiento: pd.DataFrame,
) -> pd.DataFrame:
    """wger al esquema común.

    wger es colaborativo y trae fichas repetidas (``"Side split squats left"`` /
    ``"... right"`` quedan distintas, pero hay títulos idénticos). Se deduplica
    por nombre normalizado, conservando la ficha más antigua. Los equipos
    ``accesorio`` (banco) no definen el ejercicio y se ignoran; si solo hay
    accesorios, el ejercicio es de peso corporal.
    """
    familias = _familias(equipamiento, "wger")
    orden = {familia: k for k, familia in enumerate(PRIORIDAD_FAMILIA)}

    def _equipo_principal(equipos: list[str]) -> tuple[str, str, str]:
        if len(equipos) == 0:
            return "sin dato", "otro", "cardio_otros"
        con_carga = [e for e in equipos if familias[e][0] != "accesorio"]
        if not con_carga:
            return ", ".join(equipos), "peso corporal", "peso_corporal"
        principal = min(con_carga, key=lambda e: orden[familias[e][0]])
        return ", ".join(equipos), *familias[principal]

    equipo = raw_wger["equipamiento"].map(_equipo_principal)
    unificado = pd.DataFrame({
        "ejercicio_id": "wger:" + raw_wger["id"].astype(str),
        "fuente": "wger",
        "id_fuente": raw_wger["id"].astype(str),
        "nombre": raw_wger["nombre"],
        "categoria_fuente": raw_wger["categoria"],
        "equipamiento_fuente": equipo.str[0],
        "familia_equipamiento": equipo.str[1],
        "contexto": equipo.str[2],
        "musculos_primarios": _traducir_musculos(raw_wger["musculos"], "wger", vocabulario),
        "musculos_secundarios": _traducir_musculos(raw_wger["musculos_secundarios"],
                                                   "wger", vocabulario),
        "instrucciones_en": raw_wger["descripcion"],
        "nivel": None,
        "mecanica": None,
        "fuerza": None,
        "es_estiramiento": raw_wger["nombre"].str.contains(r"\bstretch", case=False),
        "grupo_variacion": raw_wger["grupo_variacion"],
        "licencia": raw_wger["licencia"],
        "autor": raw_wger["autor"],
    })
    unificado = _cerrar_esquema(unificado)
    n_antes = len(unificado)
    unificado = (unificado.sort_values("id_fuente", key=lambda s: s.astype(int))
                 .drop_duplicates("nombre_norm").sort_index().reset_index(drop=True))
    logger.info("wger: %d fichas, %d tras deduplicar por nombre normalizado.",
                n_antes, len(unificado))
    return unificado


def _cerrar_esquema(df: pd.DataFrame) -> pd.DataFrame:
    df["nombre_norm"] = df["nombre"].map(normalizar_nombre)
    df["musculos_secundarios"] = [
        [m for m in s if m not in p]
        for p, s in zip(df["musculos_primarios"], df["musculos_secundarios"])
    ]
    for columna in ["nivel", "mecanica", "fuerza", "grupo_variacion"]:
        df[columna] = df[columna].astype("string")
    df["es_estiramiento"] = df["es_estiramiento"].astype(bool)
    if df["ejercicio_id"].duplicated().any():
        raise ValueError("ejercicio_id duplicado en " + df["fuente"].iat[0])
    return df[COLUMNAS_UNIFICADAS]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Emparejamiento entre fuentes
# ─────────────────────────────────────────────────────────────────────────────
def _articulaciones_tolerantes(
    musculo_articulacion: pd.DataFrame, regiones: pd.DataFrame,
) -> dict[str, set[str]]:
    """Articulaciones de cada músculo para *comparar* ejercicios.

    A diferencia de las aristas del grafo, aquí una región sin consenso
    (``back``) cuenta con la unión de las articulaciones de sus componentes: para
    decidir si dos nombres parecidos son el mismo ejercicio basta con que no
    apunten a zonas del cuerpo incompatibles.
    """
    por_musculo = musculo_articulacion.groupby("musculo")["articulacion"].agg(set).to_dict()
    for region, componentes in regiones.groupby("region")["musculo_componente"]:
        por_musculo[region] = set().union(*(por_musculo[c] for c in componentes))
    return por_musculo


#: Modificadores que, si uno de los nombres trae uno y el otro su opuesto,
#: describen ejercicios distintos aunque el resto del nombre coincida: cambian el
#: plano de movimiento (abducción/aducción), el fascículo cargado
#: (inclinado/declinado, frontal/posterior) o la posición del cuerpo.
MODIFICADORES_OPUESTOS: list[tuple[str, str]] = [
    ("abduct", "adduct"), ("incline", "decline"), ("internal", "external"),
    ("pronat", "supinat"), ("front", "rear"), ("seated", "standing"),
    ("lying", "standing"), ("lying", "seated"),
]
#: Modificadores que cambian el músculo cargado aunque el otro nombre no traiga
#: su opuesto: ``lateral raise`` (deltoides medio) no es ``rear lateral raise``
#: (posterior), ni ``curl`` (bíceps) es ``reverse curl`` (braquiorradial).
MODIFICADORES_EXCLUSIVOS: list[str] = ["rear", "reverse"]


def _modificadores_contradicen(a: str, b: str) -> bool:
    """``True`` si dos nombres normalizados traen modificadores opuestos, o si
    solo uno trae un modificador exclusivo."""
    tokens_a, tokens_b = a.split(), b.split()

    def _tiene(tokens: list[str], raiz: str) -> bool:
        return any(t.startswith(raiz) for t in tokens)

    def _solo(tokens: list[str], raiz: str, opuesta: str) -> bool:
        return _tiene(tokens, raiz) and not _tiene(tokens, opuesta)

    if any(_tiene(tokens_a, m) != _tiene(tokens_b, m) for m in MODIFICADORES_EXCLUSIVOS):
        return True
    return any(
        (_solo(tokens_a, x, y) and _solo(tokens_b, y, x))
        or (_solo(tokens_a, y, x) and _solo(tokens_b, x, y))
        for x, y in MODIFICADORES_OPUESTOS
    )


def emparejar_fuentes(
    base: pd.DataFrame,
    externa: pd.DataFrame,
    musculo_articulacion: pd.DataFrame,
    regiones: pd.DataFrame,
    parametros: dict[str, Any],
) -> pd.DataFrame:
    """Busca, para cada ejercicio de ``externa``, su equivalente en ``base``.

    Misma regla que ``emparejar_con_megagym`` (README §3.5): **exacto** si los
    nombres normalizados coinciden; **difuso** si ``token_sort_ratio`` ≥
    ``umbral_similitud``, el solapamiento de tokens supera ``umbral_jaccard`` y
    el par es coherente. La coherencia cambia: en lugar de regiones de
    megaGymDataset, los dos ejercicios deben compartir al menos una articulación
    en sus agonistas, y la misma familia de equipamiento (``otro`` no
    contradice). Un ejercicio sin músculos no se contrasta. Además, un par
    difuso con modificadores opuestos (``MODIFICADORES_OPUESTOS``) se rechaza.

    Un par exacto se vincula con **todos** los ejercicios de la base que
    comparten ese nombre normalizado (``"push-up (male)"`` y ``"push-up"``), para
    que las etiquetas lleguen a cada variante.

    Returns:
        Una fila por par (externo, base) aceptado, más una fila por externo sin
        par, con su mejor candidato, para auditar el rechazo.
    """
    umbral_sim = float(parametros["umbral_similitud"])
    umbral_jac = float(parametros["umbral_jaccard"])
    articulaciones = _articulaciones_tolerantes(musculo_articulacion, regiones)

    def _zona(musculos: list[str]) -> set[str]:
        return set().union(*(articulaciones.get(m, set()) for m in musculos))

    base = base.reset_index(drop=True)
    externa = externa.reset_index(drop=True)
    zona_base = base["musculos_primarios"].map(_zona).tolist()
    familia_base = base["familia_equipamiento"].to_numpy(dtype=object)
    indice_nombre = base.groupby("nombre_norm").indices
    similitud = process.cdist(externa["nombre_norm"].tolist(), base["nombre_norm"].tolist(),
                              scorer=fuzz.token_sort_ratio, workers=-1)

    filas = []
    for i, ejercicio in externa.iterrows():
        puntajes = similitud[i]
        zona = _zona(ejercicio["musculos_primarios"])
        familia = ejercicio["familia_equipamiento"]

        def _coherente(j: int) -> tuple[Any, bool]:
            region_ok = (pd.NA if not zona or not zona_base[j]
                         else bool(zona & zona_base[j]))
            equipo_ok = (familia in {"otro"} or familia_base[j] in {"otro"}
                         or familia == familia_base[j])
            return region_ok, equipo_ok

        exactos = indice_nombre.get(ejercicio["nombre_norm"])
        if exactos is not None:
            elegidos, metodo = list(exactos), "exacto"
        else:
            elegidos, metodo = [], "sin_match"
            candidatos = np.flatnonzero(puntajes >= umbral_sim)
            for j in candidatos[np.argsort(-puntajes[candidatos], kind="stable")]:
                region_ok, equipo_ok = _coherente(j)
                if region_ok is False or not equipo_ok or _modificadores_contradicen(
                        ejercicio["nombre_norm"], base.at[j, "nombre_norm"]):
                    continue
                if jaccard_tokens(ejercicio["nombre_norm"], base.at[j, "nombre_norm"]) > umbral_jac:
                    elegidos, metodo = [int(j)], "difuso"
                    break
        for j in elegidos or [int(puntajes.argmax())]:
            region_ok, equipo_ok = _coherente(j)
            filas.append({
                "ejercicio_id": ejercicio["ejercicio_id"],
                "nombre": ejercicio["nombre"],
                "candidato_id": base.at[j, "ejercicio_id"],
                "candidato_nombre": base.at[j, "nombre"],
                "similitud": round(float(puntajes[j]), 2),
                "jaccard": round(jaccard_tokens(ejercicio["nombre_norm"],
                                                base.at[j, "nombre_norm"]), 3),
                "region_coherente": region_ok,
                "equipo_coherente": equipo_ok,
                "metodo": metodo,
                "aceptado": metodo != "sin_match",
            })

    reporte = pd.DataFrame(filas)
    por_ejercicio = reporte.drop_duplicates("ejercicio_id")["metodo"].value_counts()
    logger.info(
        "Emparejamiento %s → %s: %d exactos, %d difusos, %d sin par (%d vínculos).",
        externa["fuente"].iat[0], "+".join(base["fuente"].unique()),
        por_ejercicio.get("exacto", 0), por_ejercicio.get("difuso", 0),
        por_ejercicio.get("sin_match", 0), int(reporte["aceptado"].sum()),
    )
    return reporte


def emparejar_free_exercise_db(catalogo, fedb, musculo_articulacion, regiones, parametros):
    """free-exercise-db contra el catálogo."""
    return emparejar_fuentes(catalogo, fedb, musculo_articulacion, regiones, parametros)


def emparejar_wger(catalogo, fedb, wger, musculo_articulacion, regiones, parametros):
    """wger contra catálogo + free-exercise-db: wger es la fuente de menor
    prioridad, así que solo aporta los ejercicios que no están en las otras."""
    base = pd.concat([catalogo, fedb], ignore_index=True)
    return emparejar_fuentes(base, wger, musculo_articulacion, regiones, parametros)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Consolidación: nodos y aristas
# ─────────────────────────────────────────────────────────────────────────────
ETIQUETAS = ["nivel", "mecanica", "fuerza"]


def consolidar_ejercicios(
    catalogo: pd.DataFrame,
    fedb: pd.DataFrame,
    wger: pd.DataFrame,
    match_fedb: pd.DataFrame,
    match_wger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye la tabla de ejercicios únicos y la lista de aristas
    ejercicio → músculo.

    - Prioridad de fuentes: catálogo > free-exercise-db > wger. Un ejercicio
      externo con par no crea nodo: queda registrado en ``fuentes_vinculadas``
      del nodo al que se une.
    - Etiquetas de un nodo del catálogo con par en free-exercise-db: se toman de
      ahí (si hay varios pares, del de mayor similitud). Si no, ``nivel`` sigue
      viniendo de megaGymDataset curado. ``*_origen`` registra de dónde salió
      cada etiqueta, para que los modelos puedan separar dato de fuente y dato
      transferido.
    - Los músculos de un nodo son siempre los de su propia fuente: no se mezclan
      listas de fuentes distintas en un mismo ejercicio.

    Returns:
        ``ejercicios`` (1 fila = 1 nodo) y ``ejercicio_musculo`` (aristas con
        ``rol`` = ``agonista`` o ``sinergista``).
    """
    vinculos = pd.concat([match_fedb, match_wger], ignore_index=True)
    vinculos = vinculos.loc[vinculos["aceptado"]]

    nuevos_fedb = fedb.loc[~fedb["ejercicio_id"].isin(match_fedb.loc[match_fedb["aceptado"],
                                                                      "ejercicio_id"])]
    nuevos_wger = wger.loc[~wger["ejercicio_id"].isin(match_wger.loc[match_wger["aceptado"],
                                                                      "ejercicio_id"])]
    ejercicios = pd.concat([catalogo, nuevos_fedb, nuevos_wger], ignore_index=True)

    for etiqueta in ETIQUETAS:
        ejercicios[f"{etiqueta}_origen"] = np.where(
            ejercicios[etiqueta].notna(),
            np.where(ejercicios["fuente"] == "catalogo", "megagym_curado", "fuente_propia"),
            None,
        )

    # Etiquetas transferidas desde free-exercise-db: el mejor par por nodo.
    pares_fedb = (vinculos.loc[vinculos["ejercicio_id"].str.startswith("free_exercise_db:")]
                  .sort_values(["similitud", "ejercicio_id"], ascending=[False, True])
                  .drop_duplicates("candidato_id"))
    transferidas = (pares_fedb[["candidato_id", "ejercicio_id"]]
                    .merge(fedb[["ejercicio_id", *ETIQUETAS]], on="ejercicio_id")
                    .drop(columns="ejercicio_id").set_index("candidato_id"))
    en_catalogo = ejercicios["ejercicio_id"].isin(transferidas.index)
    for etiqueta in ETIQUETAS:
        valor = ejercicios.loc[en_catalogo, "ejercicio_id"].map(transferidas[etiqueta])
        if etiqueta == "nivel":
            # Se guarda el nivel de megaGymDataset para medir el acuerdo entre fuentes.
            ejercicios["nivel_megagym"] = ejercicios["nivel"].where(
                ejercicios["fuente"] == "catalogo")
        usar = en_catalogo & pd.Series(valor, index=ejercicios.index).notna()
        ejercicios.loc[usar, etiqueta] = valor[usar]
        ejercicios.loc[usar, f"{etiqueta}_origen"] = "free_exercise_db"

    vinculadas = vinculos.groupby("candidato_id")["ejercicio_id"].agg(sorted)
    ejercicios["fuentes_vinculadas"] = ejercicios["ejercicio_id"].map(vinculadas).map(
        lambda v: v if isinstance(v, list) else [])
    ejercicios["es_rehabilitacion"] = (ejercicios["contexto"] == "rehabilitacion_funcional") | \
        ejercicios["es_estiramiento"]
    if ejercicios["ejercicio_id"].duplicated().any():
        raise ValueError("La consolidación duplicó ejercicios.")

    aristas = pd.concat([
        ejercicios[["ejercicio_id", "musculos_primarios"]]
        .explode("musculos_primarios").rename(columns={"musculos_primarios": "musculo"})
        .assign(rol="agonista"),
        ejercicios[["ejercicio_id", "musculos_secundarios"]]
        .explode("musculos_secundarios").rename(columns={"musculos_secundarios": "musculo"})
        .assign(rol="sinergista"),
    ]).dropna(subset=["musculo"]).reset_index(drop=True)

    logger.info(
        "Ejercicios únicos: %d (catálogo %d, free-exercise-db nuevos %d, wger nuevos %d); "
        "%d aristas ejercicio→músculo; %d ejercicios sin ningún músculo.",
        len(ejercicios), len(catalogo), len(nuevos_fedb), len(nuevos_wger), len(aristas),
        int((~ejercicios["ejercicio_id"].isin(aristas["ejercicio_id"])).sum()),
    )
    return ejercicios, aristas


# ─────────────────────────────────────────────────────────────────────────────
# 5. Resumen de la integración
# ─────────────────────────────────────────────────────────────────────────────
def resumir_integracion(ejercicios: pd.DataFrame, aristas: pd.DataFrame,
                        musculo_articulacion: pd.DataFrame) -> pd.DataFrame:
    """Cobertura de etiquetas y de articulaciones por fuente y por contexto.

    Es la tabla que responde si la integración cumplió su objetivo: cuántos
    ejercicios ganan etiquetas, y si la rehabilitación deja de ser el contexto
    peor cubierto (README §5.1).
    """
    con_articulacion = set(aristas.merge(musculo_articulacion, on="musculo")["ejercicio_id"])
    df = ejercicios.assign(
        con_articulacion=ejercicios["ejercicio_id"].isin(con_articulacion),
        **{f"con_{e}": ejercicios[e].notna() for e in ETIQUETAS},
    )
    metricas = ["con_articulacion", *(f"con_{e}" for e in ETIQUETAS)]

    def _resumen(grupo: pd.DataFrame) -> pd.Series:
        return pd.Series({"n_ejercicios": len(grupo),
                          **{f"pct_{m}": round(grupo[m].mean() * 100, 1) for m in metricas}})

    bloques = [
        df.groupby("fuente").apply(_resumen, include_groups=False)
        .reset_index().rename(columns={"fuente": "grupo"}).assign(dimension="fuente"),
        df.groupby("contexto").apply(_resumen, include_groups=False)
        .reset_index().rename(columns={"contexto": "grupo"}).assign(dimension="contexto"),
        df.loc[df["fuente"] == "catalogo"].groupby("contexto").apply(_resumen, include_groups=False)
        .reset_index().rename(columns={"contexto": "grupo"}).assign(dimension="contexto_catalogo"),
        _resumen(df).to_frame().T.assign(grupo="total", dimension="total"),
    ]
    resumen = pd.concat(bloques, ignore_index=True)
    return resumen[["dimension", "grupo", "n_ejercicios", *(f"pct_{m}" for m in metricas)]]
