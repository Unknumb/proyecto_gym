"""
Nodos del pipeline Grafo Biomecánico — familias de movimiento y sustitución con
articulación vetada (EV2, fase 3; decisión en ``docs/plan_ev2.md``).

**Pregunta de negocio.** ¿Qué familias de movimiento existen en el catálogo, y
cuántas alternativas reales tiene un ejercicio cuando se veta una articulación
lesionada?

**El grafo.** Nodos Ejercicio, Músculo, Articulación, Familia de equipamiento y
las etiquetas de la fase 2 (fuerza y mecánica). Las aristas Ejercicio → Músculo
pesan según el rol (agonista > sinergista); Músculo → Articulación sale del mapeo
curado de la EV1, y Región → Músculo de ``regiones_compuestas.csv``. Las regiones
son el puente entre vocabularios: free-exercise-db dice ``shoulders`` donde el
catálogo dice ``deltoids``.

**Dos técnicas no supervisadas, comparadas:**

- **Louvain** sobre la proyección Ejercicio–Ejercicio (k vecinos más parecidos
  por perfil biomecánico). Optimiza la modularidad.
- **Node2Vec + K-Means**: paseos aleatorios sesgados sobre el grafo completo,
  *embeddings* con word2vec (skip-gram) y K-Means sobre ellos. Se evalúa con
  *silhouette* y Davies-Bouldin.

**Sustitución.** El método de familias decide *qué es equivalente*; el
*embedding* ordena las alternativas; las reglas de seguridad (no cargar la
articulación vetada, conservar un músculo objetivo) son filtros duros, nunca un
puntaje que se pueda compensar.
"""

from __future__ import annotations

import logging
import zlib
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

#: Prefijo del id de nodo según su tipo. Los ejercicios conservan su
#: ``ejercicio_id`` (``catalogo:0001``), que ya identifica la fuente.
TIPOS_ENTIDAD = {
    "musculo": "musculo:",
    "articulacion": "articulacion:",
    "equipamiento": "equipamiento:",
    "fuerza": "fuerza:",
    "mecanica": "mecanica:",
}
METODOS = {"louvain": "comunidad_louvain", "kmeans_node2vec": "cluster_kmeans"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Construcción del grafo
# ─────────────────────────────────────────────────────────────────────────────
def _expandir(musculos, componentes: dict[str, set[str]]) -> set[str]:
    """Agrega los componentes de cada región (``shoulders`` → ``deltoids``, …)."""
    return set().union(*({m} | componentes.get(m, set()) for m in musculos)) if len(musculos) else set()


def construir_grafo(
    ejercicios: pd.DataFrame,
    perfil: pd.DataFrame,
    ejercicio_musculo: pd.DataFrame,
    musculo_articulacion: pd.DataFrame,
    regiones: pd.DataFrame,
    carga_por_patron: pd.DataFrame,
    parametros: dict[str, Any],
) -> tuple[nx.Graph, pd.DataFrame]:
    """Grafo heterogéneo y tabla de nodos Ejercicio.

    Se excluyen los ejercicios sin ningún músculo: sin anatomía no hay familia
    de movimiento ni articulación que vetar.

    **Carga articular.** Un ejercicio carga una articulación si *cualquiera* de
    sus músculos (agonista o sinergista, con las regiones expandidas a sus
    componentes) la cruza. Es deliberadamente conservador: en la sentadilla del
    catálogo el agonista es ``glutes`` y el cuádriceps figura como sinergista;
    mirar solo agonistas la daría por segura para la rodilla.

    **Carga por patrón de movimiento.** La carga deducida de los músculos depende
    de que la ficha los anote todos, y no siempre lo hace: *Kettlebell deadlifts*
    no anota la zona lumbar, y un peso muerto la carga. Por eso se le **suman**
    las articulaciones que el nombre delata según ``carga_por_patron`` (tabla
    curada: patrón, articulación y motivo), y se guardan aparte en
    ``articulaciones_carga_patron`` para poder auditarlas. Solo agrega
    articulaciones, nunca las quita: una ficha incompleta ya no puede hacer pasar
    un ejercicio por seguro. No toca las aristas del grafo, así que las familias
    de movimiento no cambian.

    Returns:
        El grafo (``networkx``) y una fila por ejercicio con su perfil de
        exigencia, músculos expandidos y articulaciones que carga.
    """
    pesos = parametros["pesos_aristas"]
    componentes = regiones.groupby("region")["musculo_componente"].agg(set).to_dict()
    articulaciones = musculo_articulacion.groupby("musculo")["articulacion"].agg(set).to_dict()

    con_musculo = set(ejercicio_musculo["ejercicio_id"])
    nodos = ejercicios.loc[ejercicios["ejercicio_id"].isin(con_musculo), [
        "ejercicio_id", "fuente", "nombre", "contexto", "familia_equipamiento",
        "es_estiramiento", "es_rehabilitacion",
    ]].merge(perfil[["ejercicio_id", "mecanica", "fuerza", "complejidad_tecnica", "impacto",
                     "estabilidad", "carga_graduable"]], on="ejercicio_id", how="left",
             validate="one_to_one").reset_index(drop=True)
    logger.info("Grafo: %d de %d ejercicios tienen músculos y entran al grafo.",
                len(nodos), len(ejercicios))

    relaciones = ejercicio_musculo[ejercicio_musculo["ejercicio_id"].isin(con_musculo)]
    por_rol = relaciones.groupby(["ejercicio_id", "rol"])["musculo"].agg(set).unstack()
    agonistas = por_rol["agonista"].reindex(nodos["ejercicio_id"])
    sinergistas = por_rol["sinergista"].reindex(nodos["ejercicio_id"])
    agonistas = [_expandir(a if isinstance(a, set) else set(), componentes) for a in agonistas]
    sinergistas = [_expandir(s if isinstance(s, set) else set(), componentes) for s in sinergistas]
    nodos["agonistas"] = [sorted(a) for a in agonistas]
    nodos["musculos"] = [sorted(a | s) for a, s in zip(agonistas, sinergistas)]
    por_musculos = [set().union(*(articulaciones.get(m, set()) for m in a | s))
                    for a, s in zip(agonistas, sinergistas)]
    por_patron = [set() for _ in por_musculos]
    nombres = nodos["nombre"].str.lower()
    for regla in carga_por_patron.itertuples(index=False):
        for i in np.flatnonzero(nombres.str.contains(regla.patron, regex=True).to_numpy()):
            if regla.articulacion not in por_musculos[i]:
                por_patron[i].add(regla.articulacion)
    nodos["articulaciones_carga"] = [sorted(m | p) for m, p in zip(por_musculos, por_patron)]
    nodos["articulaciones_carga_patron"] = [sorted(p) for p in por_patron]
    logger.info("Carga por patrón de movimiento: %d ejercicios suman articulaciones que sus "
                "músculos no anotan.", sum(bool(p) for p in por_patron))

    grafo = nx.Graph()
    p = TIPOS_ENTIDAD
    for fila in nodos.itertuples(index=False):
        grafo.add_node(fila.ejercicio_id, tipo="ejercicio", nombre=fila.nombre, fuente=fila.fuente)
        grafo.add_edge(fila.ejercicio_id, p["equipamiento"] + fila.familia_equipamiento,
                       weight=pesos["equipamiento"])
        if isinstance(fila.fuerza, str):
            grafo.add_edge(fila.ejercicio_id, p["fuerza"] + fila.fuerza, weight=pesos["fuerza"])
        if isinstance(fila.mecanica, str):
            grafo.add_edge(fila.ejercicio_id, p["mecanica"] + fila.mecanica, weight=pesos["mecanica"])
    # Las aristas a músculo usan el vocabulario original (sin expandir): la
    # expansión la hacen las aristas Región → Músculo dentro del propio grafo.
    for fila in relaciones.itertuples(index=False):
        grafo.add_edge(fila.ejercicio_id, p["musculo"] + fila.musculo, weight=pesos[fila.rol])
    for fila in musculo_articulacion.itertuples(index=False):
        grafo.add_edge(p["musculo"] + fila.musculo, p["articulacion"] + fila.articulacion,
                       weight=pesos["articulacion"])
    for fila in regiones.itertuples(index=False):
        grafo.add_edge(p["musculo"] + fila.region, p["musculo"] + fila.musculo_componente,
                       weight=pesos["region"])
    for nodo in grafo:
        if "tipo" not in grafo.nodes[nodo]:
            grafo.nodes[nodo]["tipo"] = nodo.split(":", 1)[0]

    tipos = pd.Series(dict(grafo.nodes(data="tipo"))).value_counts().to_dict()
    logger.info("Grafo: %d nodos %s, %d aristas.", grafo.number_of_nodes(), tipos,
                grafo.number_of_edges())
    return grafo, nodos


# ─────────────────────────────────────────────────────────────────────────────
# 2. Louvain sobre la proyección Ejercicio–Ejercicio
# ─────────────────────────────────────────────────────────────────────────────
def perfil_biomecanico(grafo: nx.Graph, ids: list[str]) -> sparse.csr_matrix:
    """Vector de cada ejercicio sobre los nodos no-ejercicio, normalizado (L2).

    Suma sus aristas directas (músculos, equipamiento, fuerza, mecánica) y lo que
    alcanzan sus músculos en un salto (articulaciones y componentes de región).
    Así ``shoulders`` y ``deltoids`` comparten la articulación ``hombro`` y el
    componente ``deltoids``, y dos fuentes con distinto vocabulario se parecen.
    """
    fila = {n: i for i, n in enumerate(ids)}
    entidades = [n for n, t in grafo.nodes(data="tipo") if t != "ejercicio"]
    columna = {n: i for i, n in enumerate(entidades)}
    directo = sparse.lil_matrix((len(ids), len(entidades)))
    salto = sparse.lil_matrix((len(entidades), len(entidades)))
    for u, v, peso in grafo.edges(data="weight"):
        if u in fila or v in fila:
            e, n = (u, v) if u in fila else (v, u)
            directo[fila[e], columna[n]] = peso
        elif grafo.nodes[u]["tipo"] == "musculo":
            salto[columna[u], columna[v]] = peso
            if grafo.nodes[v]["tipo"] == "musculo":
                salto[columna[v], columna[u]] = peso
        else:
            salto[columna[v], columna[u]] = peso
    directo = directo.tocsr()
    return normalize(directo + directo @ salto.tocsr()).tocsr()


def _proyeccion_knn(perfil: sparse.csr_matrix, vecinos: int) -> nx.Graph:
    """Grafo de k vecinos más parecidos (coseno). Peso = similitud."""
    knn = NearestNeighbors(n_neighbors=vecinos + 1, metric="cosine").fit(perfil)
    distancias, indices = knn.kneighbors(perfil)
    proyeccion = nx.Graph()
    proyeccion.add_nodes_from(range(perfil.shape[0]))
    for i, (dist, idx) in enumerate(zip(distancias[:, 1:], indices[:, 1:])):
        for d, j in zip(dist, idx):
            if d < 1:
                proyeccion.add_edge(i, int(j), weight=max(
                    1 - float(d), proyeccion.get_edge_data(i, j, {"weight": 0})["weight"]))
    return proyeccion


def _etiquetas(particion, n: int) -> np.ndarray:
    etiquetas = np.empty(n, dtype=int)
    for k, miembros in enumerate(sorted(particion, key=len, reverse=True)):
        etiquetas[list(miembros)] = k
    return etiquetas


def detectar_comunidades(grafo: nx.Graph, nodos: pd.DataFrame,
                         parametros: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Louvain con barrido de resolución.

    Para cada resolución: número de comunidades, modularidad, estabilidad (ARI
    medio entre semillas) y AMI con la fuente. Una AMI alta con la fuente sería
    mala señal: el método estaría separando *quién escribió la ficha* y no el
    movimiento. Se usa la resolución de ``parametros``; el barrido la justifica.

    Returns:
        ``(comunidades, barrido)``: comunidad por ejercicio (0 = la más grande)
        y la tabla del barrido.
    """
    p = parametros["louvain"]
    ids = nodos["ejercicio_id"].tolist()
    proyeccion = _proyeccion_knn(perfil_biomecanico(grafo, ids), int(p["vecinos"]))
    semillas = [int(parametros["semilla"]) + s for s in range(int(p["semillas_estabilidad"]))]

    barrido, elegida = [], None
    for resolucion in p["resoluciones"]:
        particiones = [nx.community.louvain_communities(
            proyeccion, weight="weight", resolution=resolucion, seed=s) for s in semillas]
        etiquetas = [_etiquetas(c, len(ids)) for c in particiones]
        barrido.append({
            "resolucion": resolucion,
            "n_comunidades": len(particiones[0]),
            "modularidad": nx.community.modularity(proyeccion, particiones[0], weight="weight"),
            "estabilidad_ari": float(np.mean([adjusted_rand_score(etiquetas[0], e)
                                              for e in etiquetas[1:]])),
            "ami_fuente": adjusted_mutual_info_score(nodos["fuente"], etiquetas[0]),
            "tamano_min": min(map(len, particiones[0])),
            "tamano_max": max(map(len, particiones[0])),
        })
        if resolucion == p["resolucion"]:
            elegida = etiquetas[0]
    if elegida is None:
        raise ValueError(f"louvain.resolucion={p['resolucion']} no está en louvain.resoluciones")

    barrido = pd.DataFrame(barrido).round(4)
    logger.info("Louvain (resolución %s): %s", p["resolucion"],
                barrido.loc[barrido["resolucion"] == p["resolucion"]].to_dict("records")[0])
    return pd.DataFrame({"ejercicio_id": ids, "comunidad_louvain": elegida}), barrido


# ─────────────────────────────────────────────────────────────────────────────
# 3. Node2Vec
# ─────────────────────────────────────────────────────────────────────────────
def _paseos_node2vec(grafo: nx.Graph, n_paseos: int, largo: int, p: float, q: float,
                     semilla: int) -> list[list[str]]:
    """Paseos aleatorios de segundo orden (Grover y Leskovec, 2016).

    Desde ``actual``, habiendo llegado desde ``previo``, el peso de ir a
    ``siguiente`` se divide por ``p`` si vuelve a ``previo`` y por ``q`` si se
    aleja (no es vecino de ``previo``). Con ``q < 1`` el paseo explora hacia
    afuera, y los *embeddings* capturan la pertenencia a una misma región del
    grafo (homofilia), que es lo que define una familia de movimiento.
    """
    rng = np.random.default_rng(semilla)
    nodos = sorted(grafo.nodes)
    vecinos = {n: sorted(grafo.neighbors(n)) for n in nodos}
    pesos = {n: np.array([grafo[n][m]["weight"] for m in vecinos[n]], dtype=float) for n in nodos}
    conjuntos = {n: set(v) for n, v in vecinos.items()}
    paseos = []
    for _ in range(n_paseos):
        for inicio in rng.permutation(len(nodos)):
            paseo = [nodos[inicio]]
            while len(paseo) < largo and vecinos[paseo[-1]]:
                actual = paseo[-1]
                w = pesos[actual].copy()
                if len(paseo) > 1:
                    previo = paseo[-2]
                    for k, siguiente in enumerate(vecinos[actual]):
                        if siguiente == previo:
                            w[k] /= p
                        elif siguiente not in conjuntos[previo]:
                            w[k] /= q
                paseo.append(vecinos[actual][rng.choice(len(w), p=w / w.sum())])
            paseos.append(paseo)
    return paseos


def _hash_estable(texto: str) -> int:
    """``hash()`` de Python cambia entre ejecuciones; word2vec lo usa para
    inicializar los vectores. Con crc32 el entrenamiento es reproducible."""
    return zlib.crc32(texto.encode("utf-8"))


def entrenar_node2vec(grafo: nx.Graph, parametros: dict[str, Any],
                      semilla: int | None = None) -> pd.DataFrame:
    """*Embeddings* Node2Vec de todos los nodos del grafo.

    word2vec (skip-gram) trata cada paseo como una frase: dos nodos quedan cerca
    si aparecen en los mismos contextos. Un solo hilo y hash estable para que el
    resultado sea idéntico en cada ``kedro run``.

    Returns:
        Una fila por nodo: ``nodo``, ``tipo`` y ``dim_00`` … ``dim_NN``.
    """
    from gensim.models import Word2Vec

    p = parametros["node2vec"]
    semilla = int(parametros["semilla"]) if semilla is None else semilla
    paseos = _paseos_node2vec(grafo, int(p["paseos_por_nodo"]), int(p["largo_paseo"]),
                              float(p["p"]), float(p["q"]), semilla)
    # gensim 4.4 declara el puntero a ``sdot`` con ``except -1``: cuando un
    # producto punto vale exactamente -1,0 lo toma por error, imprime "Exception
    # ignored in our_dot_float" y usa 0. Ocurre en un puñado de los millones de
    # actualizaciones; el efecto sobre los vectores es despreciable.
    modelo = Word2Vec(paseos, vector_size=int(p["dimensiones"]), window=int(p["ventana"]),
                      min_count=1, sg=1, negative=5, epochs=int(p["epocas"]), workers=1,
                      seed=semilla, hashfxn=_hash_estable)
    nodos = sorted(grafo.nodes)
    vectores = np.vstack([modelo.wv[n] for n in nodos])
    embeddings = pd.DataFrame(vectores, columns=[f"dim_{i:02d}" for i in range(vectores.shape[1])])
    embeddings.insert(0, "tipo", [grafo.nodes[n]["tipo"] for n in nodos])
    embeddings.insert(0, "nodo", nodos)
    logger.info("Node2Vec: %d paseos, %d nodos en %d dimensiones.", len(paseos), len(nodos),
                vectores.shape[1])
    return embeddings


def matriz_embeddings(embeddings: pd.DataFrame, ids: list[str]) -> np.ndarray:
    """Vectores de los ejercicios en el orden de ``ids``, normalizados (L2): la
    distancia euclídea pasa a ser equivalente a la de coseno."""
    columnas = [c for c in embeddings.columns if c.startswith("dim_")]
    return normalize(embeddings.set_index("nodo").loc[ids, columnas].to_numpy())


# ─────────────────────────────────────────────────────────────────────────────
# 4. K-Means sobre los embeddings
# ─────────────────────────────────────────────────────────────────────────────
def agrupar_embeddings(embeddings: pd.DataFrame, nodos: pd.DataFrame,
                       parametros: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-Means con barrido de k (método del codo, *silhouette*, Davies-Bouldin).

    Se usa el ``k`` de ``parametros``; el barrido lo justifica. Para cada k se
    mide además la estabilidad entre semillas de K-Means y la AMI con la fuente.

    Returns:
        ``(clusters, barrido)``.
    """
    p = parametros["kmeans"]
    ids = nodos["ejercicio_id"].tolist()
    X = matriz_embeddings(embeddings, ids)
    semillas = [int(parametros["semilla"]) + s for s in range(int(p["semillas_estabilidad"]))]

    barrido, elegido = [], None
    for k in p["k_candidatos"]:
        ajustes = [KMeans(k, n_init=10, random_state=s).fit(X) for s in semillas]
        etiquetas = ajustes[0].labels_
        barrido.append({
            "k": k,
            "inercia": ajustes[0].inertia_,
            "silhouette": silhouette_score(X, etiquetas),
            "davies_bouldin": davies_bouldin_score(X, etiquetas),
            "estabilidad_ari": float(np.mean([adjusted_rand_score(etiquetas, a.labels_)
                                              for a in ajustes[1:]])),
            "ami_fuente": adjusted_mutual_info_score(nodos["fuente"], etiquetas),
        })
        if k == p["k"]:
            elegido = etiquetas
    if elegido is None:
        raise ValueError(f"kmeans.k={p['k']} no está en kmeans.k_candidatos")

    # Renumerar por tamaño (0 = el más grande), igual que Louvain.
    orden = pd.Series(elegido).value_counts().index
    elegido = pd.Series(elegido).map({c: i for i, c in enumerate(orden)}).to_numpy()
    # La inercia se redondea a 2 decimales: K-Means suma en paralelo y el orden
    # de la suma cambia el cuarto decimal entre ejecuciones.
    barrido = pd.DataFrame(barrido).round(4).round({"inercia": 2})
    logger.info("K-Means (k = %s): %s", p["k"],
                barrido.loc[barrido["k"] == p["k"]].to_dict("records")[0])
    return pd.DataFrame({"ejercicio_id": ids, "cluster_kmeans": elegido}), barrido


# ─────────────────────────────────────────────────────────────────────────────
# 5. Familias de movimiento: tabla final, perfiles y comparación de métodos
# ─────────────────────────────────────────────────────────────────────────────
_FUERZA_ES = {"push": "empuje", "pull": "tracción", "static": "estático"}


def _nombre_familia(grupo: pd.DataFrame) -> str:
    """Nombre automático: los dos agonistas más frecuentes y la fuerza dominante."""
    agonistas = grupo["agonistas"].explode().value_counts()
    # Las regiones (``shoulders``) se expandieron a sus componentes: se nombra por
    # los músculos y no por la región, que repetiría la misma información.
    agonistas = agonistas[~agonistas.index.isin(["back", "chest", "shoulders", "core", "neck",
                                                  "spine"])]
    musculos = " + ".join(agonistas.head(2).index)
    if grupo["es_estiramiento"].mean() >= 0.5:
        return f"estiramientos ({musculos})"
    fuerza = grupo["fuerza"].value_counts(normalize=True)
    patron = _FUERZA_ES.get(fuerza.index[0], "") if len(fuerza) and fuerza.iloc[0] >= 0.6 else "mixto"
    return f"{musculos} · {patron}"


def consolidar_familias(
    nodos: pd.DataFrame,
    comunidades: pd.DataFrame,
    clusters: pd.DataFrame,
    embeddings: pd.DataFrame,
    grafo: nx.Graph,
    parametros: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Une las dos particiones, agrega la proyección UMAP y compara los métodos.

    La comparación usa dos varas comunes: la *silhouette* en el espacio de los
    *embeddings* (la vara de K-Means) y la modularidad en la proyección k-NN (la
    vara de Louvain). Cada método se mide también con la vara del otro.

    Returns:
        ``(familias, perfiles, comparacion)``.
    """
    import umap

    familias = nodos.merge(comunidades, on="ejercicio_id").merge(clusters, on="ejercicio_id")
    ids = familias["ejercicio_id"].tolist()
    X = matriz_embeddings(embeddings, ids)
    p = parametros["umap"]
    coordenadas = umap.UMAP(n_neighbors=int(p["vecinos"]), min_dist=float(p["distancia_minima"]),
                            metric="euclidean", random_state=int(parametros["semilla"])
                            ).fit_transform(X)
    familias["umap_x"], familias["umap_y"] = coordenadas[:, 0], coordenadas[:, 1]

    perfiles = []
    for metodo, columna in METODOS.items():
        nombres = familias.groupby(columna).apply(_nombre_familia, include_groups=False)
        familias[f"familia_{metodo}"] = familias[columna].map(nombres)
        for grupo_id, grupo in familias.groupby(columna):
            fuente = grupo["fuente"].value_counts(normalize=True)
            contexto = grupo["contexto"].value_counts(normalize=True)
            perfiles.append({
                "metodo": metodo, "grupo": grupo_id, "nombre": nombres[grupo_id],
                "n": len(grupo),
                "agonistas_top": ", ".join(grupo["agonistas"].explode().value_counts().head(3).index),
                "fuerza_dominante": grupo["fuerza"].mode().iat[0] if grupo["fuerza"].notna().any() else None,
                "pct_compuesto": (grupo["mecanica"] == "compound").mean() * 100,
                "pct_estiramiento": grupo["es_estiramiento"].mean() * 100,
                "pct_catalogo": fuente.get("catalogo", 0) * 100,
                "pct_free_exercise_db": fuente.get("free_exercise_db", 0) * 100,
                "pct_wger": fuente.get("wger", 0) * 100,
                "contexto_dominante": contexto.index[0],
                "equipamiento_dominante": grupo["familia_equipamiento"].mode().iat[0],
                "articulaciones_top": ", ".join(
                    grupo["articulaciones_carga"].explode().value_counts().head(3).index),
            })
    perfiles = pd.DataFrame(perfiles).round(1)

    proyeccion = _proyeccion_knn(perfil_biomecanico(grafo, ids), int(parametros["louvain"]["vecinos"]))
    agonista_principal = familias["agonistas"].map(lambda a: a[0] if len(a) else "")
    comparacion = []
    for metodo, columna in METODOS.items():
        etiquetas = familias[columna].to_numpy()
        particion = [set(np.flatnonzero(etiquetas == g)) for g in np.unique(etiquetas)]
        comparacion.append({
            "metodo": metodo,
            "n_grupos": len(particion),
            "silhouette_embedding": silhouette_score(X, etiquetas),
            "davies_bouldin_embedding": davies_bouldin_score(X, etiquetas),
            "modularidad_proyeccion": nx.community.modularity(proyeccion, particion, weight="weight"),
            "ami_fuente": adjusted_mutual_info_score(familias["fuente"], etiquetas),
            "ami_agonista_principal": adjusted_mutual_info_score(agonista_principal, etiquetas),
            "ami_fuerza": adjusted_mutual_info_score(familias["fuerza"].fillna("sin_etiqueta"), etiquetas),
        })
    comparacion = pd.DataFrame(comparacion)
    comparacion["ari_entre_metodos"] = adjusted_rand_score(familias["comunidad_louvain"],
                                                           familias["cluster_kmeans"])
    comparacion = comparacion.round(4)
    logger.info("Comparación de métodos:\n%s", comparacion.to_string(index=False))
    return familias, perfiles, comparacion


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sustitución con articulación vetada
# ─────────────────────────────────────────────────────────────────────────────
class Sustituidor:
    """Recomienda alternativas a un ejercicio cuando una articulación está vetada.

    Reglas, en orden:

    1. **Seguridad (filtro duro).** La alternativa no carga la articulación
       vetada, contando agonistas y sinergistas.
    2. **Músculo objetivo.** Comparte al menos un agonista del original que no
       cruce la articulación vetada. Si todos los agonistas del original la
       cruzan (extensión de rodilla con la rodilla vetada) no existe sustituto
       por definición, y el sistema lo dice en vez de inventar uno.
    3. **Cobertura mínima (README §5.1 b).** Si esos músculos tienen menos de
       ``min_ejercicios_musculo`` ejercicios en el grafo, se abstiene.
    4. **Mismo tipo.** Un ejercicio de fuerza no se reemplaza por un
       estiramiento, ni al revés.
    5. **Familia.** Primero alternativas de la misma familia de movimiento; si no
       alcanzan, se completa con otras familias, marcadas como tales.
    6. **Orden.** Similitud de coseno entre *embeddings* Node2Vec.
    7. **Cuota sin equipamiento (README §5.1 a).** Si en el top no hay ninguna
       alternativa de peso corporal y existe una válida, entra en el último lugar.
    """

    def __init__(self, familias: pd.DataFrame, embeddings: pd.DataFrame,
                 musculo_articulacion: pd.DataFrame, parametros: dict[str, Any]):
        p = parametros["sustitucion"]
        self.familias = familias.reset_index(drop=True)
        self.indice = pd.Series(self.familias.index, index=self.familias["ejercicio_id"])
        X = matriz_embeddings(embeddings, self.familias["ejercicio_id"].tolist())
        self.similitud = (X @ X.T).astype(np.float32)
        self.cruza = musculo_articulacion.groupby("musculo")["articulacion"].agg(set).to_dict()
        self.carga = [set(a) for a in self.familias["articulaciones_carga"]]
        self.agonistas = [set(a) for a in self.familias["agonistas"]]
        self.estiramiento = self.familias["es_estiramiento"].to_numpy(dtype=bool)
        conteo = self.familias["agonistas"].explode().value_counts()
        self.escasos = set(conteo[conteo < int(p["min_ejercicios_musculo"])].index)
        self.top_n = int(p["top_n"])
        self.min_alternativas = int(p["min_alternativas"])
        self.sin_equipamiento = set(p["familias_sin_equipamiento"])
        self.metodo = parametros["metodo_familias"]

    def _candidatos(self, i: int, articulacion: str, filtro: np.ndarray | None):
        objetivo = {m for m in self.agonistas[i] if articulacion not in self.cruza.get(m, set())}
        if not objetivo:
            return None, objetivo, "sin_sustituto_anatomico"
        if objetivo <= self.escasos:
            return None, objetivo, "musculo_con_pocos_ejercicios"
        validos = np.array([articulacion not in c and bool(a & objetivo)
                            for c, a in zip(self.carga, self.agonistas)])
        validos &= self.estiramiento == self.estiramiento[i]
        validos[i] = False
        if filtro is not None:
            validos &= filtro
        return np.flatnonzero(validos), objetivo, None

    def recomendar(self, ejercicio_id: str, articulacion: str, metodo: str | None = None,
                   filtro: np.ndarray | None = None, cuota: bool = True) -> tuple[pd.DataFrame, str | None]:
        """Alternativas ordenadas para ``ejercicio_id`` con ``articulacion`` vetada.

        Args:
            filtro: máscara booleana opcional sobre ``familias`` (p. ej. perfil
                de principiante).
            cuota: aplicar la cuota de alternativas sin equipamiento.

        Returns:
            ``(alternativas, motivo)``. Si el sistema se abstiene, ``alternativas``
            está vacío y ``motivo`` explica por qué.
        """
        metodo = metodo or self.metodo
        columna = METODOS[metodo]
        i = int(self.indice[ejercicio_id])
        if articulacion not in self.carga[i]:
            return self.familias.iloc[[]], "no_carga_la_articulacion"
        candidatos, _, motivo = self._candidatos(i, articulacion, filtro)
        if motivo:
            return self.familias.iloc[[]], motivo

        familia = self.familias[columna].to_numpy()
        misma = familia[candidatos] == familia[i]
        orden = np.lexsort((-self.similitud[i, candidatos], ~misma))
        elegidos = list(candidatos[orden][: self.top_n])
        if cuota and elegidos:
            equipo = self.familias["familia_equipamiento"].to_numpy()
            if not any(equipo[j] in self.sin_equipamiento for j in elegidos):
                resto = [j for j in candidatos[orden][self.top_n:] if equipo[j] in self.sin_equipamiento]
                if resto:
                    elegidos[-1] = resto[0]

        alternativas = self.familias.iloc[elegidos].assign(
            similitud=self.similitud[i, elegidos].round(3),
            misma_familia=familia[elegidos] == familia[i])
        motivo = None if len(alternativas) >= self.min_alternativas else "menos_de_3_alternativas"
        return alternativas, motivo


def evaluar_sustitucion(familias: pd.DataFrame, embeddings: pd.DataFrame,
                        musculo_articulacion: pd.DataFrame,
                        parametros: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """KPI *Cobertura de sustitución* (README §1.3) por articulación y método.

    Para cada articulación, se toman los ejercicios que la cargan (los que un
    usuario lesionado tendría que reemplazar) y se mide:

    - ``sin_sustituto_anatomico``: todos sus agonistas cruzan la articulación.
    - ``abstencion_musculo_escaso``: regla de §5.1 b.
    - ``cobertura_misma_familia``: % con ≥ 3 alternativas de su propia familia.
    - ``cobertura_total``: % con ≥ 3 alternativas, aunque sean de otra familia.
    - ``misma_fuerza_top3``: coherencia del patrón (empuje/tracción/estático)
      entre el original y sus 3 primeras alternativas.
    - ``con_sin_equipamiento_*``: % de listas con al menos una alternativa de
      peso corporal, antes y después de la cuota de §5.1 a.

    Los porcentajes se calculan sobre los ejercicios *sustituibles* (los que no
    caen en las dos primeras categorías). ``metodo = sin_familias`` es la línea
    base: solo las reglas de seguridad, sin agrupar.

    Returns:
        ``(por_articulacion, por_contexto)``.
    """
    sustituidor = Sustituidor(familias, embeddings, musculo_articulacion, parametros)
    articulaciones = sorted(set(musculo_articulacion["articulacion"]))
    equipo = familias["familia_equipamiento"].to_numpy()
    fuerza = familias["fuerza"].to_numpy()
    metodos = {**METODOS, "sin_familias": None}

    registros = []
    for articulacion in articulaciones:
        afectados = [i for i, c in enumerate(sustituidor.carga) if articulacion in c]
        for i in afectados:
            candidatos, _, motivo = sustituidor._candidatos(i, articulacion, None)
            base = {"articulacion": articulacion, "ejercicio_id": familias["ejercicio_id"].iat[i],
                    "contexto": familias["contexto"].iat[i], "motivo": motivo}
            if motivo:
                registros.extend({**base, "metodo": m} for m in metodos)
                continue
            orden = candidatos[np.argsort(-sustituidor.similitud[i, candidatos], kind="stable")]
            for metodo, columna in metodos.items():
                if columna is None:
                    misma = np.ones(len(orden), dtype=bool)
                else:
                    grupo = familias[columna].to_numpy()
                    misma = grupo[orden] == grupo[i]
                propios = orden[misma]
                lista = np.concatenate([propios, orden[~misma]])[: sustituidor.top_n]
                top3 = lista[:3]
                registros.append({
                    **base, "metodo": metodo,
                    "n_misma_familia": len(propios), "n_total": len(orden),
                    "misma_fuerza_top3": (np.mean(fuerza[top3] == fuerza[i])
                                          if isinstance(fuerza[i], str) and len(top3) else np.nan),
                    "sin_equipo_antes": any(equipo[j] in sustituidor.sin_equipamiento for j in lista),
                    "sin_equipo_disponible": any(equipo[j] in sustituidor.sin_equipamiento for j in orden),
                })
    detalle = pd.DataFrame(registros)
    detalle["sin_equipo_despues"] = detalle["sin_equipo_antes"] | detalle["sin_equipo_disponible"]

    def _resumir(grupo: pd.DataFrame) -> pd.Series:
        sustituibles = grupo[grupo["motivo"].isna()]
        n = max(len(sustituibles), 1)
        return pd.Series({
            "afectados": len(grupo),
            "sin_sustituto_anatomico": int((grupo["motivo"] == "sin_sustituto_anatomico").sum()),
            "abstencion_musculo_escaso": int((grupo["motivo"] == "musculo_con_pocos_ejercicios").sum()),
            "sustituibles": len(sustituibles),
            "cobertura_misma_familia": (sustituibles["n_misma_familia"] >= sustituidor.min_alternativas).sum() / n * 100,
            "cobertura_total": (sustituibles["n_total"] >= sustituidor.min_alternativas).sum() / n * 100,
            "misma_fuerza_top3": sustituibles["misma_fuerza_top3"].mean() * 100,
            "con_sin_equipamiento_antes": sustituibles["sin_equipo_antes"].mean() * 100,
            "con_sin_equipamiento_despues": sustituibles["sin_equipo_despues"].mean() * 100,
        })

    por_articulacion = (detalle.groupby(["metodo", "articulacion"]).apply(_resumir, include_groups=False)
                        .reset_index())
    global_ = (detalle.groupby("metodo").apply(_resumir, include_groups=False).reset_index()
               .assign(articulacion="TODAS"))
    por_articulacion = pd.concat([por_articulacion, global_], ignore_index=True).round(1)
    por_contexto = (detalle.groupby(["metodo", "contexto"]).apply(_resumir, include_groups=False)
                    .reset_index().round(1))

    resumen = global_.set_index("metodo")[["cobertura_misma_familia", "cobertura_total",
                                            "misma_fuerza_top3"]].round(1)
    logger.info("Cobertura de sustitución (todas las articulaciones):\n%s", resumen.to_string())
    return por_articulacion, por_contexto
