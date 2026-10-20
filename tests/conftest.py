"""Fixtures compartidas: cargan los artefactos versionados del repositorio.

Los tests no ejecutan ``kedro run``: verifican las salidas ya versionadas en
``data/`` y ``models/``, para correr en segundos.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]

# numexpr (dependencia indirecta de kedro-datasets) falla con algunas expresiones de
# DataFrame.query en Python 3.13; el motor de Python basta aquí.
pd.set_option("compute.use_numexpr", False)


@pytest.fixture(scope="session")
def parametros() -> dict:
    return yaml.safe_load((RAIZ / "conf/base/parameters.yml").read_text())


@pytest.fixture(scope="session")
def ejercicios() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data/03_primary/ejercicios.parquet")


@pytest.fixture(scope="session")
def ejercicio_musculo() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data/03_primary/ejercicio_musculo.parquet")


@pytest.fixture(scope="session")
def musculo_articulacion() -> pd.DataFrame:
    return pd.read_csv(RAIZ / "data/03_primary/musculo_articulacion.csv")


@pytest.fixture(scope="session")
def perfil() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data/07_model_output/perfil_exigencia.parquet")


@pytest.fixture(scope="session")
def familias() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data/07_model_output/familias_movimiento.parquet")


@pytest.fixture(scope="session")
def embeddings() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "models/embeddings_node2vec.parquet")


@pytest.fixture(scope="session")
def tabla_modelado() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data/05_model_input/tabla_modelado.parquet")


@pytest.fixture(scope="session")
def modelos() -> dict:
    return joblib.load(RAIZ / "models/modelos_supervisados.pkl")


@pytest.fixture(scope="session")
def comparacion() -> pd.DataFrame:
    return pd.read_csv(RAIZ / "data/08_reporting/modelos_comparacion.csv")
