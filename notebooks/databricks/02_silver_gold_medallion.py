# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver y Gold: tabla analítica y agregados sobre Delta
# MAGIC
# MAGIC Reimplementa en Spark los nodos `build_exercise_features` (silver) y los agregados
# MAGIC de reporting (gold) del pipeline Kedro, sobre la tabla bronze creada en el
# MAGIC notebook `01`. Las reglas de negocio son **las mismas** —incluidas las tres
# MAGIC correcciones documentadas en el README (§3)—: cambia el motor, no la lógica.
# MAGIC
# MAGIC La última sección **reconcilia** el resultado de Spark contra el Parquet que
# MAGIC produjo pandas: si ambos motores no coinciden, la migración no es correcta.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catálogo Unity")
dbutils.widgets.text("prefix", "gym", "Prefijo de esquemas")

CATALOG = dbutils.widgets.get("catalog")
PREFIX = dbutils.widgets.get("prefix")
BRONZE, SILVER, GOLD = (f"{CATALOG}.{PREFIX}_{c}" for c in ("bronze", "silver", "gold"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Silver — tabla analítica a grano ejercicio
# MAGIC
# MAGIC Una fila = un ejercicio. No se aplica `explode` sobre `secondary_muscles`: eso
# MAGIC inflaría los estadísticos univariados (ver docstring de `build_exercise_features`).

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

bronze = spark.table(f"{BRONZE}.exercises")

# Contenido en español: se descartan los 10 idiomas anidados (~15,8 MB) y se
# conserva solo el texto que consume el EDA — restricción FinOps del README.
pasos_es = F.coalesce(F.col("instruction_steps").getItem("es"), F.array())
texto_es = F.coalesce(F.col("instructions").getItem("es"), F.lit(""))

features = (
    bronze
    # Corrección 1: el id es una llave textual de 4 dígitos, no un número.
    .withColumn("id", F.lpad(F.trim(F.col("id")), 4, "0"))
    .withColumn("instrucciones_es", texto_es)
    .withColumn("n_pasos", F.size(pasos_es))
    .withColumn("n_idiomas", F.size(F.map_keys(F.col("instructions"))))
    .withColumn("n_caracteres", F.length(texto_es))
    .withColumn(
        "n_palabras",
        F.when(F.length(F.trim(texto_es)) == 0, F.lit(0)).otherwise(
            F.size(F.split(F.trim(texto_es), r"\s+"))
        ),
    )
    .withColumn(
        "n_palabras_paso_max",
        F.coalesce(
            F.array_max(F.transform(pasos_es, lambda p: F.size(F.split(F.trim(p), r"\s+")))),
            F.lit(0),
        ),
    )
    .withColumn("n_palabras_nombre", F.size(F.split(F.trim(F.col("name")), r"\s+")))
    # Corrección 2: el agonista es `target`; `muscle_group` es sinergista.
    .withColumn("musculo_primario", F.col("target"))
    .withColumn("musculo_sinergista", F.col("muscle_group"))
    .withColumn(
        "n_musculos_secundarios",
        F.size(F.coalesce(F.col("secondary_muscles"), F.array())),
    )
    # Corrección 3: conteo deduplicado de {target} ∪ secundarios.
    .withColumn(
        "n_musculos_total",
        F.size(
            F.array_distinct(
                F.array_union(
                    F.coalesce(F.col("secondary_muscles"), F.array()),
                    F.array(F.col("target")),
                )
            )
        ),
    )
    .withColumn("palabras_por_paso", F.round(F.col("n_palabras") / F.col("n_pasos"), 3))
    # Frecuencias de contexto: proxies de accesibilidad y de sesgo (IE4).
    .withColumn("frecuencia_equipamiento", F.count("*").over(Window.partitionBy("equipment")))
    .withColumn("frecuencia_musculo_primario", F.count("*").over(Window.partitionBy("target")))
    .select(
        "id", "name", "category", "body_part", "equipment",
        "musculo_primario", "musculo_sinergista", "gif_url", "media_id",
        "instrucciones_es", "n_pasos", "n_palabras", "n_caracteres",
        "palabras_por_paso", "n_palabras_paso_max", "n_palabras_nombre",
        "n_musculos_secundarios", "n_musculos_total", "n_idiomas",
        "frecuencia_equipamiento", "frecuencia_musculo_primario",
    )
)

(
    features.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER}.exercise_features")
)

display(spark.table(f"{SILVER}.exercise_features").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Gold — agregados de reporting
# MAGIC
# MAGIC Equivalentes a `data/08_reporting/`: distribuciones categóricas, estadísticos de
# MAGIC forma y outliers por criterio de Tukey (IQR).

# COMMAND ----------

silver = spark.table(f"{SILVER}.exercise_features")
total = silver.count()

for dim in ("body_part", "equipment", "musculo_primario", "category"):
    (
        silver.groupBy(F.col(dim).alias("categoria"))
        .agg(F.count("*").alias("n_ejercicios"))
        .withColumn("pct", F.round(F.col("n_ejercicios") * 100 / F.lit(total), 2))
        .withColumn("dimension", F.lit(dim))
        .orderBy(F.desc("n_ejercicios"))
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{GOLD}.distribucion_{dim}")
    )

display(spark.table(f"{GOLD}.distribucion_equipment"))

# COMMAND ----------

VARIABLES_CUANTITATIVAS = [
    "n_pasos", "n_palabras", "n_caracteres", "palabras_por_paso",
    "n_palabras_paso_max", "n_palabras_nombre", "n_musculos_secundarios",
    "n_musculos_total", "frecuencia_equipamiento", "frecuencia_musculo_primario",
]

# Cuartiles exactos (relativeError=0) — el dataset es pequeño y no hay razón
# para aceptar la aproximación del algoritmo de Greenwald-Khanna.
cuartiles = {
    var: silver.approxQuantile(var, [0.25, 0.75], 0.0) for var in VARIABLES_CUANTITATIVAS
}

filas = []
for var, (q1, q3) in cuartiles.items():
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = silver.filter((F.col(var) < low) | (F.col(var) > high)).count()
    filas.append(
        {
            "variable": var, "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "limite_inferior": float(low), "limite_superior": float(high),
            "n_outliers": n_out, "pct_outliers": round(n_out * 100 / total, 2),
        }
    )

outliers = spark.createDataFrame(filas)
outliers.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(f"{GOLD}.outliers_iqr")

display(outliers.orderBy(F.desc("pct_outliers")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reconciliación contra el pipeline de pandas
# MAGIC
# MAGIC Dos motores, un mismo resultado. Si esta celda no cierra, la migración introdujo
# MAGIC una diferencia de lógica y hay que corregirla antes de confiar en las tablas.

# COMMAND ----------

import os

import pandas as pd

NOTEBOOK_PATH = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
REPO_DIR = os.path.abspath(os.path.join("/Workspace", NOTEBOOK_PATH.lstrip("/"), "..", "..", ".."))
PARQUET = os.path.join(REPO_DIR, "data", "02_intermediate", "exercise_features.parquet")

if os.path.exists(PARQUET):
    ref = pd.read_parquet(PARQUET).sort_values("id").reset_index(drop=True)
    got = silver.toPandas().sort_values("id").reset_index(drop=True)[ref.columns]

    print(f"pandas: {ref.shape} · spark: {got.shape}")
    resumen = pd.DataFrame(
        {
            "pandas": ref[VARIABLES_CUANTITATIVAS].sum(),
            "spark": got[VARIABLES_CUANTITATIVAS].sum(),
        }
    )
    resumen["coincide"] = (resumen["pandas"] - resumen["spark"]).abs() < 1e-6
    display(resumen.reset_index().rename(columns={"index": "variable"}))
    print("ids idénticos:", ref["id"].equals(got["id"]))
else:
    print(f"No se encontró {PARQUET}; ejecuta `kedro run` localmente y sincroniza el Git folder.")
