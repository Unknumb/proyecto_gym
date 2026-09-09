# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze: ingesta gobernada del catálogo de ejercicios
# MAGIC
# MAGIC Este notebook materializa la capa **bronze** de la arquitectura medallion descrita
# MAGIC en el README (§2.3) sobre **Delta Lake**, y deja evidencia reproducible de la
# MAGIC afirmación central de esa sección: el *schema enforcement* de Delta **detiene en la
# MAGIC escritura** el defecto del `id` que en pandas se propagó silenciosamente
# MAGIC (`"0001"` → `1`, §3.3).
# MAGIC
# MAGIC **Entorno:** Databricks Free Edition (serverless). No requiere configurar clúster.
# MAGIC
# MAGIC **Prerrequisito:** el repositorio clonado como *Git folder* en el workspace, de modo
# MAGIC que `data/01_raw/exercises.json` esté disponible como archivo del workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Parámetros
# MAGIC
# MAGIC En Free Edition existe un solo metastore y el catálogo por defecto es `workspace`.
# MAGIC Los tres esquemas son literalmente bronze → silver → gold.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Catálogo Unity")
dbutils.widgets.text("prefix", "gym", "Prefijo de esquemas")

CATALOG = dbutils.widgets.get("catalog")
PREFIX = dbutils.widgets.get("prefix")

BRONZE = f"{CATALOG}.{PREFIX}_bronze"
SILVER = f"{CATALOG}.{PREFIX}_silver"
GOLD = f"{CATALOG}.{PREFIX}_gold"
VOLUME = f"/Volumes/{CATALOG}/{PREFIX}_bronze/raw"

print(f"bronze={BRONZE}\nsilver={SILVER}\ngold={GOLD}\nvolumen={VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Crear los esquemas y el volumen
# MAGIC
# MAGIC Un **volumen** de Unity Catalog es el lugar gobernado donde viven los archivos
# MAGIC crudos. En serverless no hay acceso a DBFS: los archivos van a volúmenes.

# COMMAND ----------

for schema in (BRONZE, SILVER, GOLD):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{PREFIX}_bronze.raw")
display(spark.sql(f"SHOW SCHEMAS IN {CATALOG} LIKE '{PREFIX}_*'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Copiar el JSON crudo del Git folder al volumen
# MAGIC
# MAGIC Spark no lee de forma fiable los archivos del workspace; el patrón correcto es
# MAGIC copiarlos una vez al volumen y leer desde ahí. Si el notebook no vive dentro del
# MAGIC Git folder, ajusta `REPO_DIR` o sube el JSON a mano desde
# MAGIC *Catalog → Volumes → raw → Upload*.

# COMMAND ----------

import os
import shutil

# Ruta del notebook dentro del workspace → raíz del repo clonado.
NOTEBOOK_PATH = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
REPO_DIR = os.path.abspath(os.path.join("/Workspace", NOTEBOOK_PATH.lstrip("/"), "..", "..", ".."))
SOURCE_JSON = os.path.join(REPO_DIR, "data", "01_raw", "exercises.json")
TARGET_JSON = f"{VOLUME}/exercises.json"

if os.path.exists(SOURCE_JSON):
    shutil.copyfile(SOURCE_JSON, TARGET_JSON)
    print(f"Copiado {SOURCE_JSON} → {TARGET_JSON}")
else:
    print(f"No se encontró {SOURCE_JSON}. Sube el archivo manualmente a {VOLUME}.")

display(dbutils.fs.ls(VOLUME))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. El defecto, reproducido en Spark
# MAGIC
# MAGIC Sin esquema explícito, la inferencia de tipos comete **el mismo error que pandas**:
# MAGIC `id` y `media_id` se infieren como numéricos y los ceros a la izquierda desaparecen,
# MAGIC rompiendo la llave hacia `gif_url` y `media_id`.

# COMMAND ----------

inferido = spark.read.option("multiLine", "true").json(TARGET_JSON)
print("Tipo inferido para id:", dict(inferido.dtypes)["id"])
display(inferido.select("id", "media_id", "gif_url").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Contrato de esquema explícito
# MAGIC
# MAGIC El esquema se declara, no se infiere. Los 10 idiomas se modelan como `MapType`:
# MAGIC `instructions` es `idioma → texto` e `instruction_steps` es `idioma → lista de pasos`.

# COMMAND ----------

from pyspark.sql.types import (
    ArrayType,
    MapType,
    StringType,
    StructField,
    StructType,
)

EXERCISE_SCHEMA = StructType(
    [
        StructField("id", StringType(), False),           # ← STRING: preserva "0001"
        StructField("name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("body_part", StringType(), True),
        StructField("equipment", StringType(), True),
        StructField("instructions", MapType(StringType(), StringType()), True),
        StructField(
            "instruction_steps",
            MapType(StringType(), ArrayType(StringType())),
            True,
        ),
        StructField("muscle_group", StringType(), True),
        StructField("secondary_muscles", ArrayType(StringType()), True),
        StructField("target", StringType(), True),
        StructField("image", StringType(), True),
        StructField("gif_url", StringType(), True),
        StructField("media_id", StringType(), False),     # ← STRING: llave hacia el medio
        StructField("created_at", StringType(), True),
        StructField("attribution", StringType(), True),
    ]
)

raw = spark.read.schema(EXERCISE_SCHEMA).option("multiLine", "true").json(TARGET_JSON)
print(f"{raw.count()} ejercicios · tipo de id: {dict(raw.dtypes)['id']}")
display(raw.select("id", "media_id", "gif_url").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Escritura de la tabla bronze

# COMMAND ----------

from pyspark.sql import functions as F

bronze = raw.withColumn("_ingested_at", F.current_timestamp()).withColumn(
    "_source_file", F.lit(TARGET_JSON)
)

(
    bronze.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{BRONZE}.exercises")
)

spark.sql(
    f"COMMENT ON TABLE {BRONZE}.exercises IS "
    "'Catálogo crudo de ejercicios (MIT, © 2026 Hasan Emir Yıldırım). "
    "id y media_id son STRING por contrato: los ceros a la izquierda son significativos.'"
)

display(spark.sql(f"DESCRIBE TABLE {BRONZE}.exercises"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evidencia: el *schema enforcement* rechaza la escritura corrupta
# MAGIC
# MAGIC Esta celda intenta **a propósito** anexar a la tabla los datos con el `id` inferido
# MAGIC como numérico. Delta debe rechazar la escritura. Es la prueba de la afirmación del
# MAGIC §2.3 y sirve como captura para la defensa.

# COMMAND ----------

corrupto = inferido.select("id", "media_id", "name")  # id: bigint

try:
    corrupto.write.format("delta").mode("append").saveAsTable(f"{BRONZE}.exercises")
    print("⚠️  La escritura NO fue rechazada: revisar la definición de la tabla.")
except Exception as exc:  # noqa: BLE001 — se documenta el mensaje completo
    print("✅ Delta rechazó la escritura. Motivo:\n")
    print(str(exc)[:1200])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. *Time travel*: auditar qué datos produjeron qué resultado

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {BRONZE}.exercises"))

# COMMAND ----------

# La versión 0 sigue siendo legible aunque la tabla haya cambiado después.
display(spark.read.format("delta").option("versionAsOf", 0).table(f"{BRONZE}.exercises").limit(5))
