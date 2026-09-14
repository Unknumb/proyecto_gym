# Databricks: guía de puesta en marcha

Guía operativa para ejecutar el proyecto en **Databricks Free Edition**. Complementa
la justificación arquitectónica del README §2.3; aquí solo está el *cómo*.

> **Free Edition es serverless.** No se crean clústeres, no hay consola de cuenta y no
> hay DBFS. Todo el cómputo es serverless y todos los archivos viven en **volúmenes de
> Unity Catalog**. Coste: cero, sin tarjeta de crédito. Uso no comercial.

---

## 1. Crear la cuenta (5 minutos)

1. Entra a <https://login.databricks.com/?intent=CE_SIGN_UP> y regístrate con Google o
   con correo institucional.
2. Databricks aprovisiona **un workspace y un metastore** automáticamente. No hay nada
   que configurar.
3. Anota la URL del workspace (`https://dbc-xxxxxxxx-xxxx.cloud.databricks.com`).

**Cada integrante del equipo necesita su propia cuenta**: Free Edition da un workspace
por cuenta y no permite invitar colaboradores. La colaboración ocurre en Git, no dentro
del workspace.

## 2. Clonar el repositorio como Git folder

`Workspace` → `Users` → tu usuario → menú `⋮` → **Create → Git folder**.

| Campo | Valor |
|---|---|
| Git repository URL | `https://github.com/Unknumb/proyecto_gym.git` |
| Git provider | GitHub |
| Repository name | `proyecto_gym` |

Como `data/01_raw/exercises.json` está versionado, el clon trae el dato: no hay que
subir nada a mano. Para cambiar de rama usa el selector de rama del Git folder.

> Si GitHub pide autenticación: `Settings` → `Linked accounts` → *Personal access token*
> de GitHub con permiso `repo`.

## 3. Ejecutar los notebooks

En orden, con **Serverless** seleccionado en el desplegable de cómputo (arriba a la
derecha; se conecta en segundos y no hay que configurarlo):

| Notebook | Qué hace |
|---|---|
| `notebooks/databricks/01_bronze_ingesta_delta.py` | Crea esquemas `gym_bronze/silver/gold` y el volumen `raw`, copia el JSON al volumen, lo carga **con esquema explícito** y escribe la tabla bronze. Demuestra que Delta rechaza la escritura con `id` numérico y muestra *time travel*. |
| `notebooks/databricks/02_silver_gold_medallion.py` | Construye la tabla analítica (silver) y los agregados de reporting (gold) en Spark, y **reconcilia** el resultado contra el Parquet de pandas. |

Ambos aceptan los parámetros `catalog` (por defecto `workspace`) y `prefix` (`gym`) en
los *widgets* de la parte superior.

**La celda 6 del notebook 01 es la evidencia para la defensa.** Falla a propósito: Delta
rechaza anexar un `id` inferido como `bigint` a una columna declarada `STRING`. Es el
mismo defecto que en pandas se propagó en silencio por todo el pipeline (README §3.3).
Captura esa salida.

## 4. Explorar el resultado

`Catalog` → `workspace` → `gym_bronze` / `gym_silver` / `gym_gold`. Cada tabla tiene
pestañas de *Sample data*, *Details*, *History* (versiones Delta) y *Lineage* (el grafo
de qué tabla produjo cuál — útil como figura del informe).

Desde el editor SQL:

```sql
SELECT musculo_primario, COUNT(*) AS n
FROM workspace.gym_silver.exercise_features
GROUP BY musculo_primario ORDER BY n DESC;

DESCRIBE HISTORY workspace.gym_bronze.exercises;
```

## 5. (Opcional) Ejecutar el pipeline Kedro completo en el workspace

El entorno `conf/databricks/` reapunta el catálogo a tablas Delta gestionadas **sin
tocar los nodos**: `ManagedTableDataset` con `dataframe_type: pandas` hace la conversión
en el borde de I/O.

En un notebook nuevo dentro del Git folder:

```python
%pip install -r ../../requirements.txt "kedro-datasets[databricks]"
%restart_python
```

```python
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

project_root = Path.cwd().parents[1]        # raíz del Git folder
bootstrap_project(project_root)

with KedroSession.create(project_path=project_root, env="databricks") as session:
    session.run(pipeline_name="data_understanding")
```

Es la ruta de despliegue, no un requisito de la entrega: los notebooks del paso 3 ya
cubren la evidencia. Úsala si quieres una sola definición de pipeline para local y nube.

## 6. Límites de Free Edition que afectan a este proyecto

| Límite | Consecuencia aquí |
|---|---|
| Solo cómputo serverless, sin configuración | No hay clúster *Single Node*, ni auto-terminación a 15 min, ni instancias *spot* que ajustar |
| Un workspace y un metastore por cuenta; sin consola de cuenta | No hay *budget alerts* de AWS que configurar; el coste es cero por diseño |
| Red saliente limitada a dominios de confianza | `scripts/auditar_media_fisica.py` (descarga los GIF de GymVisual) **no** corre aquí; se ejecuta local |
| Máximo 5 tareas de *job* concurrentes; cuota de uso diaria/mensual | Irrelevante a esta escala |
| Sin R ni Scala; SQL warehouse único `2X-Small` | Irrelevante: todo es Python y SQL |

> **Nota para el informe.** La tabla FinOps del README §2.3 describe controles de coste
> de un workspace *de pago* (clúster Single Node, auto-terminación, spot, budget alert en
> AWS). En Free Edition ninguno de esos controles existe ni hace falta. Conviene
> aclararlo en la defensa: la medida de coste efectivamente aplicada es *usar Free
> Edition*, y el resto queda como plan para el escenario de escalamiento.

---

## Referencias

- [Databricks Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition)
- [Limitaciones de Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)
- [Limitaciones de cómputo serverless](https://docs.databricks.com/aws/en/compute/serverless/limitations)
