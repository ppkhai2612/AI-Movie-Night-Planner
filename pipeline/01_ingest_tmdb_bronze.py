# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Ingest TMDB -> Bronze Delta
# MAGIC
# MAGIC First stage of the Spark pipeline. It crawls [TMDB](https://developer.themoviedb.org/) and lands the **raw JSON** reponses in Unity Catalog Delta tables, untouched.
# MAGIC
# MAGIC ```
# MAGIC   /genre/movie/list                       -> bronze_tmdb_genres
# MAGIC   /discover/movie  (paged)                -> the id list
# MAGIC   /movie/{id}?append_to_response=...      -> bronze_tmdb_movies
# MAGIC ```
# MAGIC
# MAGIC **Why keep raw JSON rather than parsing here?** Because the parse is the part most likely to need changing. Bronze holds exactly what the API said, so `02_build_silver_gold` can be rewritten and re-run over the same data without spending another 500 API calls - and without the results shifting underneath, since TMDB's popularity ordering changes hourly.
# MAGIC
# MAGIC **Where the parallelism is.** The detail fetch is one HTTP call per movie and is the whole cost of this notebook. `mapInPandas` runs it across the cluster: the id list is repartitioned, and each partition opens its own `TMDBClient` and streams results back. The auth token is resolved **once on the driver** and broadcast, because executors have no Databricks SDK credentials and a `WorkspaceClient()` call inside a UDF would fail
# MAGIC
# MAGIC Rate limiting still applies per executor - `TMDB_REQUEST_SPACING` (50 ms) times the number of partitions is the effective request rate, so keep `fetch_partitions` modest rather than matching the core count.

# COMMAND ----------

# MAGIC %pip install -q requests databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Every setting goes through `conf()`: Databricks widget, then environment variable, then default. That is what lets this file run as an interactive notebook, as a scheduled job task with `base_parameters`, and as a plain `python pipeline/01_ingest_tmdb_bronze.py`

# COMMAND ----------

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-tmdb-bronze")

# pipeline_common lives beside this file; the repo root (movie_db.py,
# tmdb_client.py) is one level up and has to be found rather than assumed.
sys.path.insert(
    0,
    (
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals()
        else "pipeline"
    ),
)
try:
    from pipeline_common import bootstrap, conf, conf_int, table_name
except ModuleNotFoundError:  # notebook: pipeline/ isn't on the path yet
    for _candidate in (os.getcwd(), "/Workspace" + os.getcwd()):
        _p = os.path.join(_candidate, "pipeline")
        if os.path.isdir(_p):
            sys.path.insert(0, _p)
            break
    from pipeline_common import bootstrap, conf, conf_int, table_name

REPO_ROOT = bootstrap()
logger.info("Repo root: %s", REPO_ROOT)

CATALOG = conf("uc_catalog", "main")
SCHEMA = conf("uc_schema", "movie_night")

# 25 pages x 20 results = ~500 movies, which is enough for the semantic search
# to have real competition without a crawl that takes an hour.
DISCOVER_PAGES = conf_int("tmdb_discover_pages", 25)

# Concurrency for the detail fetch. 4 partitions x ~20 req/s each stays well
# under TMDB's ~40 req/s ceiling.
FETCH_PARTITIONS = conf_int("fetch_partitions", 4)

BRONZE_MOVIES = table_name(CATALOG, SCHEMA, "bronze_tmdb_movies")
BRONZE_GENRES = table_name(CATALOG, SCHEMA, "bronze_tmdb_genres")

print(f"catalog.schema   : {CATALOG}.{SCHEMA}")
print(f"discover pages   : {DISCOVER_PAGES}  (~{DISCOVER_PAGES * 20} movies)")
print(f"fetch partitions : {FETCH_PARTITIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark session and target schema

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# CREATE CATALOG needs metastore-admin rights, which a normal user usually does
# not have - and usually does not need, because the catalog (`main`) already
# exists. Treat failure as informational; a genuinely missing catalog will fail
# clearly at CREATE SCHEMA on the next line.
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
except Exception as exc:
    print(
        f"Could not create catalog {CATALOG!r} (this is normal if it "
        f"already exists): {str(exc).splitlines()[0]}"
    )

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the TMDB token once, on the driver
# MAGIC
# MAGIC Executors have no Databricks SDK credentials, so `TMDBClient()` calling `_token()` inside a UDF would raise. Resolving here and passing the value into the closure is the only arrangement that works on a real cluster

# COMMAND ----------

import tmdb_client
from tmdb_client import TMDBClient

TMDB_TOKEN = tmdb_client._token()
print(f"TMDB token resolved ({len(TMDB_TOKEN)} chars, starts {TMDB_TOKEN[:4]}...)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Genres
# MAGIC
# MAGIC ~19 rows, one request. Written first because `movie_genres` references them downstream.

# COMMAND ----------

from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql.types import (
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

_driver_client = TMDBClient(token=TMDB_TOKEN)
genre_rows = _driver_client.genres()
ingested_at = datetime.now(timezone.utc)

genres_schema = StructType(
    [
        StructField("genre_id", LongType(), False),
        StructField("name", StringType(), False),
        StructField("ingested_at", TimestampType(), False),
    ]
)

genres_df = spark.createDataFrame(
    [(int(g["id"]), g["name"], ingested_at) for g in genre_rows],
    schema=genres_schema,
)
genres_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    BRONZE_GENRES
)
print(f"Wrote {genres_df.count()} genres to {BRONZE_GENRES}")
display(genres_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discover the movie ids
# MAGIC
# MAGIC Paged, deduplicated, and driver-side: it is ~25 sequential requests and parallelising them would only make the page boundaries less stable

# COMMAND ----------

movie_ids = _driver_client.discover_movie_ids(pages=DISCOVER_PAGES)
print(f"Discovered {len(movie_ids)} distinct movie ids")
print(f"  first few: {movie_ids[:10]}")

if not movie_ids:
    raise RuntimeError(
        "TMDB /discover returned no ids. Check the token and that "
        "api.themoviedb.org is reachable from this cluster."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch details in parallel
# MAGIC
# MAGIC One `TMDBClient` per partition (not per row) so the session, its retry policy and its request spacing are reused
# MAGIC
# MAGIC Failures are **recorded, not raised**: a single 404 or a transient 5xx on one title should not lose the other 499. Each row carries `status` and `error`, so a bad crawl is visible in the data rather than as a stack trace, and `02` filters on `status = 'success'`.

# COMMAND ----------

from typing import Iterator

import pandas as pd

bronze_schema = StructType(
    [
        StructField("movie_id", LongType(), False),
        StructField("payload", StringType(), True),
        StructField("status", StringType(), False),
        StructField("error", StringType(), True),
        StructField("ingested_at", TimestampType(), False),
    ]
)


def fetch_details(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Fetch /movie/{id} for every id in this partition.

    Runs on an executor. TMDB_TOKEN is captured from the driver's scope; do not
    replace it with a _token() call here - there are no Databricks credentials
    on an executor.
    """
    from tmdb_client import TMDBClient, TMDBError, UnknownMovieError

    client = TMDBClient(token=TMDB_TOKEN)
    now = datetime.now(timezone.utc)

    for batch in batches:
        records = []
        for movie_id in batch["movie_id"]:
            movie_id = int(movie_id)
            try:
                payload = client.movie_details(movie_id)
                records.append((movie_id, json.dumps(payload), "success", None, now))
            except UnknownMovieError as exc:
                records.append((movie_id, None, "not_found", str(exc), now))
            except TMDBError as exc:
                records.append((movie_id, None, "error", str(exc), now))
        yield pd.DataFrame.from_records(
            records,
            columns=["movie_id", "payload", "status", "error", "ingested_at"],
        )


ids_df = spark.createDataFrame(
    [(int(i),) for i in movie_ids], schema="movie_id long"
).repartition(FETCH_PARTITIONS)

details_df = ids_df.mapInPandas(fetch_details, schema=bronze_schema)

# NOTE: no .cache() here. Databricks Serverless rejects it -
# [NOT_SUPPORTED_WITH_SERVERLESS] PERSIST TABLE is not supported.
#
# It isn't needed either. `mapInPandas` makes network calls, so it must be
# evaluated EXACTLY once or the crawl costs twice the API budget - but the
# `write.saveAsTable` below is a single action, and everything after it reads
# the Delta table rather than this DataFrame. Delta is the materialisation.
#
# The rule for the rest of this notebook: never call an action on `details_df`
# again. Use `spark.table(BRONZE_MOVIES)`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write bronze
# MAGIC
# MAGIC `overwrite` rather than `append`: bronze is a full snapshot of the current crawl, and Delta keeps the previous versions anyway (`DESCRIBE HISTORY`), so nothing is actually lost.

# COMMAND ----------

details_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    BRONZE_MOVIES
)

summary = (
    spark.table(BRONZE_MOVIES)
    .groupBy("status")
    .agg(F.count("*").alias("rows"))
    .orderBy("status")
)
print(f"Wrote {spark.table(BRONZE_MOVIES).count()} rows to {BRONZE_MOVIES}")
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity check
# MAGIC
# MAGIC Fails loudly if the crawl mostly failed - better to stop here than to build a silver layer from 12 movies and wonder why search is bad.

# COMMAND ----------

counts = {row["status"]: row["rows"] for row in summary.collect()}
succeeded = counts.get("success", 0)
total = sum(counts.values())

print(f"success: {succeeded} / {total}")
for status, rows in sorted(counts.items()):
    if status != "success":
        print(f"  {status}: {rows}")

if total == 0 or succeeded / total < 0.8:
    raise RuntimeError(
        f"Only {succeeded}/{total} TMDB detail fetches succeeded. "
        "Check the token, the rate limit, and cluster network access before "
        "running 02_build_silver_gold."
    )

print("\nBronze ingest complete. Next: pipeline/02_build_silver_gold.py")