# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Bronze -> Silver -> Gold
# MAGIC
# MAGIC Pure Spark. **No network calls at all** - everything here reads the raw JSON that `01_ingest_tmdb_bronze` landed, so this notebook can be edited and re-run as often as you like without touching TMDB's rate limit
# MAGIC
# MAGIC ```
# MAGIC   bronze_tmdb_movies (raw JSON)
# MAGIC        |  from_json with an explicit schema
# MAGIC        +--> silver_movies        one row per film, typed columns
# MAGIC        +--> silver_movie_genres  exploded
# MAGIC        +--> silver_cast          exploded, top-billed only
# MAGIC        +--> silver_keywords      exploded
# MAGIC        +--> silver_reviews       exploded  <- the long-form text
# MAGIC        |
# MAGIC        +--> gold_movie_documents  one assembled doc_text per film
# MAGIC ```
# MAGIC
# MAGIC ## The one design decision worth reading
# MAGIC
# MAGIC `doc_text` deliberately concatenates **title + tagline + overview + keywords + top cast + every user review**, not just the overview.
# MAGIC
# MAGIC A TMDB overview is 300-500 characters - one chunk at `CHUNK_SIZE=800`, and the 800/100 sliding window would be dead code. Reviews run to several KB and are what makes chunking meaningful. They also carry the vocabulary people actually search with ("slow burn", "not too violent", "good for a group"), which an overview never does - so they are what makes the semantic search feel semantic rather than like a plot-keyword lookup.
# MAGIC
# MAGIC ## Why an explicit `from_json` schema
# MAGIC
# MAGIC `schema_of_json` would infer from a sample, which means a field missing on the sampled rows silently disappears for every row. The schema below is declared, so a TMDB response that stops carrying `runtime` produces nulls - visible in the checks at the bottom - rather than a vanished column.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build-silver-gold")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))
                if "__file__" in globals() else "pipeline")

try:
    from pipeline_common import bootstrap, conf, conf_int, table_name
except ModuleNotFoundError:
    for _candidate in (os.getcwd(), "/Workspace" + os.getcwd()):
        _p = os.path.join(_candidate, "pipeline")
        if os.path.isdir(_p):
            sys.path.insert(0, _p)
            break
    from pipeline_common import bootstrap, conf, conf_int, table_name

REPO_ROOT = bootstrap()

CATALOG = conf("uc_catalog", "main")
SCHEMA = conf("uc_schema", "movie_night")

# Matches movie_db / the movie_cast table's intent: top-billed only.
CAST_LIMIT = conf_int("cast_limit", 10)

# Reviews are the bulk of doc_text. A handful of long ones is plenty; TMDB
# occasionally has 40+ on a blockbuster and the tail adds noise, not signal.
REVIEW_LIMIT = conf_int("review_limit", 8)

# Guard against one pathological review dominating a film's embedding
REVIEW_CHAR_LIMIT = conf_int("review_char_limit", 4000)

BRONZE_MOVIES = table_name(CATALOG, SCHEMA, "bronze_tmdb_movies")
BRONZE_GENRES = table_name(CATALOG, SCHEMA, "bronze_tmdb_genres")

# The parsed-but-not-yet-reshaped intermediate. It exists because Serverless
# has no .cache() and seven downstream reads would otherwise re-parse the JSON
# seven times - see the note where it's written.
PARSED = table_name(CATALOG, SCHEMA, "bronze_tmdb_parsed")

SILVER_MOVIES = table_name(CATALOG, SCHEMA, "silver_movies")
SILVER_GENRES = table_name(CATALOG, SCHEMA, "silver_genres")
SILVER_MOVIE_GENRES = table_name(CATALOG, SCHEMA, "silver_movie_genres")
SILVER_CAST = table_name(CATALOG, SCHEMA, "silver_cast")
SILVER_KEYWORDS = table_name(CATALOG, SCHEMA, "silver_keywords")
SILVER_REVIEWS = table_name(CATALOG, SCHEMA, "silver_reviews")
GOLD_DOCUMENTS = table_name(CATALOG, SCHEMA, "gold_movie_documents")

print(f"{CATALOG}.{SCHEMA}  cast<={CAST_LIMIT}  reviews<={REVIEW_LIMIT}")

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType
)

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## The TMDB payload schema
# MAGIC
# MAGIC Only the fields anything downstream reads. TMDB sends far more; ignoring the rest here is free, and bronze still has it if that changes.

# COMMAND ----------

_id_name = StructType([
    StructField("id", LongType()),
    StructField("name", StringType())
])

TMDB_SCHEMA = StructType([
    StructField("id", LongType()),
    StructField("title", StringType()),
    StructField("original_title", StringType()),
    StructField("tagline", StringType()),
    StructField("overview", StringType()),
    StructField("release_date", StringType()),
    StructField("runtime", IntegerType()),
    StructField("vote_average", DoubleType()),
    StructField("vote_count", IntegerType()),
    StructField("popularity", DoubleType()),
    StructField("original_language", StringType()),
    StructField("adult", BooleanType()),
    StructField("poster_path", StringType()),
    StructField("backdrop_path", StringType()),
    StructField("homepage", StringType()),
    StructField("imdb_id", StringType()),
    StructField("genres", ArrayType(_id_name)),
    StructField("credits", StructType([
        StructField("cast", ArrayType(StructType([
            StructField("id", LongType()),
            StructField("name", StringType()),
            StructField("character", StringType()),
            StructField("order", IntegerType()),
        ]))),
    ])),
    StructField("keywords", StructType([
        StructField("keywords", ArrayType(_id_name)),
    ])),
    StructField("reviews", StructType([
        StructField("results", ArrayType(StructType([
            StructField("id", StringType()),
            StructField("author", StringType()),
            StructField("content", StringType()),
            StructField("created_at", StringType()),
        ]))),
        StructField("total_results", IntegerType()),
    ])),
    # TMDB names this appended block with a slash. Spark accepts the field name
    # fine; it just has to be reached with getField() rather than dotted syntax.
    StructField("watch/providers", StructType([
        StructField("results", StructType([
            StructField("US", StructType([
                StructField("flatrate", ArrayType(StructType([
                    StructField("provider_id", LongType()),
                    StructField("provider_name", StringType()),
                ]))),
            ])),
        ])),
    ])),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse bronze
# MAGIC
# MAGIC `status = 'success'` drops the rows where the crawl recorded a 404 or an outage - they have a null payload by construction.

# COMMAND ----------

bronze = (
    spark.table(BRONZE_MOVIES)
    .where(F.col("status") == "success")
    .select("movie_id", "payload", "ingested_at")
)

# Five silver tables and the gold table all read this, so it is evaluated seven
# times. `.cache()` would be the obvious fix, but Databricks Serverless rejects
# it - [NOT_SUPPORTED_WITH_SERVERLESS] PERSIST TABLE is not supported.
#
# Writing it to Delta is the serverless equivalent and is arguably better: the
# parse is materialised once, and the intermediate is inspectable afterwards
# when a downstream column comes out unexpectedly null.
(
    bronze
    .withColumn("d", F.from_json("payload", TMDB_SCHEMA))
    # A movie with no overview embeds to noise, and a movie with no title can't
    # be rendered. Both are cheap to drop and expensive to keep.
    .where(F.col("d.title").isNotNull())
    .where(F.length(F.coalesce(F.col("d.overview"), F.lit(""))) > 0)
    .drop("payload")  # the raw JSON stays in bronze; carrying it again is waste
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(PARSED)
)

parsed = spark.table(PARSED)

print(f"bronze rows        : {bronze.count()}")
print(f"parsed with content: {parsed.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## silver_movies

# COMMAND ----------

providers = (
    F.col("d")
    .getField("watch/providers")
    .getField("results")
    .getField("US")
    .getField("flatrate")
)

silver_movies = parsed.select(
    F.col("movie_id"),
    F.col("d.title").alias("title"),
    F.col("d.original_title").alias("original_title"),
    # TMDB sends "" rather than null for an absent tagline/homepage.
    F.nullif(F.col("d.tagline"), F.lit("")).alias("tagline"),
    F.col("d.overview").alias("overview"),
    F.to_date(F.nullif(F.col("d.release_date"), F.lit(""))).alias("release_date"),
    F.col("d.runtime").alias("runtime_minutes"),
    F.col("d.vote_average").alias("vote_average"),
    F.col("d.vote_count").alias("vote_count"),
    F.col("d.popularity").alias("popularity"),
    F.col("d.original_language").alias("original_language"),
    F.coalesce(F.col("d.adult"), F.lit(False)).alias("adult"),
    F.col("d.poster_path").alias("poster_path"),
    F.col("d.backdrop_path").alias("backdrop_path"),
    F.nullif(F.col("d.homepage"), F.lit("")).alias("homepage"),
    F.nullif(F.col("d.imdb_id"), F.lit("")).alias("imdb_id"),
    F.transform(F.col("d.genres"), lambda g: g.getField("name")).alias("genre_names"),
    F.transform(
        F.slice(F.coalesce(F.col("d.keywords.keywords"), F.array()), 1, 15),
        lambda k: k.getField("name"),
    ).alias("keyword_names"),
    F.transform(
        F.coalesce(providers, F.array()), lambda p: p.getField("provider_name")
    ).alias("provider_names"),
    F.col("ingested_at"),
)

silver_movies.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    SILVER_MOVIES
)
print(f"{SILVER_MOVIES}: {spark.table(SILVER_MOVIES).count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## silver_genres / silver_movie_genres
# MAGIC
# MAGIC The genre list comes from its own bronze table (the authoritative `/genre/movie/list`), the links from each movie's own `genres` array

# COMMAND ----------


spark.table(BRONZE_GENRES).select("genre_id", "name").distinct() \
    .write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(SILVER_GENRES)

silver_movie_genres = (
    parsed
    .select("movie_id", F.explode("d.genres").alias("g"))
    .select("movie_id", F.col("g.id").alias("genre_id"))
    .distinct()
    # Only keep links whose genre exists in the genre table, so the Lakebase
    # foreign key can't be violated later.
    .join(spark.table(SILVER_GENRES).select("genre_id"), "genre_id", "inner")
)

silver_movie_genres.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(SILVER_MOVIE_GENRES)
print(f"{SILVER_GENRES}      : {spark.table(SILVER_GENRES).count()} rows")
print(f"{SILVER_MOVIE_GENRES}: {spark.table(SILVER_MOVIE_GENRES).count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## silver_cast
# MAGIC
# MAGIC Top-billed only. TMDB's `order` field is the billing position and is already 0-based, but it is occasionally null or duplicated, so the cut is made with `row_number()` over an explicit ordering rather than by trusting `order <= 9`

# COMMAND ----------

cast_window = Window.partitionBy("movie_id").orderBy(
    F.col("billing_order").asc_nulls_last(), F.col("person_id").asc()
)

silver_cast = (
    parsed
    .select("movie_id", F.explode("d.credits.cast").alias("c"))
    .select(
        "movie_id",
        F.col("c.id").alias("person_id"),
        F.col("c.name").alias("name"),
        F.nullif(F.col("c.character"), F.lit("")).alias("character_name"),
        F.col("c.order").alias("billing_order"),
    )
    .where(F.col("person_id").isNotNull() & F.col("name").isNotNull())
    .withColumn("rn", F.row_number().over(cast_window))
    .where(F.col("rn") <= CAST_LIMIT)
    # Renumber to a dense 0..n-1, so billing_order is a contiguous rank and the
    # (movie_id, person_id, billing_order) primary key can't collide.
    .withColumn("billing_order", F.col("rn") - 1)
    .drop("rn")
)

silver_cast.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(SILVER_CAST)
print(f"{SILVER_CAST}: {spark.table(SILVER_CAST).count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## silver_keywords

# COMMAND ----------

silver_keywords = (
    parsed
    .select("movie_id", F.explode(
        F.coalesce(F.col("d.keywords.keywords"), F.array())).alias("k"))
    .select(
        "movie_id",
        F.col("k.id").alias("keyword_id"),
        F.col("k.name").alias("name"),
    )
    .where(F.col("keyword_id").isNotNull())
    .distinct()
)

silver_keywords.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(SILVER_KEYWORDS)
print(f"{SILVER_KEYWORDS}: {spark.table(SILVER_KEYWORDS).count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## silver_reviews - the long-form unstructured text

# COMMAND ----------

review_window = Window.partitionBy("movie_id").orderBy(
    F.length("content").desc(), F.col("review_id").asc()
)

silver_reviews = (
    parsed
    .select("movie_id", F.explode(
        F.coalesce(F.col("d.reviews.results"), F.array())).alias("r"))
    .select(
        "movie_id",
        F.col("r.id").alias("review_id"),
        F.col("r.author").alias("author"),
        F.col("r.content").alias("content"),
        F.to_timestamp(F.col("r.created_at")).alias("created_at"),
    )
    .where(F.length(F.coalesce(F.col("content"), F.lit(""))) > 100)
    # Longest first: a two-line "loved it" contributes nothing an embedding can
    # use, so when a film has more reviews than REVIEW_LIMIT the substantial
    # ones are the ones kept.
    .withColumn("rn", F.row_number().over(review_window))
    .where(F.col("rn") <= REVIEW_LIMIT)
    .withColumn("content", F.substring(F.col("content"), 1, REVIEW_CHAR_LIMIT))
    .drop("rn")
)

silver_reviews.write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(SILVER_REVIEWS)
print(f"{SILVER_REVIEWS}: {spark.table(SILVER_REVIEWS).count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## gold_movie_documents
# MAGIC
# MAGIC One row per film. `doc_text` is what gets chunked and embedded in `03`; `content_hash` is `sha2(doc_text, 256)` and is the whole re-embedding mechanism - change the text and the hash changes, which is what `movie_db.fetch_unembedded_documents()` joins on.
# MAGIC
# MAGIC Labelled sections ("Genres:", "Cast:") rather than bare concatenation: the embedding model sees the label as context, and a chunk that lands mid-document is still self-describing when the agent quotes it back.

# COMMAND ----------

movies_t = spark.table(SILVER_MOVIES)

cast_agg = (
    spark.table(SILVER_CAST)
    .orderBy("movie_id", "billing_order")
    .groupBy("movie_id")
    .agg(F.concat_ws(", ", F.collect_list(
        F.when(F.col("character_name").isNotNull(),
               F.concat_ws(" as ", F.col("name"), F.col("character_name")))
         .otherwise(F.col("name"))
    )).alias("cast_line"))
)

reviews_agg = (
    spark.table(SILVER_REVIEWS)
    .groupBy("movie_id")
    .agg(
        F.concat_ws("\n\n", F.collect_list(
            F.concat(F.lit("Review by "), F.coalesce(F.col("author"),
                                                     F.lit("anonymous")),
                     F.lit(":\n"), F.col("content"))
        )).alias("reviews_block"),
        F.count("*").alias("review_count"),
    )
)

gold = (
    movies_t.alias("m")
    .join(cast_agg, "movie_id", "left")
    .join(reviews_agg, "movie_id", "left")
    .withColumn(
        "doc_text",
        F.concat_ws(
            "\n\n",
            # concat_ws skips nulls, so an absent section simply doesn't appear
            # - no "Tagline: null" leaking into an embedding.
            F.concat(F.col("title"),
                     F.when(F.col("release_date").isNotNull(),
                            F.concat(F.lit(" ("), F.year("release_date").cast("string"),
                                     F.lit(")")))
                      .otherwise(F.lit(""))),
            F.when(F.col("tagline").isNotNull(),
                   F.concat(F.lit("Tagline: "), F.col("tagline"))),
            F.when(F.size(F.coalesce(F.col("genre_names"), F.array())) > 0,
                   F.concat(F.lit("Genres: "),
                            F.concat_ws(", ", F.col("genre_names")))),
            F.when(F.col("runtime_minutes").isNotNull(),
                   F.concat(F.lit("Runtime: "),
                            F.col("runtime_minutes").cast("string"),
                            F.lit(" minutes"))),
            F.concat(F.lit("Overview: "), F.col("overview")),
            F.when(F.size(F.coalesce(F.col("keyword_names"), F.array())) > 0,
                   F.concat(F.lit("Themes: "),
                            F.concat_ws(", ", F.col("keyword_names")))),
            F.when(F.col("cast_line").isNotNull(),
                   F.concat(F.lit("Cast: "), F.col("cast_line"))),
            F.when(F.col("reviews_block").isNotNull(),
                   F.concat(F.lit("What viewers said:\n"), F.col("reviews_block"))),
        ),
    )
    .select(
        "movie_id",
        "title",
        "doc_text",
        F.sha2(F.col("doc_text"), 256).alias("content_hash"),
        F.coalesce(F.col("review_count"), F.lit(0)).cast("int").alias("review_count"),
        F.length("doc_text").alias("char_length"),
        F.current_timestamp().alias("built_at"),
    )
)

gold.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    GOLD_DOCUMENTS
)
print(f"{GOLD_DOCUMENTS}: {spark.table(GOLD_DOCUMENTS).count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checks
# MAGIC
# MAGIC The important one is the length distribution. `CHUNK_SIZE` in `03` is 800, so if the median `char_length` is under ~1000 the chunker is doing nothing and the review join is the thing to look at.

# COMMAND ----------

g = spark.table(GOLD_DOCUMENTS)

display(
    g.select(
        F.count("*").alias("documents"),
        F.round(F.avg("char_length")).alias("avg_chars"),
        F.expr("percentile_approx(char_length, 0.5)").alias("median_chars"),
        F.max("char_length").alias("max_chars"),
        F.round(F.avg("review_count"), 2).alias("avg_reviews"),
        F.sum(F.when(F.col("review_count") > 0, 1).otherwise(0)).alias("with_reviews"),
        F.countDistinct("content_hash").alias("distinct_hashes"),
    )
)

stats = g.select(
    F.count("*").alias("n"),
    F.expr("percentile_approx(char_length, 0.5)").alias("median_chars"),
    F.countDistinct("content_hash").alias("distinct_hashes"),
).collect()[0]

print(f"documents      : {stats['n']}")
print(f"median chars   : {stats['median_chars']}")
print(f"distinct hashes: {stats['distinct_hashes']}")

if stats["n"] == 0:
    raise RuntimeError(
        "gold_movie_documents is empty. Re-run 01_ingest_tmdb_bronze and check "
        "its status summary."
    )

if stats["distinct_hashes"] != stats["n"]:
    # Not fatal, but it means two films assembled to identical text, which
    # almost always signals a join that fanned out wrongly.
    print(
        f"WARNING: {stats['n'] - stats['distinct_hashes']} documents share a "
        "content_hash with another - check the cast/review joins."
    )

if stats["median_chars"] < 800:
    print(
        "WARNING: median document is shorter than CHUNK_SIZE (800), so most "
        "films will produce a single chunk. Check that silver_reviews is "
        "populated - reviews are what make chunking meaningful here."
    )

print("\nSilver/gold build complete. Next: pipeline/03_embed_and_publish.py")