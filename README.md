# AI Movie Night Planner

Users create a group, rate movies, describe what they want to watch, and ask an agent to recommend something everyone will enjoy. The agent searches a semantically-indexed film catalogue, explains its pick, and writes it back - so the recommendation, the watchlist and the ratings are all still there tomorrow.

## Requirements

| **Requirement** | **Where it lives** | **Notes** |
|-|-|-|
| **A data pipeline in Spark** | `pipeline/01 -> 02 -> 03` ||
| **A third-party API** | `tmdb_client.py` | |
| **Unstructured data processing** | `pipeline/02` + `03` ||
| **A Databricks App with a frontend** | `app/` ||
| **An AI agent that does stuff** | `mcp_server/` + `agent/` ||

## Architecture

```
   TMDB API  (api.themoviedb.org/3, Bearer token, ~40 req/s)
        │
        │  [1] pipeline/01_ingest_tmdb_bronze.py      ← Spark, mapInPandas fan-out
        ▼
   bronze_tmdb_movies / bronze_tmdb_genres     (raw JSON, Delta, Unity Catalog)
        │
        │  [2] pipeline/02_build_silver_gold.py       ← Spark, no network
        ├─► bronze_tmdb_parsed   (from_json once; stands in for .cache())
        ▼
   silver_movies · silver_genres · silver_movie_genres
   silver_cast · silver_keywords · silver_reviews
   gold_movie_documents  (movie_id, doc_text, content_hash)
        │
        │  [3] pipeline/03_embed_and_publish.py       ← Spark pandas_udf + pg8000
        ▼
   ┌─────────────────── Lakebase (Postgres + pgvector) ───────────────────┐
   │  catalogue   movies · genres · movie_genres · movie_cast            │
   │              movie_documents · movie_embeddings VECTOR(384) + HNSW  │
   │  application users · groups · group_members · ratings               │
   │              watchlist_items · recommendations                      │
   └──────────────┬───────────────────────────────┬──────────────────────┘
                  │                               │
        app/  Databricks App #1          mcp_server/  Databricks App #2
        FastAPI + single-page UI          FastMCP, 10 tools (6 read / 4 write)
                  │                               │
                  │                        Agent Bricks agent
                  │                        (external MCP server)
                  │                               │
                  └───── recommendations table ◄──┘
                         agent writes · app renders
```

**`movie_db.py` owns every SQL statement in the project**. The app, the MCP server and the Spark pipeline all go through it. That is what makes the two halves provably consistent: when the agent says a film is on the watchlist, it ran the same statement the web page renders from, and the semantic search the user sees is the same ranking the agent reasons over

## Repository layout



## The design decisions worth explaining



## Setup



## Verification

### Known limitations



TMDB API — movies, actors, genres, posters, plot summaries, reviews, trailers, and streaming-provider availability.




Context engineering

Embed plot summaries, keywords, cast information, and reviews.
Retrieve movies using semantic requests such as "a funny sci-fi movie that isn't too violent and is under two hours."
Agent capabilities

Search and explain recommendations.
Compare several movies.
Add a movie to the group watchlist.
Record ratings after the group watches it.
Avoid movies already watched or disliked by group members.