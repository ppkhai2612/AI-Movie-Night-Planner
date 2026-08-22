"""
Lakebase (Databricks-managed Postgres) connection helper.

Connects using a single LAKEBASE_URL (a standard Postgres connection URL,
e.g. postgresql://role:password@host:5432/databricks_postgres?sslmode=require)
pointing at a native Postgres role with a static, non-expiring password. That
keeps setup to one secret instead of five separate env vars.

**Driver: pg8000, everywhere.** The same SQL in movie_db.py runs inside the
FastAPI app, inside the MCP server, and inside the Spark pipeline notebooks.
psycopg2's native C extensions crash the kernel on Databricks *Serverless*
(SIGABRT 134), which is exactly where pipeline/03_embed_and_publish.py runs, so
the whole project uses pg8000 - pure Python, no C extensions, works everywhere.

pg8000.dbapi is DBAPI 2.0 with the `format` paramstyle, so callers write
ordinary `%s` placeholders exactly as they would under psycopg2. The one thing
it lacks is psycopg2's RealDictCursor, so rows_as_dicts() below fills in.

Credential resolution, in order:

    1. LAKEBASE_URL in the environment  - local development, no Databricks auth
    2. the Databricks secret scope/key  - how it works when deployed

The Databricks Secrets API base64-encodes secrets *on read* - whatever you
stored, `get_secret(...).value` comes back base64 - so this decodes once.
setup_secrets.py therefore stores the value RAW; encoding on write too would
double-encode it, and the decode here would hand back base64 text instead of a
connection URL.

The WorkspaceClient is constructed lazily: building it at import time raises
`ValueError: default auth: cannot configure default credentials` on a machine
with no Databricks config, which would make this module unimportable locally
even when LAKEBASE_URL is set.
"""

import base64
import os
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse

import pg8000.dbapi

# sslmode values that mean "don't negotiate TLS". Lakebase always wants
# sslmode=require, but a local Postgres started for testing has no certificate,
# and pg8000 fails the handshake rather than falling back - so the URL has to be
# able to say so. Anything else (including an absent sslmode) gets TLS.
_NO_SSL_MODES = {"disable", "allow"}

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "movie-planner")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

_w = None


def _client():
    """Lazily construct the WorkspaceClient (avoids needing creds at import)."""
    global _w
    if _w is None:
        from databricks.sdk import WorkspaceClient

        _w = WorkspaceClient()
    return _w


def lakebase_url() -> str:
    """The Postgres connection URL, from the environment or the secret scope."""
    env_url = os.environ.get("LAKEBASE_URL")
    if env_url:
        return env_url
    secret = _client().secrets.get_secret(scope=_SCOPE, key=_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


@contextmanager
def get_connection():
    """Yield a pg8000 connection to Lakebase, always closing it afterwards."""
    parsed = urlparse(lakebase_url())
    sslmode = (parse_qs(parsed.query).get("sslmode") or ["require"])[0].lower()
    conn = pg8000.dbapi.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        database=parsed.path.lstrip("/") or "databricks_postgres",
        user=parsed.username,
        password=parsed.password,
        # True is pg8000's "use a default TLS context", i.e. sslmode=require.
        ssl_context=None if sslmode in _NO_SSL_MODES else True,
    )
    try:
        yield conn
    finally:
        conn.close()


def rows_as_dicts(cur) -> list[dict]:
    """Name the columns of a pg8000 result set, the way RealDictCursor would.

    pg8000 reports column names in cursor.description as str on current versions
    and as bytes on older ones, so decode defensively.
    """
    if cur.description is None:
        return []
    columns = [
        col[0].decode("utf-8") if isinstance(col[0], bytes) else col[0]
        for col in cur.description
    ]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def run_query(sql: str, params: tuple | list | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or [])
        return rows_as_dicts(cur)


def run_write(sql: str, params: tuple | list | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected row count"""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params or [])
        conn.commit()
        return cur.rowcount