"""
One-time setup: store this project's two secrets in a Databricks secret scope.

    movie-planner/lakebase-url   Postgres connection URL for Lakebase
    movie-planner/tmdb-token     TMDB API Read Access Token (v4 Bearer token)

Encoding note - this is the part that is easy to get wrong.

**Write the value RAW.** The Databricks Secrets API base64-encodes secrets
*on read*: `w.secrets.get_secret(...).value` is always base64, whatever you
put in. So the readers (lakebase.lakebase_url() and tmdb_client._token())
base64-*decode* once, and that round-trips only if the stored value was raw.

Encoding here as well would double-encode: the reader decodes once and gets
back the base64 text instead of the secret. For a TMDB token that surfaces as
an opaque 401; for the Lakebase URL, as a connection failure.

Worse, `base64.b64decode` silently DISCARDS characters outside the base64
alphabet - and a TMDB v4 token is a JWT full of '.', '-' and '_'. So a
mis-encoded token doesn't raise, it comes back as a mangled fragment.

The verification step at the end of this script reads each secret back and
checks it matches what you typed, so none of the above can fail silently.

Getting a TMDB token: sign up free at https://www.themoviedb.org/signup, then
Settings -> API -> request an API key (choose "Developer", noncommercial /
educational use). Copy the **API Read Access Token**, not the shorter v3 API
key - this project authenticates with `Authorization: Bearer <token>`.

Run locally with the Databricks CLI configured, or from a notebook:
    python setup_secrets.py
"""

import base64
import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

SCOPE = "movie-planner"
LAKEBASE_KEY = "lakebase-url"
TMDB_KEY = "tmdb-token"

w = WorkspaceClient()

existing = {s.name for s in w.secrets.list_scopes()}
if SCOPE not in existing:
    print(f"Creating secret scope {SCOPE!r}...")
    w.secrets.create_scope(scope=SCOPE)
else:
    print(f"Secret scope {SCOPE!r} already exists, reusing it.")


def _store(key: str, value: str) -> None:
    """Store RAW - the API base64-encodes on read. See the module docstring."""
    w.secrets.put_secret(scope=SCOPE, key=key, string_value=value)
    print(f"  stored {SCOPE}/{key} ({len(value)} chars)")


def _verify(key: str, expected: str) -> bool:
    """Read the secret back the way the apps do and check it round-trips.

    This is the whole point of the exercise: it exercises the identical code
    path as lakebase.lakebase_url() / tmdb_client._token(), so a
    write/read encoding mismatch is caught here rather than as a 401 halfway
    through a pipeline run.
    """
    try:
        raw = w.secrets.get_secret(scope=SCOPE, key=key).value
        got = base64.b64decode(raw).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ {SCOPE}/{key}: could not read it back - {exc}")
        return False

    if got == expected:
        print(f"  ✓ {SCOPE}/{key} round-trips ({len(got)} chars)")
        return True

    print(f"  ✗ {SCOPE}/{key} does NOT round-trip:")
    print(f"      stored {len(expected)} chars, read back {len(got)}")
    print(f"      expected {expected[:16]}... got {got[:16]}...")
    if got.startswith("ZXlK") or len(got) > len(expected):
        print("      Looks double-encoded - the value was base64'd on write "
              "as well as on read.")
    return False


# --- Lakebase -----------------------------------------------------------------
url = getpass.getpass(
    "Lakebase connection URL "
    "(postgresql://role:password@host:5432/databricks_postgres?sslmode=require)\n"
    "  - press Enter to skip: "
).strip()

if url:
    if not url.startswith("postgres"):
        raise SystemExit(
            f"That doesn't look like a Postgres URL (got {url[:20]!r}...). "
            "Nothing was stored."
        )
    _store(LAKEBASE_KEY, url)
else:
    print(f"  skipped {SCOPE}/{LAKEBASE_KEY}")


# --- TMDB ---------------------------------------------------------------------
token = getpass.getpass(
    "TMDB API Read Access Token (the long eyJ... one from Settings -> API)\n"
    "  - press Enter to skip: "
).strip()

if token:
    # v4 read access tokens are JWTs. The v3 key is a 32-char hex string and
    # will not work with `Authorization: Bearer`, so catch that swap early -
    # it otherwise surfaces as an opaque 401 halfway through the pipeline.
    if not token.startswith("eyJ"):
        raise SystemExit(
            "That looks like a v3 API key, not a v4 Read Access Token "
            "(expected it to start with 'eyJ'). Nothing was stored for TMDB. "
            "Find the Read Access Token under Settings -> API on themoviedb.org."
        )
    _store(TMDB_KEY, token)
else:
    print(f"  skipped {SCOPE}/{TMDB_KEY}")


# Both Databricks Apps run as service principals that need read access.
w.secrets.put_acl(
    scope=SCOPE,
    principal="users",
    permission=workspace.AclPermission.READ,
)
print(f"Granted READ on scope {SCOPE!r} to 'users'.")


# --- Verify ------------------------------------------------------------------
# Read each secret back through the identical code path the apps use, and
# confirm it matches what was typed. A write/read encoding mismatch otherwise
# only shows up much later as a 401 or a connection failure.
print("\nVerifying...")
ok = True
if url:
    ok &= _verify(LAKEBASE_KEY, url)
if token:
    ok &= _verify(TMDB_KEY, token)

if not (url or token):
    print("  nothing was stored, nothing to verify")
elif ok:
    print("\nBoth halves agree: stored raw, read back with one base64 decode.")
else:
    raise SystemExit(
        "\nA secret did not round-trip. Do NOT deploy until this is fixed - "
        "the apps read secrets exactly the way _verify() just did.\n"
        "Most likely cause: the value was stored base64-encoded by an earlier "
        "script (the day-3 homework's setup_secrets.py did this). Re-run this "
        "script and paste the raw value to overwrite it."
    )

print()
print("Next: apply sql/01_schema_catalog.sql then sql/02_schema_app.sql to Lakebase.")