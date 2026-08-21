"""
TMDB API client - the project's third-party data source.

Every HTTP call to themoviedb.org lives in this module. Nothing else in the
repo imports `requests`; the Spark pipeline and the MCP tools both go through
`TMDBClient`. This is what makes the pipeline testable against
recorded fixtures instead of the live API

Auth
----
TMDB offers two credentials. This project uses the **v4 API Read Access
Token** as `Authorization: Bearer <token>` - the shorter 32-char v3 key goes in
a query parameter and will 401 against a Bearer header. `setup_secrets.py`
rejects the v3 key up front so the mistake surfaces at setup rather than
halfway through a pipeline run.

The token comes from the environment (`TMDB_TOKEN`, local dev) or the
Databricks secret scope `database/tmdb-token`.

The Secrets API base64-encodes on read, so `_token()` decodes once and
setup_secrets.py stores the value RAW. Getting this wrong is nastier here than
elsewhere: a TMDB v4 token is a JWT full of '.', '-' and '_', none of which are
in the base64 alphabet, and `base64.b64decode` *silently discards* them - so a
double-encoded token comes back as a mangled fragment rather than an error, and
surfaces as an opaque 401.

Rate limits
-----------
TMDB retired its published 40-req/10s limit in 2019 and now enforces an
unpublished ceiling "somewhere in the 40 requests per second range", with no
daily cap. A full catalogue build makes one request per movie, so the ceiling
is reachable: requests are spaced by `TMDB_REQUEST_SPACING` (default 50 ms,
~20 req/s) and 429 is in the retry status list with backoff, per TMDB's request
that callers honour it.

Endpoints used
--------------
    /discover/movie   the catalogue seed - popular titles, paged
    /movie/{id}       full details, with credits+keywords+reviews+providers
                      folded in via append_to_response (one request, not five;
                      TMDB caps the list at 20 items)
    /search/movie     title lookup, for when the agent is asked about a film
                      the catalogue doesn't have yet

Run locally to smoke-test the credential and the response shapes:
    $ TMDB_TOKEN=eyJ...
    $ python tmdb_client.py "Arrival"
"""

import os
import base64
import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("tmdb-client")

BASE_URL = os.environ.get("TMDB_BASE_URL", "https://api.themoviedb.org/3")
IMAGE_BASE_URL = os.environ.get("TMDB_IMAGE_BASE_URL", "https://image.tmdb.org/t/p")
LANGUAGE = os.environ.get("TMDB_LANGUAGE", "en-US")

_SECRET_SCOPE = os.environ.get("TMDB_SECRET_SCOPE", "movie-planner")
_SECRET_KEY = os.environ.get("TMDB_SECRET_KEY", "tmdb-token")

_DEFAULT_TIMEOUT = 30
_SECONDS_BETWEEN_REQUESTS = float(os.environ.get("TMDB_REQUEST_SPACING", "0.05"))

# One /movie/{id} call carries all of these. TMDB allows up to 20 appended
# namespaces; four keeps the payload manageable and covers every field the
# silver/gold layer reads.
APPEND_TO_RESPONSE = "credits,keywords,reviews,watch/providers"

# How many top-billed actors to keep. The full credits list runs to hundreds of
# rows per film and nothing downstream needs the second boom operator.
CAST_LIMIT = 10

# TMDB paginates /discover at 20 results per page and refuses page > 500.
RESULTS_PER_PAGE = 20
MAX_DISCOVER_PAGE = 500

_w = None


class TMDBError(RuntimeError):
    """The TMDB API itself failed - network error, 5xx, or malformed JSON.

    Callers surface this as a transient outage. It is deliberately distinct
    from UnknownMovieError: "TMDB is down" and "that film doesn't exist" call
    for different responses from the agent.
    """


class UnknownMovieError(ValueError):
    """A movie id or title could not be found on TMDB (a 404, or no search hits)."""


def _client():
    """Lazily construct the WorkspaceClient (avoids needing creds at import)."""
    global _w
    if _w is None:
        from databricks.sdk import WorkspaceClient

        _w = WorkspaceClient()
    return _w


def _token() -> str:
    """The TMDB read access token, from the environment or the secret scope."""
    env_token = os.environ.get("TMDB_TOKEN")
    if env_token:
        return env_token.strip()
    secret = _client().secrets.get_secret(scope=_SECRET_SCOPE, key=_SECRET_KEY)
    return base64.b64decode(secret.value).decode("utf-8").strip()


def poster_url(poster_path: str | None, size: str = "w342") -> str | None:
    """Turn TMDB's relative poster path into a full image URL.

    TMDB stores `/abc123.jpg` and expects the caller to prepend a base and a
    size. Kept here so the app template doesn't hardcode the CDN host.
    """
    if not poster_path:
        return None
    return f"{IMAGE_BASE_URL}/{size}{poster_path}"
    

class TMDBClient:
    """Thin wrapper around api.themoviedb.org with a retrying, authenticated session."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.timeout = timeout
        self._last_request_at = 0.0

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token or _token()}",
                "Accept": "application/json",
            }
        )
        # 429 is in the list because TMDB explicitly asks callers to honour it;
        # the 5xx codes cover the usual transient blips. backoff_factor=1.0
        # gives 0s / 2s / 4s, which is well inside a pipeline task's patience.
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    # --- transport ---

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a path relative to the API base, returning parsed JSON.
        
        Raises:
            UnknownMovieError: on 404 - the resource genuinely isn't there.
            TMDBError: on anything else, including malformed JSON.
        """
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < _SECONDS_BETWEEN_REQUESTS:
            time.sleep(_SECONDS_BETWEEN_REQUESTS - elapsed)

        url = path if path.startswith("http") else f"{self.base_url}{path}"
        merged = {"language": LANGUAGE, **(params or {})}

        try:
            response = self._session.get(url, params=merged, timeout=self.timeout)
            self._last_request_at = time.monotonic()
            if response.status_code == 404:
                raise UnknownMovieError(f"TMDB has no resource at {path!r}.")
            if response.status_code == 401:
                # Almost always the v3-key-instead-of-v4-token swap, and the
                # raw TMDB message ("Invalid API key") doesn't say so.
                raise TMDBError(
                    "TMDB rejected the credential (401). Check that the stored "
                    "value is the v4 API Read Access Token (starts with 'eyJ'), "
                    "not the shorter v3 API key."
                )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            raise TMDBError(f"TMDB request failed for {path!r}: {exc}") from exc
        except ValueError as exc:  # malformed JSON
            raise TMDBError(f"TMDB returned invalid JSON for {path!r}: {exc}") from exc

    # --- endpoints ---

    def discover_page(self, page: int, **filters: Any) -> dict[str, Any]:
        """One page of /discover/movie, sorted by popularity.

        Args:
            page: 1-based page number; TMDB refuses anything above 500.
            **filters: extra TMDB discover parameters, e.g.
                `primary_release_date_gte="2015-01-01"` (underscores are
                translated to TMDB's dotted `.gte` form).

        Returns:
            The raw TMDB response dict, with `results`, `page`, `total_pages`.
        """
        if not 1 <= page <= MAX_DISCOVER_PAGE:
            raise ValueError(
                f"page must be between 1 and {MAX_DISCOVER_PAGE} (got {page})."
            )
        params: dict[str, Any] = {
            "page": page,
            "sort_by": "popularity.desc",
            "include_adult": "false",
            "include_video": "false",
            # Below this, TMDB's popularity tail is full of records with no
            # overview at all - which would embed to nothing useful.
            "vote_count.gte": 50,
        }
        for key, value in filters.items():
            params[key.replace("_gte", ".gte").replace("_lte", ".lte")] = value
        return self.get("/discover/movie", params)

    def discover_movie_ids(self, pages: int) -> list[int]:
        """Collect TMDB ids from the first `pages` pages of /discover/movie.

        This is the catalogue seed. Deduplicated, because TMDB's popularity
        ordering shifts between requests and the same title can appear on two
        pages of one crawl.
        """
        seen: dict[int, None] = {}
        for page in range(1, pages + 1):
            payload = self.discover_page(page)
            results = payload.get("results") or []
            if not results:
                logger.info("Discover page %s returned nothing; stopping early.", page)
                break
            for item in results:
                movie_id = item.get("id")
                if isinstance(movie_id, int):
                    seen[movie_id] = None
            total_pages = payload.get("total_pages")
            if isinstance(total_pages, int) and page >= total_pages:
                break
        return list(seen)

    def movie_details(self, movie_id: int) -> dict[str, Any]:
        """Full details for one movie, with credits/keywords/reviews/providers.

        Returns the raw TMDB payload. Normalisation into columns happens in
        Spark (pipeline/02_build_silver_gold.py) so the bronze layer keeps
        everything the API said and silver can be rebuilt without re-crawling.

        Raises:
            UnknownMovieError: if TMDB has no such id.
            TMDBError: on any other failure.
        """
        return self.get(
            f"/movie/{movie_id}", {"append_to_response": APPEND_TO_RESPONSE}
        )

    def search_movies(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Title search - used when the agent is asked about a film not in the catalogue.

        Returns a list of trimmed result dicts (id, title, release_date,
        overview, vote_average, poster_path), most relevant first.

        Raises:
            UnknownMovieError: if the search returns no hits at all.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        payload = self.get(
            "/search/movie", {"query": query.strip(), "include_adult": "false"}
        )
        results = payload.get("results") or []
        if not results:
            raise UnknownMovieError(f"TMDB found no movies matching {query!r}.")

        return [
            {
                "movie_id": item.get("id"),
                "title": item.get("title"),
                "release_date": item.get("release_date") or None,
                "overview": item.get("overview"),
                "vote_average": item.get("vote_average"),
                "poster_url": poster_url(item.get("poster_path")),
            }
            for item in results[:limit]
        ]

    def genres(self) -> list[dict[str, Any]]:
        """The full TMDB movie genre list - id and name.

        Fetched once and written to the `genres` table, so movie_genres has
        something to reference. TMDB's list is ~19 entries and changes rarely.
        """
        payload = self.get("/genre/movie/list")
        return payload.get("genres") or []


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    client = TMDBClient()

    if len(sys.argv) > 1:
        term = " ".join(sys.argv[1:])
        hits = client.search_movies(term)
        print(json.dumps(hits, indent=2))
        if hits:
            details = client.movie_details(hits[0]["movie_id"])
            print(f"\n{details.get('title')} ({details.get('release_date')})")
            print(f"  runtime      : {details.get('runtime')} min")
            print(f"  genres       : {[g['name'] for g in details.get('genres', [])]}")
            print(f"  cast         : "
                  f"{[c['name'] for c in details.get('credits', {}).get('cast', [])[:5]]}")
            print(f"  keywords     : "
                  f"{[k['name'] for k in details.get('keywords', {}).get('keywords', [])][:8]}")
            print(f"  reviews      : "
                  f"{details.get('reviews', {}).get('total_results')} total")
    else:
        ids = client.discover_movie_ids(pages=1)
        print(f"Discovered {len(ids)} movie ids: {ids[:10]}...")