"""
Shared bootstrap for the three pipeline notebooks.

Two jobs, both of which exist because Databricks notebooks are not ordinary
Python scripts:

1. **Import path.** The notebooks live in `pipeline/` but import `movie_db`,
   `lakebase` and `tmdb_client` from the repo root. There is no `__file__` in a
   notebook and the working directory is frequently `/databricks/driver` rather
   than the notebook's folder, so the root has to be *found* rather than
   assumed. `bootstrap()` tries every plausible starting point and walks up
   looking for a directory containing `movie_db.py`.

2. **Config.** `conf()` reads a Databricks widget first, then an environment
   variable, then a default - which is what lets the same file run as a
   scheduled `notebook_task` with `base_parameters`, as an interactive notebook
   with widgets, and as a plain script with env vars.

Import it from a notebook cell with the two-line preamble the notebooks use:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")) or ".")
"""

import os
import sys

_MARKER = "movie_db.py"


def _candidate_dirs() -> list[str]:
    """Plausible directories to start the search from, best guess first."""
    candidates: list[str] = []

    # 1. Running as a script: the file's own directory is authoritative.
    if "__file__" in globals():
        candidates.append(os.path.dirname(os.path.abspath(__file__)))

    # 2. Running as a Databricks notebook: ask Databricks where the notebook
    #    lives. The context returns a workspace path such as
    #    /Users/me@corp.com/capstone/pipeline/01_ingest_tmdb_bronze, which maps
    #    onto the driver filesystem under /Workspace - but not on every runtime,
    #    so try it both ways.
    try:
        ctx = (
            dbutils.notebook.entry_point.getDbutils()  # noqa: F821
            .notebook()
            .getContext()
        )
        notebook_path = ctx.notebookPath().get()
        candidates.append(os.path.dirname("/Workspace" + notebook_path))
        candidates.append(os.path.dirname(notebook_path))
    except Exception:  # noqa: BLE001 - dbutils is absent outside Databricks
        pass

    # 3. Wherever we happen to be running from.
    candidates.append(os.getcwd())
    return candidates


def find_repo_root() -> str:
    """The first candidate directory (or ancestor) that contains movie_db.py."""
    tried: list[str] = []
    for start in _candidate_dirs():
        current = start
        # The marker is one level up from pipeline/; walking a few more costs
        # nothing and rescues a copy left somewhere unexpected.
        for _ in range(5):
            tried.append(current)
            if os.path.isfile(os.path.join(current, _MARKER)):
                return current
            parent = os.path.dirname(current)
            if parent == current:  # filesystem root
                break
            current = parent

    listing = ""
    for directory in dict.fromkeys(tried):
        try:
            names = sorted(os.listdir(directory))[:20]
            listing += f"\n  {directory}: {names}"
        except OSError:
            listing += f"\n  {directory}: <unreadable>"
    raise RuntimeError(
        f"Could not find the repo root (no directory containing {_MARKER!r}).\n"
        f"Searched:{listing}\n"
        "If you imported only the pipeline/ folder into Databricks, import the "
        "whole repo instead - these notebooks import movie_db, lakebase and "
        "tmdb_client from the root."
    )


def bootstrap() -> str:
    """Put the repo root on sys.path and return it."""
    root = find_repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def conf(name: str, default: str | None = None) -> str | None:
    """Read a setting from a Databricks widget, then the environment, then a default.

    Widget first is deliberate: when a job passes `base_parameters`, those
    arrive as widgets, and they should win over whatever happens to be in the
    cluster's environment.
    """
    try:
        value = dbutils.widgets.get(name)  # noqa: F821
        if value not in (None, ""):
            return value
    except Exception:  # noqa: BLE001 - no widget defined, or no dbutils at all
        pass
    return os.environ.get(name.upper(), default)


def conf_int(name: str, default: int) -> int:
    """conf(), coerced to int, tolerating an empty widget."""
    raw = conf(name, str(default))
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def table_name(catalog: str, schema: str, table: str) -> str:
    """Fully-qualified Unity Catalog name."""
    return f"{catalog}.{schema}.{table}"