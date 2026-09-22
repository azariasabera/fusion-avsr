"""Repo-relative paths shared across scripts, modules, and notebooks.

Derived from this file's own location rather than the current working
directory, so every caller resolves the same paths regardless of where
the repo is checked out or which directory a script/notebook happens to
be run from.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Where built manifests (and the GRID word-segment table) are cached as
# CSVs. See `fusion_avsr.data.manifest_builder.load_or_build_manifest`:
# the first call for a given manifest builds and saves it here; every
# later call just loads the cached file.
MANIFEST_DIR = REPO_ROOT / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
