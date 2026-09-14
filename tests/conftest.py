"""Shared pytest setup.

fusion_avsr is not installed as a package (no pyproject.toml/setup.py
yet), so it is only importable if the repo root is on sys.path. Adding
it here means `pytest` works from any working directory, without
requiring callers to set PYTHONPATH themselves.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
