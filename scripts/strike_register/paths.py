"""Shared path configuration for the 08.07.2026 strike-register pipeline.

Every script in this directory resolves its inputs/outputs through this
module instead of a hardcoded absolute path, so the whole pipeline runs with
cwd anywhere:

    .venv/bin/python scripts/strike_register/<script>.py

Roots:
  DATA_ROOT   -- env ROWII_STRIKE_DATA, else
                 <workspace>/data/illwerke-080726/20260708 Messung
                 (rowii-monitor sits at <workspace>/repos/rowii-monitor; the
                 raw burst files are read directly via Stream/glob, not via
                 rowii.io.dataset.discover, so this is intentionally its own
                 knob and independent of the repo-wide ROWII_DATA_ROOT).
  OUTPUT_ROOT -- env ROWII_STRIKE_OUT, else <repo>/results/strike-register/
  GROUNDTRUTH -- <repo>/docs/groundtruth/ (annotated marks CSVs and
                 verification-080726/ already live there; not overridable,
                 it is versioned data checked into the repo).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # scripts/strike_register/.. -> repo root

DATA_ROOT = Path(os.environ.get(
    "ROWII_STRIKE_DATA",
    str(REPO_ROOT.parent.parent / "data" / "illwerke-080726" / "20260708 Messung"),
))
OUTPUT_ROOT = Path(os.environ.get(
    "ROWII_STRIKE_OUT", str(REPO_ROOT / "results" / "strike-register")))
GROUNDTRUTH = REPO_ROOT / "docs" / "groundtruth"

SESSION_DIR = {"ST": DATA_ROOT / "ST_STRIKES", "PU": DATA_ROOT / "PU_STRIKES"}


def ensure_rowii_importable() -> None:
    """Make ``import rowii...`` work with or without an editable install.

    The repo venv normally has ``rowii`` importable directly (``pip install
    -e .`` registers ``src/`` via the package finder, independent of cwd).
    This is only a defensive fallback for a bare interpreter: insert
    ``<repo>/src``, derived from ``__file__`` rather than a hardcoded
    absolute path.
    """
    try:
        import rowii  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(REPO_ROOT / "src"))
