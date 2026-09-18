"""Portable path configuration for the public reproducibility package.

Set ``LOESS_RAW_DATA_DIR`` when the third-party source datasets are stored
outside the repository.  Derived data distributed with the package are in
``data/``; newly generated figures are written to ``outputs/figures/``.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = Path(os.environ.get("LOESS_RAW_DATA_DIR", DATA_DIR / "raw"))

TEMPORAL_DATA_DIR = DATA_DIR / "temporal_change"
PROCESS_DATA_DIR = DATA_DIR / "process_associations"
ROBUSTNESS_DATA_DIR = DATA_DIR / "robustness"
FIGURE_SOURCE_DIR = DATA_DIR / "figure_source"

OUTPUT_DIR = Path(os.environ.get("LOESS_OUTPUT_DIR", REPO_ROOT / "outputs"))
FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"
