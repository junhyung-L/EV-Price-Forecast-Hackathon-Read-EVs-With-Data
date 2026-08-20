"""Portable paths and defaults for the EV-price competition pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("EV_PRICE_DATA_DIR", PROJECT_ROOT / "data"))
RESULTS_DIR = Path(os.getenv("EV_PRICE_RESULTS_DIR", PROJECT_ROOT / "results"))
TRAIN_CSV = Path(os.getenv("EV_PRICE_TRAIN_CSV", DATA_DIR / "train.csv"))
TEST_CSV = Path(os.getenv("EV_PRICE_TEST_CSV", DATA_DIR / "test.csv"))
RANDOM_SEED = 42
TEST_SIZE = 0.20


def result_path(filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / filename
