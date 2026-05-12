"""Resolve ``recursive_ITHP`` (datasets, pretrained, pickles) from this package location."""

from __future__ import annotations

from pathlib import Path


def ithp_workdir() -> Path:
    """``ITHP/recursive_ITHP`` — parent directory of ``fixed_experiment``."""
    return Path(__file__).resolve().parents[2]
