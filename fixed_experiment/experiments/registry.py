"""
Registry of frozen Optuna trials + helpers to turn JSON into train.py argv.

Experiment JSON layout (see ../config/*.json):
  dataset, source_*, hyperparameters{}, train_flags{}
Optional keys:
  silver_span_cache — relative path under recursive_ITHP (default: datasets/{dataset}_silver_spans.pkl)
  extra_cli — dict of extra train.py flags for ablations (merged last)
"""

from __future__ import annotations

import json
from pathlib import Path

# Preset name -> path relative to fixed_experiment/
PRESET_CONFIG_FILES: dict[str, str] = {
    "mosi_refine2_t278": "config/mosi_refine2_best_trial278.json",
    "mosei_t95": "config/mosei_best_trial95.json",
}


def fixed_experiment_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ithp_root() -> Path:
    """``ITHP/recursive_ITHP`` — parent of embedded ``fixed_experiment``."""
    return fixed_experiment_root().parent


def resolve_config_path(preset_or_path: str) -> Path:
    """Preset short name or absolute/relative path to a JSON config."""
    root = fixed_experiment_root()
    if preset_or_path in PRESET_CONFIG_FILES:
        return root / PRESET_CONFIG_FILES[preset_or_path]
    p = Path(preset_or_path)
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def load_experiment_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def default_silver_cache_rel(dataset: str) -> str:
    return f"datasets/{dataset}_silver_spans.pkl"


def _flatten_experiment(exp: dict) -> dict:
    """Single dict of train.py argument names -> values."""
    dataset = exp["dataset"]
    h = dict(exp.get("hyperparameters") or {})
    tf = dict(exp.get("train_flags") or {})
    out: dict = {"dataset": dataset, **tf, **h}
    if "silver_span_cache" in exp and exp["silver_span_cache"]:
        out["silver_span_cache"] = exp["silver_span_cache"]
    else:
        out.setdefault("silver_span_cache", default_silver_cache_rel(dataset))
    extra = exp.get("extra_cli") or {}
    if not isinstance(extra, dict):
        raise TypeError("extra_cli must be a JSON object")
    out.update(extra)
    return out


def flatten_experiment(exp: dict) -> dict:
    """Public alias: experiment JSON → flat train.py argument dict."""
    return _flatten_experiment(exp)


_TRAIN_ARG_ORDER = [
    "dataset",
    "n_epochs",
    "silver_span_cache",
    "merge_trace_samples",
    "selection_metric",
    "early_stopping_patience",
    "seed",
    "learning_rate",
    "p_beta",
    "p_gamma",
    "B0_dim",
    "B1_dim",
    "max_recursion_depth",
    "halting_threshold",
    "dropout_prob",
    "silver_span_loss_weight",
    "syntax_temperature",
]


def build_train_argv(flat: dict, *, extras: dict | None = None) -> list[str]:
    """
    Build argv tokens after ``train.py`` (same flag order as scripts/optuna_search.build_train_command
    for the core MOSI/MOSEI fields, then any remaining keys).
    """
    m = dict(flat)
    if extras:
        m.update(extras)

    def emit(key: str) -> list[str]:
        if key not in m:
            return []
        return [f"--{key}", str(m.pop(key))]

    parts: list[str] = []
    for key in _TRAIN_ARG_ORDER:
        parts.extend(emit(key))
    # Remaining keys (e.g. --model, ablation-only flags), stable order for logs
    for key in sorted(m.keys()):
        parts.extend([f"--{key}", str(m[key])])
    return parts


def parse_cli_overrides(tokens: list[str]) -> dict:
    """
    Parse ``--name value`` pairs from an argv tail (unknown args).
    Values are coerced: int -> float -> str.
    """
    out: dict = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            i += 1
            continue
        key = t[2:].replace("-", "_")
        if "=" in key:
            k, _, v = key.partition("=")
            out[k] = _coerce(v)
            i += 1
            continue
        if i + 1 >= len(tokens):
            raise ValueError(f"Missing value for {t}")
        out[key] = _coerce(tokens[i + 1])
        i += 2
    return out


def _coerce(s: str):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s
