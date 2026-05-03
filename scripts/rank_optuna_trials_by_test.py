#!/usr/bin/env python3
"""Rank completed Optuna trials by a *test* metric (post-hoc), for valid/test gap analysis.

Supports:

* ``train`` backend: ``optuna_search.py`` / ``train.py`` trials store ``user_attrs['metrics']``
  parsed from the ``TEST:`` log line (e.g. ``mae``, ``corr``, ``F1_score``).
* ``hkt`` backend: ``optuna_hkt_search.py`` trials store ``user_attrs['result']['best']['test']``.

Examples::

    python scripts/rank_optuna_trials_by_test.py --backend train --dataset simsv2 \\
        --sqlite log/4080_restart/simsv2/optuna_study.sqlite3 \\
        --study-name ithp_silver_20260430_114654_simsv2_mae \\
        --test-metric-key mae --direction minimize --top-k 15

    python scripts/rank_optuna_trials_by_test.py --backend hkt --dataset mustard \\
        --sqlite log/4080_restart/mustard/optuna_study.sqlite3 \\
        --study-name ithp_hkt_silver_20260430_114654_mustard_valid_accuracy \\
        --test-metric-key accuracy --direction maximize --top-k 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna


def parse_args():
    p = argparse.ArgumentParser(description="Rank Optuna trials by test metric.")
    p.add_argument("--repo-root", type=str, default="")
    p.add_argument("--sqlite", required=True, help="Path to optuna_study.sqlite3")
    p.add_argument("--study-name", required=True)
    p.add_argument("--backend", choices=["train", "hkt"], required=True)
    p.add_argument("--dataset", default="", help="Label for output only.")
    p.add_argument("--test-metric-key", required=True, help="e.g. mae, corr, accuracy")
    p.add_argument("--direction", choices=["minimize", "maximize"], required=True)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--output-json", default="", help="Optional path to write ranking JSON.")
    return p.parse_args()


def test_metric_train(trial, key: str):
    metrics = trial.user_attrs.get("metrics") or {}
    v = metrics.get(key)
    return float(v) if v is not None else None


def test_metric_hkt(trial, key: str):
    result = trial.user_attrs.get("result") or {}
    best = result.get("best") or {}
    test = best.get("test") or {}
    v = test.get(key)
    if v is None and key == "accuracy" and "acc" in test:
        v = test.get("acc")
    return float(v) if v is not None else None


def main():
    args = parse_args()
    repo = Path(args.repo_root) if args.repo_root else Path(__file__).resolve().parent.parent
    sqlite = Path(args.sqlite)
    if not sqlite.is_file():
        raise FileNotFoundError(sqlite)
    uri = f"sqlite:///{sqlite.resolve()}"
    study = optuna.load_study(study_name=args.study_name, storage=uri)

    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if args.backend == "train":
            tv = test_metric_train(trial, args.test_metric_key)
        else:
            tv = test_metric_hkt(trial, args.test_metric_key)
        if tv is None:
            continue
        key_out = f"test_{args.test_metric_key}"
        rows.append(
            {
                "trial": trial.number,
                "phase": trial.user_attrs.get("phase"),
                "optuna_value": trial.value,
                key_out: tv,
                "config": trial.user_attrs.get("config"),
            }
        )

    reverse = args.direction == "maximize"
    key_out = f"test_{args.test_metric_key}"
    rows.sort(key=lambda r: r[key_out], reverse=reverse)
    top = rows[: max(0, args.top_k)]

    out = {
        "study_name": args.study_name,
        "dataset": args.dataset,
        "backend": args.backend,
        "test_metric_key": args.test_metric_key,
        "direction": args.direction,
        "complete_with_test_metric": len(rows),
        "top_k": top,
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        outp = repo / args.output_json
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
        print(f"Wrote {outp.resolve()}", flush=True)


if __name__ == "__main__":
    main()
