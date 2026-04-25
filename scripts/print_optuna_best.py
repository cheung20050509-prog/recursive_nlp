#!/usr/bin/env python3
"""Load an Optuna study from SQLite and print best_trial hyperparameters (and metrics).

Typical storage from this repo (see scripts/optuna_search.py):
  <output_dir>/<dataset>/optuna_study.sqlite3
  study name:  {study_prefix}_{dataset}_{primary_metric}  (default prefix: ithp)

Examples:
  python scripts/print_optuna_best.py --storage optuna_results_mae/mosei/optuna_study.sqlite3
  python scripts/print_optuna_best.py --storage /path/to/optuna_study.sqlite3 --study-name ithp_mosei_mae
  python scripts/print_optuna_best.py --storage optuna_study.sqlite3 --list-studies

If the SQLite file was reset or is a partial copy, the full-search best config may still exist in the
Optuna *parent* log (stdout from scripts/optuna_search.py), e.g. lines like
`[tpe][Trial N] Config: {...}` or `Trial N finished with value: ... and parameters: {...}`.
For this repo, MOSEI ~0.503 is recorded in `optuna_mosei_mae_gpu3.log` and copied to
`configs/mosei_optuna_best_trial117.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna


def storage_uri_from_path(path: Path) -> str:
    path = path.resolve()
    return f"sqlite:///{path.as_posix()}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Print Optuna best_trial config from SQLite.")
    parser.add_argument(
        "--storage",
        type=str,
        required=True,
        help="Path to optuna_study.sqlite3, or a full RDB URL (e.g. sqlite:////abs/path.db)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name (e.g. ithp_mosei_mae). If omitted and only one study exists, use it; else list and exit.",
    )
    parser.add_argument(
        "--list-studies",
        action="store_true",
        help="Only list study names and trial counts, then exit.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="If >0, also print this many top completed trials by objective (after --list-studies logic).",
    )
    args = parser.parse_args()

    if args.storage.startswith("sqlite://") or args.storage.startswith("mysql://") or args.storage.startswith("postgresql://"):
        storage = args.storage
    else:
        storage = storage_uri_from_path(Path(args.storage))

    summaries = optuna.get_all_study_summaries(storage=storage)
    if not summaries:
        print("No studies found in storage.", file=sys.stderr)
        return 1

    if args.list_studies:
        for s in summaries:
            print(f"{s.study_name}\tn_trials={s.n_trials}")
        return 0

    study_name = args.study_name
    if study_name is None:
        if len(summaries) == 1:
            study_name = summaries[0].study_name
            print(f"Using single study: {study_name}\n", file=sys.stderr)
        else:
            print("Multiple studies; pass --study-name. Use --list-studies:\n", file=sys.stderr)
            for s in summaries:
                print(f"  {s.study_name}  (n_trials={s.n_trials})")
            return 2

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f"Study not found: {study_name!r}", file=sys.stderr)
        print("Available:", [s.study_name for s in summaries], file=sys.stderr)
        return 1

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print("No COMPLETE trials in this study.", file=sys.stderr)
        return 1

    best = study.best_trial
    out = {
        "study_name": study_name,
        "storage": storage,
        "best_trial_number": best.number,
        "objective_value": best.value,
        "config": best.user_attrs.get("config"),
        "metrics": best.user_attrs.get("metrics"),
        "phase": best.user_attrs.get("phase"),
        "log_path": best.user_attrs.get("log_path"),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.top > 0:
        direction = study.direction
        completed_sorted = sorted(
            complete,
            key=lambda t: t.value if t.value is not None else (float("inf") if direction == optuna.study.StudyDirection.MINIMIZE else float("-inf")),
            reverse=(direction == optuna.study.StudyDirection.MAXIMIZE),
        )
        top = completed_sorted[: args.top]
        rows = [
            {
                "trial": t.number,
                "value": t.value,
                "config": t.user_attrs.get("config"),
            }
            for t in top
        ]
        print(f"\n--- top {args.top} by objective ---", file=sys.stderr)
        print(json.dumps(rows, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
