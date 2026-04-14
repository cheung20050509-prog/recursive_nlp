#!/usr/bin/env python
"""Random hyperparameter search for ITHP recursive model.

Usage:
    python scripts/random_search.py --dataset mosi --gpu 3 --n_trials 20
    python scripts/random_search.py --dataset mosei --gpu 2 --n_trials 20
"""

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

SEARCH_SPACE = {
    "learning_rate": [5e-6, 1e-5, 2e-5, 3e-5],
    "p_beta": [4, 8, 16],
    "p_gamma": [16, 32, 64],
    "B0_dim": [64, 128, 256],
    "B1_dim": [32, 64, 128],
    "dropout_prob": [0.3, 0.5],
    "silver_span_loss_weight": [0.05, 0.1, 0.2],
    "syntax_temperature": [0.5, 1.0, 2.0],
}

METRIC_DIRECTIONS = {
    "acc7": "max",
    "acc2_zero": "max",
    "f1_score_zero": "max",
    "acc2_no_zero": "max",
    "f1_score_no_zero": "max",
    "mae": "min",
    "corr": "max",
    "test_acc": "max",
}

METRIC_ALIASES = {
    "acc2_no_zero": ["acc2_no_zero", "test_acc"],
    "f1_score_no_zero": ["f1_score_no_zero", "f1_score"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mosi", "mosei"])
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--n_trials", default=20, type=int)
    parser.add_argument("--n_epochs", default=20, type=int)
    parser.add_argument("--seed", default=128, type=int)
    parser.add_argument("--output_dir", default="search_results", type=str)
    parser.add_argument(
        "--primary_metric",
        default="acc2_no_zero",
        choices=sorted(METRIC_DIRECTIONS.keys()),
        help="Metric used to rank trials",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already-completed trials")
    return parser.parse_args()


def sample_configs(n_trials, seed):
    rng = random.Random(seed)
    configs = []
    for _ in range(n_trials):
        config = {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}
        configs.append(config)
    return configs


PYTHON = "/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python"


def build_train_command(dataset, config, n_epochs, trial_log_path):
    cmd = [
        PYTHON, "-u", "train.py",
        "--dataset", dataset,
        "--n_epochs", str(n_epochs),
        "--silver_span_cache", f"datasets/{dataset}_silver_spans.pkl",
        "--merge_trace_samples", "0",
    ]
    for key, value in config.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def parse_test_line(log_path):
    """Extract TEST metrics from log file."""
    test_line = None
    with open(log_path, "r") as f:
        for line in f:
            if line.startswith("TEST:"):
                test_line = line.strip()

    if test_line is None:
        return None

    metrics = {}
    for pair in test_line[len("TEST: "):].split(", "):
        key, value = pair.split(":")
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def metric_value(metrics, metric_name):
    if metrics is None:
        return None

    for candidate in METRIC_ALIASES.get(metric_name, [metric_name]):
        value = metrics.get(candidate)
        if value is not None:
            return value

    return None


def format_metric(metrics, metric_name):
    value = metric_value(metrics, metric_name)
    if value is None:
        return "NA"

    return f"{value:.4f}"


def sort_key(result, metric_name):
    metrics = result.get("metrics")
    value = metric_value(metrics, metric_name)
    if value is None:
        return float("inf")

    if METRIC_DIRECTIONS[metric_name] == "max":
        return -value

    return value


def main():
    args = parse_args()
    work_dir = Path(__file__).resolve().parent.parent
    os.chdir(work_dir)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.jsonl"

    # Determine completed trial indices
    completed = set()
    if args.resume and summary_path.exists():
        with open(summary_path, "r") as f:
            for line in f:
                record = json.loads(line)
                completed.add(record["trial"])

    configs = sample_configs(args.n_trials, args.seed)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"=== Random Search: {args.dataset}, {args.n_trials} trials, GPU {args.gpu} ===")
    print(f"Output: {output_dir}")
    print(f"Primary metric: {args.primary_metric} ({METRIC_DIRECTIONS[args.primary_metric]})")
    print(f"Search space size: {' x '.join(str(len(v)) for v in SEARCH_SPACE.values())} = "
          f"{sum(1 for _ in itertools.product(*SEARCH_SPACE.values()))} total combinations")
    print()

    for trial_idx, config in enumerate(configs):
        if trial_idx in completed:
            print(f"[Trial {trial_idx}] SKIP (already completed)")
            continue

        trial_log = output_dir / f"trial_{trial_idx:03d}.log"
        cmd = build_train_command(args.dataset, config, args.n_epochs, trial_log)

        print(f"[Trial {trial_idx}] Config: {config}")
        print(f"[Trial {trial_idx}] Log: {trial_log}")
        print(f"[Trial {trial_idx}] Starting...")

        t0 = time.time()
        with open(trial_log, "w") as log_file:
            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(work_dir),
            )
        elapsed = time.time() - t0

        metrics = parse_test_line(trial_log)
        result = {
            "trial": trial_idx,
            "config": config,
            "metrics": metrics,
            "elapsed_seconds": round(elapsed, 1),
            "returncode": proc.returncode,
        }

        with open(summary_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        if metrics:
            print(f"[Trial {trial_idx}] DONE in {elapsed:.0f}s — "
                  f"acc2_no_zero={format_metric(metrics, 'acc2_no_zero')}, "
                  f"f1_no_zero={format_metric(metrics, 'f1_score_no_zero')}, "
                  f"mae={metrics.get('mae', '?')}, "
                  f"best_epoch={metrics.get('best_epoch', '?')}")
        else:
            print(f"[Trial {trial_idx}] FAILED (returncode={proc.returncode}, elapsed={elapsed:.0f}s)")

        print()

    # Print final leaderboard
    print("\n=== LEADERBOARD ===")
    if summary_path.exists():
        results = []
        with open(summary_path, "r") as f:
            for line in f:
                results.append(json.loads(line))

        valid = [r for r in results if r["metrics"] is not None]
        valid.sort(key=lambda result: sort_key(result, args.primary_metric))

        for rank, r in enumerate(valid[:10], 1):
            m = r["metrics"]
            print(
                f"  #{rank} Trial {r['trial']}: "
                f"acc7={format_metric(m, 'acc7')}, "
                f"acc2_zero={format_metric(m, 'acc2_zero')}, "
                f"f1_zero={format_metric(m, 'f1_score_zero')}, "
                f"acc2_no_zero={format_metric(m, 'acc2_no_zero')}, "
                f"f1_no_zero={format_metric(m, 'f1_score_no_zero')}, "
                f"mae={format_metric(m, 'mae')}, "
                f"corr={format_metric(m, 'corr')}, "
                f"best_epoch={m.get('best_epoch', '?')}, "
                  f"lr={r['config']['learning_rate']}, beta={r['config']['p_beta']}, "
                  f"gamma={r['config']['p_gamma']}, B0={r['config']['B0_dim']}, "
                  f"B1={r['config']['B1_dim']}, silver={r['config']['silver_span_loss_weight']}, "
                  f"temp={r['config']['syntax_temperature']}"
            )


if __name__ == "__main__":
    main()
