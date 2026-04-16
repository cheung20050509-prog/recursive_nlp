#!/usr/bin/env python
"""Standalone Optuna search with local phase-2 refinement.

Phase 1:
- import completed random-search results from summary.jsonl when available, or
- run a fresh broad random Optuna phase.

Phase 2:
- build a narrowed discrete search space around the best phase-1 config, then
- run TPE only inside that local neighborhood.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import optuna


SEARCH_SPACE = {
    "learning_rate": [5e-6, 1e-5, 2e-5, 3e-5],
    "p_beta": [4, 8, 16],
    "p_gamma": [16, 32, 64],
    "B0_dim": [64, 128, 256],
    "B1_dim": [32, 64, 128],
    "max_recursion_depth": [3, 4, 5],
    "halting_threshold": [0.02, 0.0285, 0.04, 0.06],
    "dropout_prob": [0.3, 0.5],
    "silver_span_loss_weight": [0.05, 0.1, 0.2],
    "syntax_temperature": [0.5, 1.0, 2.0],
}

PHASE1_SEARCH_SPACE_OVERRIDES = {
    "mosi": {
        "learning_rate": [1e-5, 2e-5],
        "p_beta": [4, 8],
        "p_gamma": [16, 32],
        "B0_dim": [64, 128],
        "B1_dim": [64, 128],
        "max_recursion_depth": [4, 5],
        "halting_threshold": [0.02, 0.0285, 0.04],
        "dropout_prob": [0.3, 0.5],
        "silver_span_loss_weight": [0.1, 0.2],
        "syntax_temperature": [0.5, 1.0],
    }
}

PRIOR_ANCHOR_CONFIGS = {
    "mosi": {
        "learning_rate": 2e-5,
        "p_beta": 4,
        "p_gamma": 16,
        "B0_dim": 64,
        "B1_dim": 128,
        "max_recursion_depth": 5,
        "halting_threshold": 0.04,
        "dropout_prob": 0.5,
        "silver_span_loss_weight": 0.2,
        "syntax_temperature": 1.0,
    }
}

DEFAULT_CONFIG = {
    "learning_rate": 1e-5,
    "p_beta": 8,
    "p_gamma": 32,
    "B0_dim": 128,
    "B1_dim": 64,
    "max_recursion_depth": 3,
    "halting_threshold": 0.0285,
    "dropout_prob": 0.5,
    "silver_span_loss_weight": 0.1,
    "syntax_temperature": 1.0,
}

PHASE1_NAMES = {"anchor", "random", "seed_random"}

METRIC_DIRECTIONS = {
    "acc7": "maximize",
    "acc2_zero": "maximize",
    "f1_score_zero": "maximize",
    "acc2_no_zero": "maximize",
    "f1_score_no_zero": "maximize",
    "mae": "minimize",
    "corr": "maximize",
    "test_acc": "maximize",
}

METRIC_ALIASES = {
    "acc2_no_zero": ["acc2_no_zero", "test_acc"],
    "f1_score_no_zero": ["f1_score_no_zero", "f1_score"],
}

PYTHON = "/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python"


def default_n_epochs(dataset):
    return 10 if dataset == "mosei" else 20


def default_early_stopping_patience(dataset):
    return 3 if dataset == "mosei" else 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mosi", "mosei"])
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--random_trials", default=50, type=int)
    parser.add_argument("--local_trials", default=60, type=int)
    parser.add_argument("--n_epochs", default=None, type=int)
    parser.add_argument("--seed", default=128, type=int)
    parser.add_argument("--output_dir", default="optuna_results_local_refine", type=str)
    parser.add_argument(
        "--primary_metric",
        default="mae",
        choices=sorted(METRIC_DIRECTIONS.keys()),
    )
    parser.add_argument(
        "--selection_metric",
        default="mae",
        choices=["valid_loss", "mae"],
    )
    parser.add_argument("--study_prefix", default="ithp_local_refine", type=str)
    parser.add_argument("--tpe_startup_trials", default=5, type=int)
    parser.add_argument("--early_stopping_patience", default=None, type=int)
    parser.add_argument("--phase1_summary", default="", type=str)
    parser.add_argument("--force_random_phase1", action="store_true")
    parser.add_argument("--local_radius", default=1, type=int)
    args = parser.parse_args()

    if args.n_epochs is None:
        args.n_epochs = default_n_epochs(args.dataset)
    if args.early_stopping_patience is None:
        args.early_stopping_patience = default_early_stopping_patience(args.dataset)
    if args.local_radius < 0:
        raise ValueError("local_radius must be non-negative")
    return args


def parse_test_line(log_path):
    test_line = None
    with open(log_path, "r") as handle:
        for line in handle:
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


def display_path(path, work_dir):
    try:
        return str(Path(path).resolve().relative_to(work_dir))
    except ValueError:
        return str(Path(path).resolve())


def is_numeric_choice_list(choices):
    return all(isinstance(choice, (int, float)) for choice in choices)


def coerce_choice(name, value):
    choices = SEARCH_SPACE[name]
    if value is None:
        return DEFAULT_CONFIG[name]

    for choice in choices:
        if choice == value:
            return choice

    if is_numeric_choice_list(choices):
        return min(choices, key=lambda choice: abs(float(choice) - float(value)))

    return DEFAULT_CONFIG[name]


def normalize_config(config):
    config = config or {}
    normalized = {}
    for name in SEARCH_SPACE:
        normalized[name] = coerce_choice(name, config.get(name, DEFAULT_CONFIG[name]))
    return normalized


def config_signature(config):
    return json.dumps(normalize_config(config), sort_keys=True)


def get_phase1_search_space(dataset):
    override = PHASE1_SEARCH_SPACE_OVERRIDES.get(dataset)
    if override is None:
        return SEARCH_SPACE

    return {
        name: list(override.get(name, SEARCH_SPACE[name]))
        for name in SEARCH_SPACE
    }


def get_prior_anchor_config(dataset):
    config = PRIOR_ANCHOR_CONFIGS.get(dataset)
    if config is None:
        return None
    return normalize_config(config)


def build_train_command(dataset, config, n_epochs, selection_metric, early_stopping_patience):
    command = [
        PYTHON,
        "-u",
        "train.py",
        "--dataset",
        dataset,
        "--n_epochs",
        str(n_epochs),
        "--silver_span_cache",
        f"datasets/{dataset}_silver_spans.pkl",
        "--merge_trace_samples",
        "0",
        "--selection_metric",
        selection_metric,
        "--early_stopping_patience",
        str(early_stopping_patience),
    ]
    for key, value in config.items():
        command.extend([f"--{key}", str(value)])
    return command


def suggest_config(trial, search_space):
    return {
        name: trial.suggest_categorical(name, choices)
        for name, choices in search_space.items()
    }


def trial_sort_key(trial, primary_metric):
    value = metric_value(trial.user_attrs.get("metrics"), primary_metric)
    if value is None:
        return float("inf") if METRIC_DIRECTIONS[primary_metric] == "minimize" else float("-inf")
    return value


def pick_best_trial(trials, primary_metric):
    valid = [trial for trial in trials if metric_value(trial.user_attrs.get("metrics"), primary_metric) is not None]
    if not valid:
        return None

    reverse = METRIC_DIRECTIONS[primary_metric] == "maximize"
    return sorted(valid, key=lambda trial: trial_sort_key(trial, primary_metric), reverse=reverse)[0]


def count_completed_trials(study, phase_name):
    return sum(
        1
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs.get("phase") == phase_name
    )


def summarize_trials(study, primary_metric, limit=10):
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    reverse = METRIC_DIRECTIONS[primary_metric] == "maximize"
    completed.sort(
        key=lambda trial: trial_sort_key(trial, primary_metric),
        reverse=reverse,
    )

    leaderboard = []
    for trial in completed[:limit]:
        leaderboard.append(
            {
                "trial_number": trial.number,
                "phase": trial.user_attrs.get("phase"),
                "value": trial.value,
                "metrics": trial.user_attrs.get("metrics"),
                "config": trial.user_attrs.get("config"),
                "log_path": trial.user_attrs.get("log_path"),
                "duplicate_of": trial.user_attrs.get("duplicate_of"),
            }
        )
    return leaderboard


def build_phase_counts(study):
    phase_counts = {}
    for trial in study.trials:
        phase_name = trial.user_attrs.get("phase", "unknown")
        phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
    return phase_counts


def serialize_trial(trial):
    if trial is None:
        return None

    return {
        "trial_number": trial.number,
        "value": trial.value,
        "phase": trial.user_attrs.get("phase"),
        "metrics": trial.user_attrs.get("metrics"),
        "config": trial.user_attrs.get("config"),
        "log_path": trial.user_attrs.get("log_path"),
        "duplicate_of": trial.user_attrs.get("duplicate_of"),
    }


def write_artifacts(
    study,
    output_dir,
    primary_metric,
    phase1_source=None,
    phase1_search_space=None,
    phase1_anchor_config=None,
    local_search_space=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    global_best = pick_best_trial(completed, primary_metric)
    phase1_trials = [trial for trial in completed if trial.user_attrs.get("phase") in PHASE1_NAMES]
    local_trials = [trial for trial in completed if trial.user_attrs.get("phase") == "local_tpe"]

    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name.lower(),
        "primary_metric": primary_metric,
        "total_trials": len(study.trials),
        "complete_trials": len(completed),
        "phase_counts": build_phase_counts(study),
        "phase1_summary_source": phase1_source,
        "phase1_search_space": phase1_search_space,
        "phase1_anchor_config": phase1_anchor_config,
        "phase1_best_trial": serialize_trial(pick_best_trial(phase1_trials, primary_metric)),
        "local_best_trial": serialize_trial(pick_best_trial(local_trials, primary_metric)),
        "best_trial": serialize_trial(global_best),
        "local_search_space": local_search_space,
        "leaderboard": summarize_trials(study, primary_metric),
    }

    with open(output_dir / "study_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


def resolve_phase1_summary(work_dir, args):
    if args.force_random_phase1:
        return None

    if args.phase1_summary:
        candidate = Path(args.phase1_summary)
        if not candidate.is_absolute():
            candidate = work_dir / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"phase1 summary not found: {candidate}")
        return candidate

    candidates = [
        work_dir / "search_results_acc2_no_zero_50trial" / args.dataset / "summary.jsonl",
        work_dir / "search_results" / args.dataset / "summary.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def infer_log_path(summary_path, trial_idx, work_dir):
    if trial_idx is None:
        return None
    candidate = summary_path.parent / f"trial_{int(trial_idx):03d}.log"
    if candidate.exists():
        return display_path(candidate, work_dir)
    return None


def load_phase1_records(summary_path, primary_metric, work_dir):
    records = []
    with open(summary_path, "r") as handle:
        for line in handle:
            record = json.loads(line)
            metrics = record.get("metrics")
            if record.get("returncode") != 0 or metrics is None:
                continue

            primary_value = metric_value(metrics, primary_metric)
            if primary_value is None:
                continue

            config = normalize_config(record.get("config"))
            log_path = record.get("log_path") or infer_log_path(summary_path, record.get("trial"), work_dir)
            records.append(
                {
                    "trial": record.get("trial"),
                    "config": config,
                    "metrics": metrics,
                    "elapsed_seconds": record.get("elapsed_seconds"),
                    "returncode": record.get("returncode"),
                    "value": primary_value,
                    "log_path": log_path,
                }
            )
    return records


def build_distributions(search_space):
    return {
        name: optuna.distributions.CategoricalDistribution(choices)
        for name, choices in search_space.items()
    }


def import_phase1_summary(study, summary_path, primary_metric, work_dir):
    records = load_phase1_records(summary_path, primary_metric, work_dir)
    if not records:
        raise RuntimeError(f"No valid completed trials found in {summary_path}")

    existing_signatures = {
        config_signature(trial.user_attrs.get("config"))
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs.get("config") is not None
    }

    added = 0
    distributions = build_distributions(SEARCH_SPACE)
    for record in records:
        signature = config_signature(record["config"])
        if signature in existing_signatures:
            continue

        imported_trial = optuna.trial.create_trial(
            params=record["config"],
            distributions=distributions,
            value=record["value"],
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs={
                "phase": "seed_random",
                "config": record["config"],
                "metrics": record["metrics"],
                "elapsed_seconds": record.get("elapsed_seconds"),
                "returncode": record.get("returncode"),
                "log_path": record.get("log_path"),
                "seed_source_summary": display_path(summary_path, work_dir),
                "seed_source_trial": record.get("trial"),
            },
        )
        study.add_trial(imported_trial)
        existing_signatures.add(signature)
        added += 1

    return added, len(records)


def nearest_choice_index(name, value):
    choices = SEARCH_SPACE[name]
    coerced = coerce_choice(name, value)
    for index, choice in enumerate(choices):
        if choice == coerced:
            return index
    return 0


def build_local_search_space(center_config, radius):
    center_config = normalize_config(center_config)
    local_space = {}

    for name, choices in SEARCH_SPACE.items():
        center_index = nearest_choice_index(name, center_config[name])
        start = max(0, center_index - radius)
        end = min(len(choices), center_index + radius + 1)
        local_space[name] = list(choices[start:end])

    return local_space


def find_duplicate_completed_trial(study, config):
    signature = config_signature(config)
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        existing_config = trial.user_attrs.get("config")
        if existing_config is None:
            continue
        if config_signature(existing_config) == signature:
            return trial
    return None


def run_trial(trial, study, args, phase_name, search_space, work_dir, output_dir):
    config = normalize_config(suggest_config(trial, search_space))
    duplicate_trial = find_duplicate_completed_trial(study, config)

    trial.set_user_attr("phase", phase_name)
    trial.set_user_attr("config", config)

    if duplicate_trial is not None:
        metrics = duplicate_trial.user_attrs.get("metrics")
        trial.set_user_attr("duplicate_of", duplicate_trial.number)
        trial.set_user_attr("elapsed_seconds", 0.0)
        trial.set_user_attr("returncode", duplicate_trial.user_attrs.get("returncode", 0))
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("log_path", duplicate_trial.user_attrs.get("log_path"))
        print(
            f"[{phase_name}][Trial {trial.number}] DUPLICATE of trial {duplicate_trial.number} "
            f"— {args.primary_metric}={format_metric(metrics, args.primary_metric)}"
        )
        return duplicate_trial.value

    phase_dir = output_dir / "trial_logs" / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    log_path = phase_dir / f"trial_{trial.number:04d}.log"
    trial.set_user_attr("log_path", display_path(log_path, work_dir))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    command = build_train_command(
        args.dataset,
        config,
        args.n_epochs,
        args.selection_metric,
        args.early_stopping_patience,
    )

    print(f"[{phase_name}][Trial {trial.number}] Config: {config}")
    print(f"[{phase_name}][Trial {trial.number}] Log: {display_path(log_path, work_dir)}")

    t0 = time.time()
    with open(log_path, "w") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(work_dir),
        )
    elapsed = time.time() - t0

    metrics = parse_test_line(log_path)
    trial.set_user_attr("elapsed_seconds", round(elapsed, 1))
    trial.set_user_attr("returncode", process.returncode)
    trial.set_user_attr("metrics", metrics)

    if process.returncode != 0 or metrics is None:
        raise RuntimeError(
            f"Trial {trial.number} failed for dataset={args.dataset}, phase={phase_name}, returncode={process.returncode}"
        )

    primary_value = metric_value(metrics, args.primary_metric)
    if primary_value is None:
        raise RuntimeError(
            f"Trial {trial.number} missing primary metric '{args.primary_metric}'"
        )

    print(
        f"[{phase_name}][Trial {trial.number}] DONE in {elapsed:.0f}s — "
        f"{args.primary_metric}={format_metric(metrics, args.primary_metric)}, "
        f"f1_no_zero={format_metric(metrics, 'f1_score_no_zero')}, "
        f"mae={format_metric(metrics, 'mae')}, "
        f"best_epoch={metrics.get('best_epoch', 'NA')}"
    )

    return primary_value


def run_phase(
    args,
    study_name,
    storage_uri,
    work_dir,
    output_dir,
    phase_name,
    sampler,
    n_trials,
    search_space,
    phase1_source=None,
    phase1_search_space=None,
    phase1_anchor_config=None,
    local_search_space=None,
):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        load_if_exists=True,
        direction=METRIC_DIRECTIONS[args.primary_metric],
        sampler=sampler,
    )

    completed = count_completed_trials(study, phase_name)
    remaining = max(0, n_trials - completed)

    print(f"=== Phase {phase_name}: completed={completed}, remaining={remaining} ===")
    if remaining == 0:
        return study

    objective = lambda trial: run_trial(trial, study, args, phase_name, search_space, work_dir, output_dir)
    study.optimize(objective, n_trials=remaining, catch=(RuntimeError,))
    write_artifacts(
        study,
        output_dir,
        args.primary_metric,
        phase1_source=phase1_source,
        phase1_search_space=phase1_search_space,
        phase1_anchor_config=phase1_anchor_config,
        local_search_space=local_search_space,
    )
    return study


def main():
    args = parse_args()
    work_dir = Path(__file__).resolve().parent.parent
    os.chdir(work_dir)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = (output_dir / "optuna_study.sqlite3").resolve()
    storage_uri = f"sqlite:///{storage_path}"
    study_name = f"{args.study_prefix}_{args.dataset}_{args.primary_metric}"

    print(f"=== Local Refine Optuna: dataset={args.dataset}, gpu={args.gpu} ===")
    print(f"Output: {output_dir}")
    print(f"Storage: {storage_uri}")
    print(f"Study: {study_name}")
    print(f"Primary metric: {args.primary_metric} ({METRIC_DIRECTIONS[args.primary_metric]})")
    print(f"Selection metric: {args.selection_metric} (minimize)")
    print(f"Epochs per trial: {args.n_epochs}")
    print(f"Early stopping patience: {args.early_stopping_patience}")

    phase1_summary = resolve_phase1_summary(work_dir, args)
    phase1_source = display_path(phase1_summary, work_dir) if phase1_summary is not None else None
    phase1_search_space = get_phase1_search_space(args.dataset)
    phase1_anchor_config = get_prior_anchor_config(args.dataset)

    base_sampler = optuna.samplers.RandomSampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        load_if_exists=True,
        direction=METRIC_DIRECTIONS[args.primary_metric],
        sampler=base_sampler,
    )

    if phase1_summary is not None:
        added, total_records = import_phase1_summary(study, phase1_summary, args.primary_metric, work_dir)
        print(
            f"Phase 1 source: imported {added}/{total_records} completed trials from {phase1_source}"
        )
    else:
        print("Phase 1 source: fresh random Optuna phase")
        print(f"Phase 1 search space: {phase1_search_space}")
        if phase1_anchor_config is not None:
            print(f"Phase 1 anchor config: {phase1_anchor_config}")
            study.enqueue_trial(phase1_anchor_config)
            study = run_phase(
                args,
                study_name,
                storage_uri,
                work_dir,
                output_dir,
                phase_name="anchor",
                sampler=base_sampler,
                n_trials=1,
                search_space=phase1_search_space,
                phase1_source=phase1_source,
                phase1_search_space=phase1_search_space,
                phase1_anchor_config=phase1_anchor_config,
            )
        study = run_phase(
            args,
            study_name,
            storage_uri,
            work_dir,
            output_dir,
            phase_name="random",
            sampler=base_sampler,
            n_trials=args.random_trials,
            search_space=phase1_search_space,
            phase1_source=phase1_source,
            phase1_search_space=phase1_search_space,
            phase1_anchor_config=phase1_anchor_config,
        )

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    phase1_trials = [trial for trial in completed if trial.user_attrs.get("phase") in PHASE1_NAMES]
    phase1_best = pick_best_trial(phase1_trials, args.primary_metric)
    if phase1_best is None:
        raise RuntimeError("No completed phase-1 trials available for local refinement")

    center_config = normalize_config(phase1_best.user_attrs.get("config"))
    local_search_space = build_local_search_space(center_config, args.local_radius)

    print("=== Phase 1 Best ===")
    print(f"trial={phase1_best.number}, phase={phase1_best.user_attrs.get('phase')}")
    print(f"{args.primary_metric}={format_metric(phase1_best.user_attrs.get('metrics'), args.primary_metric)}")
    print(f"config={center_config}")
    print(f"local_search_space={local_search_space}")

    tpe_sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=args.tpe_startup_trials,
    )
    study = run_phase(
        args,
        study_name,
        storage_uri,
        work_dir,
        output_dir,
        phase_name="local_tpe",
        sampler=tpe_sampler,
        n_trials=args.local_trials,
        search_space=local_search_space,
        phase1_source=phase1_source,
        local_search_space=local_search_space,
    )

    write_artifacts(
        study,
        output_dir,
        args.primary_metric,
        phase1_source=phase1_source,
        phase1_search_space=phase1_search_space,
        phase1_anchor_config=phase1_anchor_config,
        local_search_space=local_search_space,
    )

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    best_trial = pick_best_trial(completed, args.primary_metric)
    if best_trial is not None:
        best_metrics = best_trial.user_attrs.get("metrics") or {}
        print("=== BEST TRIAL ===")
        print(f"trial={best_trial.number}, phase={best_trial.user_attrs.get('phase')}")
        print(f"{args.primary_metric}={format_metric(best_metrics, args.primary_metric)}")
        print(
            f"f1_no_zero={format_metric(best_metrics, 'f1_score_no_zero')}, "
            f"mae={format_metric(best_metrics, 'mae')}, "
            f"corr={format_metric(best_metrics, 'corr')}"
        )
        print(f"config={best_trial.user_attrs.get('config')}")


if __name__ == "__main__":
    main()