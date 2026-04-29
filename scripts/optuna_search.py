#!/usr/bin/env python
"""Two-phase Optuna search for ITHP recursive model.

Phase 1 uses RandomSampler for broad exploration.
Phase 2 reuses the same Optuna study with TPESampler for stronger search.

CH-SIMSv2 (``--dataset simsv2``): defaults tuned for BERT-Chinese regression —
``n_epochs=25``, ``early_stopping_patience=4``, ``random_trials=20``,
``tpe_trials=60`` (80 trials total), ``SEARCH_SPACE_SIMSV2`` (no silver spans;
``silver_span_loss_weight`` fixed to 0; extra ``train.py`` knobs). Use
``--base_model /path/to/bert-base-chinese`` to pin weights. MOSI/MOSEI keep the
original grid and 50+100 trials unless overridden. Default ``--output_dir`` for
``simsv2`` is ``log/4080_restart`` (study lives under ``.../log/4080_restart/simsv2/``).
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import optuna


GRACEFUL_STOP_REQUESTED = False
GRACEFUL_STOP_SIGNAL = None


# MOSI / MOSEI (silver span cache + regression / acc7 auxiliary).
SEARCH_SPACE_DEFAULT = {
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

# CH-SIMSv2: no silver cache; keep syntax_temperature; fix silver span loss to 0;
# add schedule / IB-adjacent knobs aligned with ``train.py`` CLI.
SEARCH_SPACE_SIMSV2 = {
    "learning_rate": [5e-6, 1e-5, 2e-5, 3e-5],
    "p_beta": [4, 8, 16],
    "p_gamma": [16, 32, 64],
    "p_lambda": [0.2, 0.3, 0.5],
    "beta_shift": [0.8, 1.0, 1.2],
    "B0_dim": [64, 128, 256],
    "B1_dim": [32, 64, 128],
    "inter_dim": [128, 256],
    "max_recursion_depth": [3, 4, 5],
    "halting_threshold": [0.02, 0.0285, 0.04, 0.06],
    "dropout_prob": [0.3, 0.5],
    "drop_prob": [0.2, 0.3],
    "warmup_proportion": [0.05, 0.1],
    "syntax_temperature": [0.5, 1.0, 2.0],
    "silver_span_loss_weight": [0.0],
    "train_batch_size": [4, 8, 16],
    "gradient_accumulation_step": [1, 2],
    "max_grad_norm": [0.5, 1.0],
}

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
    "acc2_no_zero": ["acc2_no_zero", "test_acc", "Mult_acc_2"],
    "f1_score_no_zero": ["f1_score_no_zero", "f1_score", "F1_score"],
}

PYTHON = "/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python"


def _signal_name(signum):
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


def request_graceful_stop(signum, _frame):
    global GRACEFUL_STOP_REQUESTED, GRACEFUL_STOP_SIGNAL
    GRACEFUL_STOP_REQUESTED = True
    GRACEFUL_STOP_SIGNAL = _signal_name(signum)
    print(
        f"GRACEFUL_STOP: received {GRACEFUL_STOP_SIGNAL}; the current trial will finish before the search stops.",
        flush=True,
    )


def install_signal_handlers():
    for signum in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if signum is not None:
            signal.signal(signum, request_graceful_stop)


def maybe_stop_study_after_trial(study, trial):
    if not GRACEFUL_STOP_REQUESTED:
        return

    print(
        f"GRACEFUL_STOP: trial {trial.number} finished; stopping before the next trial.",
        flush=True,
    )
    study.stop()


def default_n_epochs(dataset):
    if dataset == "mosei":
        return 10
    if dataset == "simsv2":
        return 25
    return 20


def default_early_stopping_patience(dataset):
    if dataset == "mosei":
        return 3
    if dataset == "simsv2":
        return 4
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mosi", "mosei", "simsv2"])
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument(
        "--random_trials",
        default=None,
        type=int,
        help="Random phase trials (default: 50 mosi/mosei, 20 simsv2).",
    )
    parser.add_argument(
        "--tpe_trials",
        default=None,
        type=int,
        help="TPE phase trials (default: 100 mosi/mosei, 60 simsv2).",
    )
    parser.add_argument("--n_epochs", default=None, type=int)
    parser.add_argument("--seed", default=128, type=int)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Trial logs + sqlite root. Default: log/4080_restart for simsv2, optuna_results otherwise.",
    )
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
    parser.add_argument("--study_prefix", default="ithp", type=str)
    parser.add_argument("--tpe_startup_trials", default=10, type=int)
    parser.add_argument("--early_stopping_patience", default=None, type=int)
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="If set, forwarded as ``--model`` to train.py (e.g. local bert-base-chinese path).",
    )
    args = parser.parse_args()
    if args.random_trials is None:
        args.random_trials = 20 if args.dataset == "simsv2" else 50
    if args.tpe_trials is None:
        args.tpe_trials = 60 if args.dataset == "simsv2" else 100
    if args.n_epochs is None:
        args.n_epochs = default_n_epochs(args.dataset)
    if args.early_stopping_patience is None:
        args.early_stopping_patience = default_early_stopping_patience(args.dataset)
    if args.output_dir is None:
        args.output_dir = (
            "log/4080_restart" if args.dataset == "simsv2" else "optuna_results"
        )
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


def build_train_command(
    dataset, config, n_epochs, selection_metric, early_stopping_patience, base_model=None
):
    if dataset == "simsv2":
        silver_arg = ["--silver_span_cache", ""]
    else:
        silver_arg = ["--silver_span_cache", f"datasets/{dataset}_silver_spans.pkl"]
    command = [
        PYTHON,
        "-u",
        "train.py",
        "--dataset",
        dataset,
        "--n_epochs",
        str(n_epochs),
        *silver_arg,
        "--merge_trace_samples",
        "0",
        "--selection_metric",
        selection_metric,
        "--early_stopping_patience",
        str(early_stopping_patience),
    ]
    for key, value in config.items():
        command.extend([f"--{key}", str(value)])
    if base_model:
        command.extend(["--model", str(base_model)])
    return command


def suggest_config(trial, dataset):
    space = SEARCH_SPACE_SIMSV2 if dataset == "simsv2" else SEARCH_SPACE_DEFAULT
    return {
        name: trial.suggest_categorical(name, choices)
        for name, choices in space.items()
    }


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
        key=lambda trial: metric_value(trial.user_attrs.get("metrics"), primary_metric)
        if metric_value(trial.user_attrs.get("metrics"), primary_metric) is not None
        else (float("-inf") if reverse else float("inf")),
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
            }
        )
    return leaderboard


def write_artifacts(study, output_dir, primary_metric):
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    best_trial = study.best_trial if completed else None

    summary = {
        "study_name": study.study_name,
        "direction": study.direction.name.lower(),
        "primary_metric": primary_metric,
        "total_trials": len(study.trials),
        "complete_trials": len(completed),
        "best_trial": None,
        "leaderboard": summarize_trials(study, primary_metric),
    }

    if best_trial is not None:
        summary["best_trial"] = {
            "trial_number": best_trial.number,
            "value": best_trial.value,
            "phase": best_trial.user_attrs.get("phase"),
            "metrics": best_trial.user_attrs.get("metrics"),
            "config": best_trial.user_attrs.get("config"),
            "log_path": best_trial.user_attrs.get("log_path"),
        }

    with open(output_dir / "study_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


def run_trial(trial, args, phase_name, work_dir, output_dir):
    config = suggest_config(trial, args.dataset)
    phase_dir = output_dir / "trial_logs" / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    log_path = phase_dir / f"trial_{trial.number:04d}.log"

    trial.set_user_attr("phase", phase_name)
    trial.set_user_attr("config", config)
    trial.set_user_attr("log_path", str(log_path))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    command = build_train_command(
        args.dataset,
        config,
        args.n_epochs,
        args.selection_metric,
        args.early_stopping_patience,
        base_model=args.base_model,
    )

    print(f"[{phase_name}][Trial {trial.number}] Config: {config}")
    print(f"[{phase_name}][Trial {trial.number}] Log: {log_path}")

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
        f"f1_or_F1={format_metric(metrics, 'f1_score_no_zero')}, "
        f"mae={metrics.get('mae', 'NA')}, best_epoch={metrics.get('best_epoch', 'NA')}"
    )

    return primary_value


def run_phase(args, study_name, storage_uri, work_dir, output_dir, phase_name, sampler, n_trials):
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
    if GRACEFUL_STOP_REQUESTED:
        print(
            f"=== Phase {phase_name}: graceful stop already requested; skipping {remaining} pending trials ===",
            flush=True,
        )
        return study

    objective = lambda trial: run_trial(trial, args, phase_name, work_dir, output_dir)
    study.optimize(
        objective,
        n_trials=remaining,
        catch=(RuntimeError,),
        callbacks=[maybe_stop_study_after_trial],
    )
    write_artifacts(study, output_dir, args.primary_metric)
    return study


def main():
    install_signal_handlers()
    args = parse_args()
    work_dir = Path(__file__).resolve().parent.parent
    os.chdir(work_dir)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = (output_dir / "optuna_study.sqlite3").resolve()
    storage_uri = f"sqlite:///{storage_path}"
    study_name = f"{args.study_prefix}_{args.dataset}_{args.primary_metric}"

    print(f"=== Optuna Search: dataset={args.dataset}, gpu={args.gpu} ===")
    print(f"Output: {output_dir}")
    print(f"Storage: {storage_uri}")
    print(f"Study: {study_name}")
    print(f"Primary metric: {args.primary_metric} ({METRIC_DIRECTIONS[args.primary_metric]})")
    print(f"Selection metric: {args.selection_metric} (minimize)")
    print(f"Epochs per trial: {args.n_epochs}")
    print(f"Early stopping patience: {args.early_stopping_patience}")

    random_sampler = optuna.samplers.RandomSampler(seed=args.seed)
    study = run_phase(
        args,
        study_name,
        storage_uri,
        work_dir,
        output_dir,
        phase_name="random",
        sampler=random_sampler,
        n_trials=args.random_trials,
    )

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
        phase_name="tpe",
        sampler=tpe_sampler,
        n_trials=args.tpe_trials,
    )

    write_artifacts(study, output_dir, args.primary_metric)
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        best_trial = study.best_trial
        best_metrics = study.best_trial.user_attrs.get("metrics") or {}
        print("=== BEST TRIAL ===")
        print(f"trial={best_trial.number}, phase={best_trial.user_attrs.get('phase')}")
        print(f"{args.primary_metric}={format_metric(best_metrics, args.primary_metric)}")
        print(f"f1_no_zero={format_metric(best_metrics, 'f1_score_no_zero')}, mae={format_metric(best_metrics, 'mae')}, corr={format_metric(best_metrics, 'corr')}")
        print(f"config={best_trial.user_attrs.get('config')}")


if __name__ == "__main__":
    main()