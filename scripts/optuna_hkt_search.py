#!/usr/bin/env python3
"""Two-phase Optuna search for the HKT binary trainer.

Phase 1 uses ``RandomSampler``; phase 2 reuses the same study with
``TPESampler``. Each trial shells out to ``train_hkt_binary.py`` (which writes
``result.json`` per run); we parse that file to pull the primary metric.
Optional **dev-tuned decision threshold** (``--decision-threshold-mode tune_on_valid``
and ``--primary_metric valid_*_threshold_tuned``) forwards extra flags to the trainer;
see ``train_hkt_binary.py`` docstring for ``threshold_tuning`` in ``result.json``.
The ``syntax_loss_weight`` hyperparameter is searched when ``datasets/<dataset>_silver_spans.pkl``
exists; otherwise Optuna restricts it to ``[0.0]`` (see README silver section).
Use ``--syntax-loss-weight-sampling uniform`` for ``suggest_float`` in
``[0, --syntax-loss-weight-high]`` (requires a new ``--study_prefix``).

Default primary metric is ``valid_accuracy`` (model-selection protocol the
trainer uses by default). For MUStARD the search runs on the HKT single-fold
pickle (539/68/68); set ``--fold k`` to pin to a specific speaker-independent
fold instead. UR-FUNNY always uses its official fold.

**Trial length defaults (``default_n_epochs`` / ``default_early_stopping``):**
UR-FUNNY used to run only 5 epochs (underfitting); it now defaults to **10**
epochs with patience **2**. MUStARD used 15 epochs with **no** early stopping
(patience 0), which tends to overfit valid; it now defaults to patience **3**
with the same 15-epoch cap. Override with ``--n_epochs`` /
``--early_stopping_patience`` when needed.

Typical launch (total budget ~80 trials: random exploration + TPE):

    python scripts/optuna_hkt_search.py --dataset mustard --gpu 0 \
        --random_trials 20 --tpe_trials 60 \
        --output_dir log/4080_restart_hkt

Fleet launcher (default all on GPU 0, outputs under ``log/4080_restart/{mustard,urfunny,simsv2}/``):
``bash scripts/start_silver_span_optuna.sh``
starts mustard, urfunny, and simsv2 Optuna drivers in parallel. To stop one
driver, send SIGINT to its PID (current trial finishes, then exit).

Reusing or restarting studies: the same ``--output_dir``, ``--study_prefix``,
and ``--primary_metric`` load ``optuna_study.sqlite3`` with
``load_if_exists=True``. To append trials, pass ``--random_trials`` /
``--tpe_trials`` as the *remaining* counts for each phase. For a clean sweep,
use a new ``--output_dir`` or ``--study_prefix``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna


GRACEFUL_STOP_REQUESTED = False
GRACEFUL_STOP_SIGNAL = None


SEARCH_SPACE = {
    "learning_rate": [5e-6, 1e-5, 2e-5, 3e-5],
    "p_beta": [4, 8, 16],
    "p_gamma": [16, 32, 64],
    "p_lambda": [0.2, 0.3, 0.5],
    "beta_shift": [0.8, 1.0, 1.2],
    "B0_dim": [64, 128, 256],
    "B1_dim": [32, 64, 128],
    "inter_dim": [128, 256, 384],
    "max_recursion_depth": [3, 4, 5],
    "halting_threshold": [0.02, 0.0285, 0.04, 0.06],
    "dropout_prob": [0.3, 0.5],
    "drop_prob": [0.2, 0.3, 0.4],
    "warmup_proportion": [0.05, 0.1, 0.15],
    "train_batch_size": [8, 16, 32],
    "gradient_accumulation_step": [1, 2],
    "max_grad_norm": [0.5, 1.0],
    # Same role as MOSI ``silver_span_loss_weight``; forwarded to train_hkt_binary --syntax_loss_weight.
    "syntax_loss_weight": [0.0, 0.05, 0.1],
}

# Mutable copy; ``main()`` may restrict ``syntax_loss_weight`` to ``[0.0]`` when the silver pickle is missing.
ACTIVE_SEARCH_SPACE = dict(SEARCH_SPACE)

METRIC_DIRECTIONS = {
    "valid_accuracy": "maximize",
    "valid_accuracy_threshold_tuned": "maximize",
    "valid_f1": "maximize",
    "valid_f1_threshold_tuned": "maximize",
    "valid_loss": "minimize",
    "test_accuracy": "maximize",
    "test_f1": "maximize",
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
        f"GRACEFUL_STOP: received {GRACEFUL_STOP_SIGNAL}; current trial will finish before stopping.",
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
    """UR-FUNNY: was 5 (often underfit); 10 gives more room before early stop."""
    return 10 if dataset == "urfunny" else 15


def default_early_stopping(dataset):
    """MUStARD: non-zero patience reduces valid overfitting; UR-FUNNY keeps 2."""
    return 2 if dataset == "urfunny" else 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mustard", "urfunny", "sarcasm", "humor"])
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--random_trials", default=20, type=int)
    parser.add_argument("--tpe_trials", default=60, type=int)
    parser.add_argument("--n_epochs", default=None, type=int)
    parser.add_argument("--seed", default=5149, type=int)
    parser.add_argument("--early_stopping_patience", default=None, type=int)
    parser.add_argument("--output_dir", default="log/4080_restart_hkt", type=str)
    parser.add_argument(
        "--primary_metric",
        default="valid_accuracy",
        choices=sorted(METRIC_DIRECTIONS.keys()),
    )
    parser.add_argument(
        "--selection_metric",
        default="accuracy",
        choices=["accuracy", "f1", "valid_loss"],
        help="How train_hkt_binary.py picks its best epoch per trial.",
    )
    parser.add_argument(
        "--fold",
        default=None,
        type=int,
        help="MUStARD: pin to a specific speaker-independent fold. Default: HKT pickle single split.",
    )
    parser.add_argument("--study_prefix", default="ithp_hkt", type=str)
    parser.add_argument("--tpe_startup_trials", default=10, type=int)
    parser.add_argument(
        "--github_style",
        action="store_true",
        help=(
            "Forward --github_style to every train_hkt_binary.py trial. "
            "Use a fresh --output_dir/--study_prefix so the new trials don't "
            "mix with existing runs that used the ITHP-default processing."
        ),
    )
    parser.add_argument(
        "--hkt_paper_style",
        action="store_true",
        help="Forward --hkt_paper_style to every trial (60+36, z-score, HCF, binary F1). Incompatible with --github_style.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="If set, forwards --model to train_hkt_binary (e.g. path to albert-base-v2).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="auto",
        choices=["auto", "albert", "deberta"],
        help="Forwarded to train_hkt_binary --backbone when not 'auto'.",
    )
    parser.add_argument(
        "--syntax-loss-weight-sampling",
        choices=["categorical", "uniform"],
        default="categorical",
        help=(
            "categorical: use SEARCH_SPACE discrete grid for syntax_loss_weight. "
            "uniform: Optuna suggest_float in [0, --syntax-loss-weight-high] (requires silver pickle; "
            "use a new --study_prefix when switching modes — incompatible with old categorical studies)."
        ),
    )
    parser.add_argument(
        "--syntax-loss-weight-high",
        default=0.15,
        type=float,
        help="Upper bound for uniform syntax_loss_weight sampling (inclusive; lower bound is 0).",
    )
    parser.add_argument(
        "--decision-threshold-mode",
        dest="decision_threshold_mode",
        choices=["fixed", "tune_on_valid"],
        default="fixed",
        help="Forwarded to train_hkt_binary --decision_threshold_mode.",
    )
    parser.add_argument(
        "--threshold-tune-objective",
        dest="threshold_tune_objective",
        choices=["accuracy", "f1"],
        default="accuracy",
        help="Forwarded to train_hkt_binary when using tune_on_valid.",
    )
    parser.add_argument(
        "--threshold-grid-size",
        dest="threshold_grid_size",
        type=int,
        default=91,
        help="Forwarded to train_hkt_binary --threshold_grid_size.",
    )
    args = parser.parse_args()

    # Normalise dataset aliases the same way train_hkt_binary.py does.
    if args.dataset == "sarcasm":
        args.dataset = "mustard"
    if args.dataset == "humor":
        args.dataset = "urfunny"

    if args.n_epochs is None:
        args.n_epochs = default_n_epochs(args.dataset)
    if args.early_stopping_patience is None:
        args.early_stopping_patience = default_early_stopping(args.dataset)
    if args.dataset == "urfunny" and args.fold is not None:
        raise SystemExit("--fold is only meaningful for dataset=mustard")
    if args.github_style and args.hkt_paper_style:
        raise SystemExit("Use either --github_style or --hkt_paper_style, not both")
    if args.syntax_loss_weight_sampling == "uniform" and args.syntax_loss_weight_high <= 0:
        raise SystemExit("--syntax-loss-weight-high must be > 0 when using --syntax-loss-weight-sampling uniform")
    if args.threshold_grid_size < 3:
        raise SystemExit("--threshold-grid-size must be >= 3")
    if args.primary_metric in ("valid_accuracy_threshold_tuned", "valid_f1_threshold_tuned"):
        if args.decision_threshold_mode == "fixed":
            args.decision_threshold_mode = "tune_on_valid"
            print(
                "NOTE: primary_metric requires dev threshold tuning; "
                "setting --decision-threshold-mode tune_on_valid",
                flush=True,
            )
    return args


def parse_result_json(result_path):
    if not os.path.exists(result_path):
        return None
    with open(result_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_value(result, metric_name):
    if result is None:
        return None
    if metric_name in ("valid_accuracy_threshold_tuned", "valid_f1_threshold_tuned"):
        tt = result.get("threshold_tuning") or {}
        valid = tt.get("valid") or {}
        metric_key = "accuracy" if metric_name == "valid_accuracy_threshold_tuned" else "f1"
        value = valid.get(metric_key)
        if value is None:
            return None
        return float(value)
    best = result.get("best") or {}
    split_name, metric_key = {
        "valid_accuracy": ("valid", "accuracy"),
        "valid_f1": ("valid", "f1"),
        "valid_loss": ("valid", "loss"),
        "test_accuracy": ("test", "accuracy"),
        "test_f1": ("test", "f1"),
    }[metric_name]
    bucket = best.get(split_name) or {}
    value = bucket.get(metric_key)
    if value is None:
        return None
    return float(value)


def build_train_command(args, config, run_name, repo_root: Path):
    command = [
        PYTHON,
        "-u",
        "train_hkt_binary.py",
        "--dataset",
        args.dataset,
        "--selection_metric",
        args.selection_metric,
        "--n_epochs",
        str(args.n_epochs),
        "--early_stopping_patience",
        str(args.early_stopping_patience),
        "--seed",
        str(args.seed),
        "--output_dir",
        args.output_dir,
        "--run_name",
        run_name,
    ]
    if args.dataset == "mustard":
        command.extend(["--fold", str(args.fold if args.fold is not None else -1)])
    if getattr(args, "github_style", False):
        command.append("--github_style")
    if getattr(args, "hkt_paper_style", False):
        command.append("--hkt_paper_style")
    if getattr(args, "base_model", None):
        command.extend(["--model", str(args.base_model)])
    if getattr(args, "backbone", "auto") != "auto":
        command.extend(["--backbone", str(args.backbone)])

    silver_rel = f"datasets/{args.dataset}_silver_spans.pkl"
    silver_path = repo_root / silver_rel
    if silver_path.is_file():
        command.extend(["--silver_span_cache", silver_rel])

    if getattr(args, "decision_threshold_mode", "fixed") != "fixed":
        command.extend(["--decision_threshold_mode", args.decision_threshold_mode])
        command.extend(["--threshold_tune_objective", args.threshold_tune_objective])
        command.extend(["--threshold_grid_size", str(args.threshold_grid_size)])

    for key, value in config.items():
        command.extend([f"--{key}", str(value)])
    return command


def _merge_frozen_categorical_distributions(study: optuna.Study, space: dict) -> dict:
    """Align categorical ``choices`` with distributions already stored in the study (resume-safe)."""
    frozen: dict[str, tuple] = {}
    for t in study.trials:
        for name, dist in (t.distributions or {}).items():
            if name in frozen or not hasattr(dist, "choices"):
                continue
            frozen[name] = tuple(dist.choices)
    if not frozen:
        return space
    merged = dict(space)
    for name, choices in frozen.items():
        if name not in merged:
            continue
        cur = merged[name]
        if isinstance(cur, (list, tuple)) and tuple(cur) != choices:
            merged[name] = list(choices)
    return merged


def suggest_config(trial, args):
    space = dict(ACTIVE_SEARCH_SPACE)
    space = _merge_frozen_categorical_distributions(trial.study, space)
    use_uniform_syntax = (
        args.syntax_loss_weight_sampling == "uniform"
        and len(space.get("syntax_loss_weight", [0.0])) > 1
    )
    if use_uniform_syntax:
        space.pop("syntax_loss_weight", None)
    config = {name: trial.suggest_categorical(name, choices) for name, choices in space.items()}
    if use_uniform_syntax:
        hi = float(args.syntax_loss_weight_high)
        config["syntax_loss_weight"] = trial.suggest_float("syntax_loss_weight", 0.0, hi)
    return config


def count_completed_trials(study, phase_name):
    return sum(
        1
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs.get("phase") == phase_name
    )


def summarise_trials(study, primary_metric, limit=15):
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    reverse = METRIC_DIRECTIONS[primary_metric] == "maximize"
    completed.sort(
        key=lambda trial: (
            trial.value
            if trial.value is not None
            else (float("-inf") if reverse else float("inf"))
        ),
        reverse=reverse,
    )
    leaderboard = []
    for trial in completed[:limit]:
        leaderboard.append(
            {
                "trial_number": trial.number,
                "phase": trial.user_attrs.get("phase"),
                "value": trial.value,
                "config": trial.user_attrs.get("config"),
                "result_path": trial.user_attrs.get("result_path"),
                "metrics": (trial.user_attrs.get("result") or {}).get("best"),
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
        "leaderboard": summarise_trials(study, primary_metric),
    }
    if best_trial is not None:
        summary["best_trial"] = {
            "trial_number": best_trial.number,
            "value": best_trial.value,
            "phase": best_trial.user_attrs.get("phase"),
            "config": best_trial.user_attrs.get("config"),
            "result_path": best_trial.user_attrs.get("result_path"),
            "metrics": (best_trial.user_attrs.get("result") or {}).get("best"),
        }
    with open(output_dir / "study_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def run_trial(trial, args, phase_name, work_dir, output_dir):
    config = suggest_config(trial, args)
    run_name = f"{args.dataset}/trial_logs/{phase_name}/trial_{trial.number:04d}"
    trial_dir = output_dir / "trial_logs" / phase_name / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "parent.log"
    result_path = trial_dir / "result.json"

    trial.set_user_attr("phase", phase_name)
    trial.set_user_attr("config", config)
    trial.set_user_attr("log_path", str(log_path))
    trial.set_user_attr("result_path", str(result_path))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    command = build_train_command(args, config, run_name, work_dir)

    print(f"[{phase_name}][Trial {trial.number}] Config: {config}", flush=True)
    print(f"[{phase_name}][Trial {trial.number}] Log: {log_path}", flush=True)

    started_at = time.time()
    with open(log_path, "w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(work_dir),
        )
    elapsed = time.time() - started_at

    result = parse_result_json(str(result_path))
    trial.set_user_attr("elapsed_seconds", round(elapsed, 1))
    trial.set_user_attr("returncode", process.returncode)
    trial.set_user_attr("result", result)

    if process.returncode != 0 or result is None:
        raise RuntimeError(
            f"Trial {trial.number} failed for dataset={args.dataset}, phase={phase_name}, "
            f"returncode={process.returncode}"
        )

    primary = metric_value(result, args.primary_metric)
    if primary is None:
        raise RuntimeError(f"Trial {trial.number} missing primary metric '{args.primary_metric}'")

    best = result.get("best") or {}
    valid = best.get("valid") or {}
    test = best.get("test") or {}
    tt = result.get("threshold_tuning") or {}
    tt_test = tt.get("test") or {}
    extra = ""
    if tt_test:
        extra = (
            f", tuned_test_acc={tt_test.get('accuracy')}, tuned_test_f1={tt_test.get('f1')}"
        )
    print(
        f"[{phase_name}][Trial {trial.number}] DONE in {elapsed:.0f}s — "
        f"{args.primary_metric}={primary:.4f}, "
        f"valid_acc={valid.get('accuracy')}, test_acc={test.get('accuracy')}, test_f1={test.get('f1')}"
        f"{extra}",
        flush=True,
    )
    return primary


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

    print(f"=== Phase {phase_name}: completed={completed}, remaining={remaining} ===", flush=True)
    if remaining == 0:
        return study
    if GRACEFUL_STOP_REQUESTED:
        print(
            f"=== Phase {phase_name}: graceful stop requested; skipping {remaining} pending trials ===",
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

    global ACTIVE_SEARCH_SPACE
    ACTIVE_SEARCH_SPACE = dict(SEARCH_SPACE)
    silver_path = work_dir / "datasets" / f"{args.dataset}_silver_spans.pkl"
    if not silver_path.is_file():
        ACTIVE_SEARCH_SPACE["syntax_loss_weight"] = [0.0]
        print(
            "NOTICE: HKT Optuna — "
            f"datasets/{args.dataset}_silver_spans.pkl not found; syntax_loss_weight is restricted to [0.0]. "
            "Build cache with:\n"
            f"  python scripts/build_hkt_silver_span_cache.py --dataset {args.dataset}",
            flush=True,
        )

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = (output_dir / "optuna_study.sqlite3").resolve()
    storage_uri = f"sqlite:///{storage_path}"
    study_name = f"{args.study_prefix}_{args.dataset}_{args.primary_metric}"
    if args.fold is not None:
        study_name += f"_fold{args.fold}"

    print(f"=== HKT Optuna Search: dataset={args.dataset}, gpu={args.gpu} ===")
    print(f"Output: {output_dir}")
    print(f"Storage: {storage_uri}")
    print(f"Study: {study_name}")
    print(f"Primary metric: {args.primary_metric} ({METRIC_DIRECTIONS[args.primary_metric]})")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Epochs per trial: {args.n_epochs}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    if args.dataset == "mustard":
        print(
            "NOTE: MUStARD default early_stopping_patience>0 to curb valid overfit "
            "(pass --early_stopping_patience 0 to reproduce old behaviour).",
            flush=True,
        )
    elif args.dataset == "urfunny":
        print(
            "NOTE: UR-FUNNY default n_epochs=10 (was 5) to reduce underfit "
            "(pass --n_epochs 5 to reproduce old behaviour).",
            flush=True,
        )
    print(
        f"syntax_loss_weight sampling: {args.syntax_loss_weight_sampling}"
        + (
            f" (uniform [0, {args.syntax_loss_weight_high}])"
            if args.syntax_loss_weight_sampling == "uniform"
            else ""
        ),
        flush=True,
    )

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
        best_result = best_trial.user_attrs.get("result") or {}
        best_metrics = best_result.get("best") or {}
        print("=== BEST TRIAL ===")
        print(f"trial={best_trial.number}, phase={best_trial.user_attrs.get('phase')}")
        print(f"{args.primary_metric}={best_trial.value}")
        print(f"valid={best_metrics.get('valid')}")
        print(f"test={best_metrics.get('test')}")
        print(f"config={best_trial.user_attrs.get('config')}")


if __name__ == "__main__":
    main()
