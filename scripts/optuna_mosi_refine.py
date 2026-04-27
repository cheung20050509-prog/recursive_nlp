#!/usr/bin/env python
"""MOSI-only Optuna refine with an extended categorical space.

**Round 1 (historical):** new study under ``log/4080_restart/mosi_refine/``,
widening the first broad search (``p_beta`` 24/32, ``p_gamma`` 8, etc.).

**Round 2+ (default now):** best trial often **sat on bounds** of that space
(``B0_dim`` min, ``B1_dim`` max, ``max_recursion_depth`` min, ``dropout_prob`` max,
``syntax_temperature`` min). The ``SEARCH_SPACE`` below extends those edges:

- ``B0_dim``: add **32, 48** below 64.
- ``B1_dim``: add **256** above 128.
- ``max_recursion_depth``: add **2** below 3 (``train.py`` allows ``>=1``).
- ``dropout_prob``: add **0.55, 0.6** above 0.5.
- ``syntax_temperature``: add **0.15, 0.2** below 0.25.

Optuna 4 cannot widen an existing study's categoricals: use a **new** output
dir + study name (defaults: ``mosi_refine2``, ``ithp_mosi_mae_refine2``). Warm
start imports history from the previous refine sqlite
(``--seed_sqlite`` / ``--seed_study_name``).

Typical launch:

    nohup .../python -u scripts/optuna_mosi_refine.py --gpu 0 \\
        > log/4080_restart/mosi_refine2/optuna_parent.log 2>&1 &
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import optuna


SEARCH_SPACE = {
    "learning_rate": [5e-6, 1e-5, 2e-5, 3e-5],
    "p_beta": [4, 8, 16, 24, 32],
    "p_gamma": [8, 16, 32, 64],
    # Prior best often used B0=64 (old min) / B1=128 (old max) — explore below/above.
    "B0_dim": [32, 48, 64, 128, 256],
    "B1_dim": [32, 64, 128, 256],
    # Best often used depth=3 (old min) — try shallower 2.
    "max_recursion_depth": [2, 3, 4, 5],
    "halting_threshold": [0.02, 0.0285, 0.04, 0.06, 0.08, 0.1],
    # Best often used dropout=0.5 (old max) — try slightly higher.
    "dropout_prob": [0.2, 0.3, 0.5, 0.55, 0.6],
    "silver_span_loss_weight": [0.05, 0.1, 0.2],
    # Best often used syntax_temperature=0.25 (old min) — try lower.
    "syntax_temperature": [0.15, 0.2, 0.25, 0.35, 0.5, 1.0, 2.0],
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
    "acc2_no_zero": ["acc2_no_zero", "test_acc"],
    "f1_score_no_zero": ["f1_score_no_zero", "f1_score"],
}

PYTHON = "/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python"

GRACEFUL_STOP_REQUESTED = False
GRACEFUL_STOP_SIGNAL = None


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--random_trials", default=30, type=int)
    parser.add_argument("--tpe_trials", default=120, type=int)
    parser.add_argument("--n_epochs", default=20, type=int)
    parser.add_argument("--seed", default=128, type=int)
    parser.add_argument(
        "--output_dir",
        default="log/4080_restart/mosi_refine2",
        type=str,
        help="New directory for this study (do not reuse an old sqlite with a smaller space).",
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
    parser.add_argument("--early_stopping_patience", default=0, type=int)
    parser.add_argument(
        "--study_name",
        default="ithp_mosi_mae_refine2",
        type=str,
        help="Must be a new study name when SEARCH_SPACE changes.",
    )
    parser.add_argument("--tpe_startup_trials", default=10, type=int)
    parser.add_argument(
        "--seed_sqlite",
        default="log/4080_restart/mosi_refine/optuna_study.sqlite3",
        type=str,
        help="Prior completed refine study to import as frozen history (default: round-1 refine).",
    )
    parser.add_argument(
        "--seed_study_name",
        default="ithp_mosi_mae_refine",
        type=str,
        help="Study name inside --seed_sqlite (default: ithp_mosi_mae_refine).",
    )
    parser.add_argument(
        "--import_mode",
        default="history",
        choices=["history", "enqueue", "none"],
        help="history: copy prior complete trials as FrozenTrial snapshots (no retrain); "
             "enqueue: queue prior top-K configs to be RE-RUN once; none: cold start.",
    )
    parser.add_argument("--seed_top_k", default=5, type=int)
    return parser.parse_args()


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


def build_train_command(config, n_epochs, selection_metric, early_stopping_patience):
    command = [
        PYTHON,
        "-u",
        "train.py",
        "--dataset",
        "mosi",
        "--n_epochs",
        str(n_epochs),
        "--silver_span_cache",
        "datasets/mosi_silver_spans.pkl",
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


def suggest_config(trial):
    return {
        name: trial.suggest_categorical(name, choices)
        for name, choices in SEARCH_SPACE.items()
    }


def count_completed_trials(study, phase_name):
    return sum(
        1
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs.get("phase") == phase_name
    )


def summarize_trials(study, primary_metric, limit=15):
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
        "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
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
    with open(output_dir / "study_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def clip_to_space(config):
    """Snap each value to the nearest legal choice in SEARCH_SPACE."""
    snapped = {}
    for name, choices in SEARCH_SPACE.items():
        value = config.get(name)
        if value is None:
            continue
        if value in choices:
            snapped[name] = value
        else:
            # Use string comparison for float keys to avoid precision surprises.
            str_choices = {str(c): c for c in choices}
            if str(value) in str_choices:
                snapped[name] = str_choices[str(value)]
            else:
                # Fallback: nearest numeric.
                try:
                    snapped[name] = min(choices, key=lambda c: abs(float(c) - float(value)))
                except Exception:
                    snapped[name] = choices[0]
    return snapped


def _load_prior_study(args):
    if not args.seed_sqlite or not os.path.exists(args.seed_sqlite):
        print(f"[seed] no prior sqlite at {args.seed_sqlite}; skipping warm start.")
        return None
    try:
        return optuna.load_study(
            study_name=args.seed_study_name,
            storage=f"sqlite:///{os.path.abspath(args.seed_sqlite)}",
        )
    except KeyError:
        print(f"[seed] could not find study {args.seed_study_name!r} in {args.seed_sqlite}")
        return None


def enqueue_prior_best(study, args):
    """Queue the top-K prior configs to be RE-RUN in the new space."""
    prior = _load_prior_study(args)
    if prior is None:
        return 0
    completed = [t for t in prior.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[seed] prior study has no complete trials; skipping warm start.")
        return 0
    reverse = METRIC_DIRECTIONS[args.primary_metric] == "maximize"
    completed.sort(
        key=lambda t: (
            t.value
            if t.value is not None
            else (float("-inf") if reverse else float("inf"))
        ),
        reverse=reverse,
    )
    enqueued = 0
    seen_sig = set()
    for prior_trial in completed[: max(0, args.seed_top_k)]:
        cfg = prior_trial.user_attrs.get("config") or {}
        snapped = clip_to_space(cfg)
        signature = json.dumps(snapped, sort_keys=True)
        if signature in seen_sig:
            continue
        seen_sig.add(signature)
        study.enqueue_trial(snapped)
        enqueued += 1
        print(f"[seed] enqueued prior trial #{prior_trial.number} value={prior_trial.value:.4f} -> {snapped}")
    print(f"[seed] enqueued {enqueued} warm-start configs from {args.seed_study_name}")
    return enqueued


def import_prior_history(study, args):
    """Import every prior completed trial as a FrozenTrial snapshot.

    TPE uses this history directly for density estimation without re-running
    anything. The prior trials' recorded distributions are rebuilt against the
    new SEARCH_SPACE so subsequent ``suggest_categorical`` calls remain
    compatible (Optuna 4 refuses mismatched CategoricalDistributions).
    """
    prior = _load_prior_study(args)
    if prior is None:
        return 0
    completed = [t for t in prior.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[seed] prior study has no complete trials; skipping history import.")
        return 0

    new_distributions = {
        name: optuna.distributions.CategoricalDistribution(choices=tuple(choices))
        for name, choices in SEARCH_SPACE.items()
    }

    imported = 0
    skipped = 0
    for prior_trial in completed:
        cfg = prior_trial.user_attrs.get("config")
        if cfg is None:
            # Fallback to the raw params recorded by the sampler.
            cfg = dict(prior_trial.params)
        if cfg is None:
            skipped += 1
            continue

        rebuilt_params = {}
        rebuilt_distributions = {}
        incompatible = False
        for name, distribution in new_distributions.items():
            if name not in cfg:
                continue
            value = cfg[name]
            # Snap numeric types so ``value in distribution.choices`` works reliably.
            if value in distribution.choices:
                rebuilt_params[name] = value
            else:
                snapped = clip_to_space({name: value}).get(name)
                if snapped is None or snapped not in distribution.choices:
                    incompatible = True
                    break
                rebuilt_params[name] = snapped
            rebuilt_distributions[name] = distribution

        if incompatible:
            skipped += 1
            continue

        user_attrs = dict(prior_trial.user_attrs) if prior_trial.user_attrs else {}
        user_attrs["phase"] = f"imported:{args.seed_study_name}"
        user_attrs["imported_from_trial"] = prior_trial.number

        frozen = optuna.trial.create_trial(
            params=rebuilt_params,
            distributions=rebuilt_distributions,
            value=prior_trial.value,
            user_attrs=user_attrs,
            state=optuna.trial.TrialState.COMPLETE,
        )
        study.add_trial(frozen)
        imported += 1

    print(f"[seed] imported {imported} prior trials into {args.study_name} (skipped {skipped})")
    return imported


def run_trial(trial, args, phase_name, work_dir, output_dir):
    config = suggest_config(trial)
    phase_dir = output_dir / "trial_logs" / phase_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    log_path = phase_dir / f"trial_{trial.number:04d}.log"

    trial.set_user_attr("phase", phase_name)
    trial.set_user_attr("config", config)
    trial.set_user_attr("log_path", str(log_path))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    command = build_train_command(config, args.n_epochs, args.selection_metric, args.early_stopping_patience)

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

    metrics = parse_test_line(str(log_path))
    trial.set_user_attr("elapsed_seconds", round(elapsed, 1))
    trial.set_user_attr("returncode", process.returncode)
    trial.set_user_attr("metrics", metrics)

    if process.returncode != 0 or metrics is None:
        raise RuntimeError(
            f"Trial {trial.number} failed: rc={process.returncode}"
        )

    primary = metric_value(metrics, args.primary_metric)
    if primary is None:
        raise RuntimeError(f"Trial {trial.number} missing primary metric '{args.primary_metric}'")

    print(
        f"[{phase_name}][Trial {trial.number}] DONE in {elapsed:.0f}s — "
        f"{args.primary_metric}={format_metric(metrics, args.primary_metric)}, "
        f"f1_no_zero={format_metric(metrics, 'f1_score_no_zero')}, "
        f"mae={metrics.get('mae', 'NA')}, best_epoch={metrics.get('best_epoch', 'NA')}",
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = (output_dir / "optuna_study.sqlite3").resolve()
    storage_uri = f"sqlite:///{storage_path}"
    study_name = args.study_name

    print(f"=== MOSI Optuna REFINE: gpu={args.gpu} ===")
    print(f"Output: {output_dir}")
    print(f"Storage: {storage_uri}")
    print(f"Study: {study_name}")
    print(f"Primary metric: {args.primary_metric} ({METRIC_DIRECTIONS[args.primary_metric]})")
    print(f"Selection metric: {args.selection_metric}")
    print(f"Epochs per trial: {args.n_epochs}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    print(f"Search space: {json.dumps(SEARCH_SPACE, ensure_ascii=False)}")

    # Warm-start the new study from the prior study. ``import_mode=history``
    # copies every prior completed trial as FrozenTrial snapshots so TPE
    # benefits from the full density right away (no retrain cost).
    warm_study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        load_if_exists=True,
        direction=METRIC_DIRECTIONS[args.primary_metric],
    )
    already_has_trials = bool(warm_study.trials)
    if not already_has_trials:
        if args.import_mode == "history":
            import_prior_history(warm_study, args)
        elif args.import_mode == "enqueue":
            enqueue_prior_best(warm_study, args)
        else:
            print("[seed] import_mode=none; skipping warm start.")
    else:
        print(f"[seed] study already contains {len(warm_study.trials)} trials; skipping warm start.")

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
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        print("=== BEST TRIAL ===")
        print(f"trial={best.number}  phase={best.user_attrs.get('phase')}")
        print(f"{args.primary_metric}={format_metric(best.user_attrs.get('metrics'), args.primary_metric)}")
        print(f"config={best.user_attrs.get('config')}")


if __name__ == "__main__":
    main()
