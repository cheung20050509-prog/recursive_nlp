# Fixed experiments (MOSI refine2 / MOSEI best trials)

Frozen hyperparameters from Optuna studies under `ITHP/recursive_ITHP/log/4080_restart/`. This directory is **self-contained for training code**: runs use the embedded stack in [`fixed_training/`](fixed_training/) (a copy of the upstream `train.py` + model modules). **Data and silver caches** still live under [`ITHP/recursive_ITHP/datasets/`](../datasets/); at startup the embedded train script `chdir`s there so relative paths behave like the original.

## Layout

| Path | Role |
|------|------|
| [`fixed_training/`](fixed_training/) | Embedded training stack (`train.py`, `deberta_ITHP.py`, `ITHP.py`, `global_configs.py`, `simsv2_metrics.py`, `_paths.py`) |
| [`experiments/registry.py`](experiments/registry.py) | Preset names, JSON load, flat arg dict → `train.py` argv |
| [`config/*.json`](config/) | Frozen trial metadata + `hyperparameters` + `train_flags`; optional `extra_cli` for ablations |
| [`run_fixed.py`](run_fixed.py) | CLI: preset or `--config`, optional overrides, `--dry-run` |
| `run_*_best.sh` | Thin wrappers around `run_fixed.py` |

## What is fixed

| Run | Source | Study | Trial | `study_summary.json` |
|-----|--------|-------|-------|------------------------|
| MOSI | `mosi_refine2` | `ithp_mosi_mae_refine2` | 278 | `recursive_ITHP/log/4080_restart/mosi_refine2/study_summary.json` |
| MOSEI | `mosei` | `ithp_mosei_mae` | 95 | `recursive_ITHP/log/4080_restart/mosei/study_summary.json` |

Training flags align with [`scripts/optuna_search.py`](../scripts/optuna_search.py) `build_train_command` plus dataset-specific `n_epochs` / `early_stopping_patience`.

## Prerequisites

- Pickles under **`ITHP/recursive_ITHP`**: `datasets/mosi.pkl`, `datasets/mosei.pkl` (or your paths; not tracked by default).
- Silver caches: `datasets/mosi_silver_spans.pkl`, `datasets/mosei_silver_spans.pkl`.
- Python with torch/transformers (default: `ITHP5090` env); override with `PY=...`.

## How to run

**Recommended (embedded train + JSON preset):**

```bash
cd /path/to/recursive_nlp/ITHP/recursive_ITHP/fixed_experiment
python run_fixed.py mosi_refine2_t278
python run_fixed.py mosei_t95
python run_fixed.py --config config/mosei_best_trial95.json
python run_fixed.py mosei_t95 --dry-run   # print command only
```

**CLI overrides** (appended after preset; merged into config):

```bash
python run_fixed.py mosei_t95 --learning_rate 5e-6 --p_beta 8
```

**Shell wrappers** (same as above, forward args to `run_fixed.py`):

```bash
bash run_mosi_refine2_best.sh
bash run_mosei_best.sh
bash run_mosi_refine2_best.sh --dry-run
```

**Direct module** (from `fixed_experiment` cwd):

```bash
python -m fixed_training.train --dataset mosi --n_epochs 20 ...
```

**Both at once (nohup)** — [`run_both_repro_nohup.sh`](run_both_repro_nohup.sh) uses absolute paths; optional `MOSI_CUDA` / `MOSEI_CUDA` for two GPUs.

```bash
bash /path/to/ITHP/recursive_ITHP/fixed_experiment/run_both_repro_nohup.sh
```

When finished, compare the final `TEST:` line in logs to `study_summary.json` / original Optuna `trial_logs/.../trial_*.log`.

## Ablations

1. Copy a preset JSON under `config/` (e.g. `config/ablation_mosei_no_syntax.json`).
2. Edit `hyperparameters` / `train_flags`, or add an `extra_cli` object, e.g. `"extra_cli": {"silver_span_loss_weight": 0.0}`.
3. Run: `python run_fixed.py --config config/ablation_mosei_no_syntax.json` (plus optional CLI overrides).

Configs for inspection: [`config/*.json`](config/).

## Syncing with upstream `recursive_ITHP`

The embedded files duplicate [`train.py`](../train.py), [`deberta_ITHP.py`](../deberta_ITHP.py), [`ITHP.py`](../ITHP.py), [`global_configs.py`](../global_configs.py), [`simsv2_metrics.py`](../simsv2_metrics.py). If you change the upstream training or model, **merge manually** into `fixed_training/` (keep relative imports and `ithp_workdir()` / `os.chdir` behavior).

## Note on outputs

`train.py` does **not** expose `--output_dir`; checkpoints and logs follow whatever the embedded `train.py` writes (typically under `recursive_ITHP` cwd after `chdir`). To isolate runs, use a separate working tree or symlinked `datasets/` as needed.
