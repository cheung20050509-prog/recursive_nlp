# Recursive-ITHP

**Recursive Information Theoretical Halting Process (Recursive-ITHP)** extends the ITHP multimodal fusion idea with a **learned halting** mechanism: the model may stop early when the latent state is sufficient, or recurse deeper when integrating audio / video under Information Bottleneck-style objectives.

Upstream ITHP reference: [joshuaxiao98/ITHP](https://github.com/joshuaxiao98/ITHP). This fork adds regression and classification paths, **CH-SIMS v2**, and **Optuna** tooling.

## What is in this repository

- **CMU-MOSI / CMU-MOSEI**: recursive sentiment training (`train.py`), silver-span syntax losses where applicable, KuDA-style test metrics.
- **CH-SIMS v2 (`simsv2`)**: MMSA-style pickled features, `bert-base-chinese` (or local weights via `--model`), normalization helpers in `simsv2_data.py`, metrics in `simsv2_metrics.py`.
- **MUStARD / UR-FUNNY**: HKT-paper-style binary classification (`train_hkt_binary.py`) and Optuna driver `scripts/optuna_hkt_search.py` (ALBERT / pickle splits).
- **Hyperparameter search**: two-phase Random + TPE Optuna studies (`scripts/optuna_search.py` for `mosi` / `mosei` / `simsv2`; SQLite storage, resumable).
- **Paper draft assets**: `recursive_ITHP_manuscript/` (LaTeX), `baseline_table.tex` (baseline grids; build PDFs locally, not committed).

## Requirements

Python 3.x, PyTorch, and dependencies from `requirements.txt`. An internal environment name used in scripts is `ITHP5090` (adjust paths in `scripts/optuna_search.py` if needed).

## Quick start

Clone (default branch or `regress_class_optuna`):

```bash
git clone https://github.com/cheung20050509-prog/recursive_nlp.git
cd recursive_nlp
pip install -r requirements.txt
```

### Train (examples)

```bash
# CMU sentiment (BERT-base English features + recursive ITHP)
python train.py --dataset mosi
python train.py --dataset mosei

# CH-SIMS v2 — use MMSA-normalized pickle; pass Chinese BERT path
python train.py --dataset simsv2 --model /path/to/bert-base-chinese
```

### HKT binary (humor / sarcasm)

```bash
python train_hkt_binary.py --dataset mustard --train_batch_size 32
python train_hkt_binary.py --dataset urfunny --train_batch_size 32
```

### Optuna

```bash
# MOSI / MOSEI (default output roots differ; override with --output_dir)
python scripts/optuna_search.py --dataset mosei --gpu 0

# SIMSv2 — defaults include smaller trial counts and CH-specific search space;
# default --output_dir is log/4080_restart (study DB under log/4080_restart/simsv2/)
python scripts/optuna_search.py --dataset simsv2 --gpu 0 --primary_metric mae
```

HKT classification search:

```bash
python scripts/optuna_hkt_search.py --help
```

SIMSv2 pickle preparation (when building from raw MMSA assets):

```bash
python scripts/build_simsv2_ithp_pkl.py --help
bash scripts/download_bert_chinese.sh   # optional local weights
```

## Results (indicative)

Numbers depend on split, seed, and Optuna budget. Recent Optuna-best **test** checkpoints are roughly:

| Dataset | Notes |
|--------|--------|
| **MOSI** | Strong **MAE / Corr** vs many published rows; Acc2/F1 competitive with top baselines (see `baseline_table.tex`). |
| **MOSEI** | **MAE / Corr** competitive; not every Acc column leads the public table. |
| **CH-SIMS v2** | Solid run; **KuDA-class rows remain ahead** on several columns — report as an extra Chinese benchmark, not blanket SOTA. |
| **MUStARD / UR-FUNNY** | HKT-fair pipeline; test accuracy still **below published MOAC** on the same-style table — useful for coverage or appendix. |

Treat `baseline_table.tex` as the single source for the exact figures you cite in the paper.

## Layout

| Path | Role |
|------|------|
| `train.py` | Main MOSI/MOSEI/SIMSv2 trainer |
| `train_hkt_binary.py` | MUStARD / UR-FUNNY binary trainer |
| `Recursive_ITHP.py`, `ITHP.py` | Model cores |
| `simsv2_data.py`, `simsv2_metrics.py` | SIMSv2 IO / metrics |
| `global_configs.py` | Shared knobs |
| `scripts/optuna_search.py` | Optuna for mosi / mosei / simsv2 |
| `scripts/optuna_hkt_search.py` | Optuna for HKT datasets |
| `log/` | Ignored by git — store Optuna DBs and trial logs here |

## Branching

Active development for regression, classification, and Optuna integration lives on **`regress_class_optuna`**.

## License

Follow the license of the upstream ITHP project unless otherwise specified in this repository.
