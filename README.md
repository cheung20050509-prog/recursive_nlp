# RITHM

**RITHM** (**R**ecursive **I**nformation-**T**heoretic **H**alting **M**odel) is our implementation of **recursive multimodal fusion with learned halting**: the model may stop early when the latent state is sufficient, or recurse deeper when integrating audio / video under Information Bottleneck–style objectives. It builds on the ITHP line of work (see also *Recursive Information Theoretical Halting Process* in the paper draft).

Upstream ITHP reference: [joshuaxiao98/ITHP](https://github.com/joshuaxiao98/ITHP). This repository adds regression and classification paths, **CH-SIMS v2**, and **Optuna** tooling around **RITHM**.

## What is in this repository

- **CMU-MOSI / CMU-MOSEI**: **RITHM** sentiment training (`train.py`), silver-span syntax losses where applicable, KuDA-style test metrics.
- **CH-SIMS v2 (`simsv2`)**: MMSA-style pickled features, `bert-base-chinese` (or local weights via `--model`), normalization helpers in `simsv2_data.py`, metrics in `simsv2_metrics.py`.
- **MUStARD / UR-FUNNY**: HKT-paper-style binary classification (`train_hkt_binary.py`) and Optuna driver `scripts/optuna_hkt_search.py` (ALBERT / pickle splits). Optional **dev-tuned decision threshold** (`--decision_threshold_mode tune_on_valid` on the trainer, or Optuna `--decision-threshold-mode` / `--primary_metric valid_accuracy_threshold_tuned`) writes `threshold_tuning` in `result.json` (see trainer docstring). Optional **benepar silver constituency spans** on the **target utterance** (see silver coverage matrix below).
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
# CMU sentiment (BERT-base English features + RITHM)
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
# MOSI / MOSEI
python scripts/optuna_search.py --dataset mosei --gpu 0

# SIMSv2 — default --simsv2-search-space aligned: same categorical grid as MOSI/MOSEI
# (50 random + 100 TPE trials unless overridden). Default --output_dir log/4080_restart.
python scripts/optuna_search.py --dataset simsv2 --gpu 0 --primary_metric mae

# Legacy SIMSv2 grid (extra IB/schedule knobs; 20+60 trial defaults)
python scripts/optuna_search.py --dataset simsv2 --gpu 0 --simsv2-search-space wide
```

**SIMSv2 silver spans (aligned search):** Optuna matches MOSI/MOSEI on `silver_span_loss_weight` only when `datasets/simsv2_silver_spans.pkl` exists. Otherwise weights are restricted to `0.0`. Build the cache (benepar + normalized pickle; Chinese text may need a non-English parser model — see script docstring):

```bash
python scripts/build_simsv2_silver_span_cache.py --input datasets/simsv2.pkl \
  --output datasets/simsv2_silver_spans.pkl --parser-model benepar_en3 --device cpu
```

### Silver span coverage (cache + supervised syntax loss)

| Training entry | Dataset | Silver constituency cache | Span supervision in the model |
|----------------|---------|----------------------------|------------------------------|
| `train.py` | MOSI, MOSEI | `datasets/{dataset}_silver_spans.pkl` (or `--silver_span_cache`) | Yes: `silver_span_loss_weight * syntax_loss` when a cache is loaded |
| `train.py` | simsv2 | Optional: `scripts/build_simsv2_silver_span_cache.py` → `datasets/simsv2_silver_spans.pkl` | Same as above; Optuna restricts `silver_span_loss_weight` to `[0.0]` if the pickle is missing |
| `train_hkt_binary.py` | mustard, urfunny | Optional: `scripts/build_hkt_silver_span_cache.py` → `datasets/{mustard,urfunny}_silver_spans.pkl` | Yes when a cache is loaded: pass `--syntax_loss_weight` (same role as MOSI **`silver_span_loss_weight`**; there is no separate HKT flag name). `--syntax_loss_weight > 0` requires a readable cache. |

**HKT cache build** (English benepar default `benepar_en3`; use the same `--fold` / `--dataset_cache` / `--seed` / `--dev_ratio` as training when you rely on MUStARD k-fold or a custom pickle):

```bash
python scripts/build_hkt_silver_span_cache.py --dataset mustard
python scripts/build_hkt_silver_span_cache.py --dataset mustard --fold 0 --seed 5149 --dev-ratio 0.1
python scripts/build_hkt_silver_span_cache.py --dataset urfunny
```

**HKT Optuna:** `scripts/optuna_hkt_search.py` includes `syntax_loss_weight` in its search space; if `datasets/<dataset>_silver_spans.pkl` is missing, the driver prints a notice and restricts `syntax_loss_weight` to `[0.0]` (same idea as SIMSv2 aligned search).

**Study sqlite:** Switching between `aligned` and `wide` changes suggested parameter names; reuse the same `optuna_study.sqlite3` only if you know what you are doing. Prefer a new `--study_prefix` or `--output_dir` for a clean aligned run.

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

Numbers depend on split, seed, and Optuna budget. Recent **RITHM** Optuna-best **test** checkpoints are roughly:

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
| `Recursive_ITHP.py`, `ITHP.py` | **RITHM** model cores (legacy filenames; class names still use ITHP) |
| `simsv2_data.py`, `simsv2_metrics.py` | SIMSv2 IO / metrics |
| `global_configs.py` | Shared knobs |
| `scripts/optuna_search.py` | Optuna for mosi / mosei / simsv2 |
| `scripts/build_simsv2_silver_span_cache.py` | Normalize SIMSv2 pickle + benepar silver cache |
| `scripts/build_hkt_silver_span_cache.py` | Benepar silver cache for MUStARD / UR-FUNNY HKT pickles |
| `scripts/optuna_hkt_search.py` | Optuna for HKT datasets |
| `log/` | Ignored by git — store Optuna DBs and trial logs here |

## Branching

Active development for regression, classification, and Optuna integration lives on **`regress_class_optuna`**.

## License

Follow the license of the upstream ITHP project unless otherwise specified in this repository.
