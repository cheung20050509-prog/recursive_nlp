# Information-Theoretic Hierarchical Perception (ITHP)

[![GitHub stars](https://img.shields.io/github/stars/joshuaxiao98/ITHP.svg?style=social&label=Star)](https://github.com/joshuaxiao98/ITHP/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/joshuaxiao98/ITHP.svg?style=social&label=Fork)](https://github.com/joshuaxiao98/ITHP/network/members)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/joshuaxiao98/58265fa4270f19e5eea307b5cf56448f/ithp_test.ipynb)

This repository hosts the official code for the paper "Information-Theoretic Hierarchical Perception for Multimodal Learning."

## Overview

Drawing on neurological models, the ITHP model employs the information bottleneck method to form compact and informative latent states, forging connections across modalities. Its hierarchical architecture incrementally distills information, offering a novel multimodal learning approach.

![Model](./assets/Model.png)

## Quick Start

1. Clone the repository and install dependencies:

   ```bash
   git clone https://github.com/joshuaxiao98/ITHP.git
   pip install -r requirements.txt
   ```

2. Download the datasets to `./datasets` by running `download_datasets.sh`. For details, see the [multimodal transformer dataset guide](https://github.com/WasifurRahman/BERT_multimodal_transformer).

3. Train the model on `MOSI` or `MOSEI` datasets using the `--dataset` flag:

   ```bash
   python train.py --dataset mosi   # For MOSI (default)
   python train.py --dataset mosei  # For MOSEI
   ```

4. Optional: run hyperparameter search scripts under `scripts/`:

   ```bash
   python scripts/random_search.py --dataset mosi --gpu 0 --n_trials 20
   python scripts/optuna_search.py --dataset mosei --gpu 0 --random_trials 20 --tpe_trials 40
   python scripts/optuna_local_refine_search.py --dataset mosi --gpu 0 --random_trials 20 --local_trials 40 --force_random_phase1
   ```

## Training Options

- Customize `train.py` for variable, loss function, or output modifications.
- Reduce `max_seq_length` from the default `50` for memory efficiency.
- Adjust `train_batch_size` to fit memory constraints.

## Silver Span Supervision

The recursive composer can consume an offline silver constituency cache and add a span-level auxiliary loss during training without changing the runtime model path.

1. Install parser dependencies in a separate preprocessing environment, for example `benepar` and `nltk`, then download a parser model such as `benepar_en3`.
2. Build a cache aligned with the existing dataset order:

    ```bash
    python scripts/build_silver_span_cache.py \
       --dataset-path datasets/mosi.pkl \
       --output-path datasets/mosi_silver_spans.pkl \
       --parser-model benepar_en3
    ```

3. Train with span supervision enabled:

    ```bash
    python train.py \
       --dataset mosi \
       --silver_span_cache datasets/mosi_silver_spans.pkl \
       --silver_span_loss_weight 0.1
    ```

If `--silver_span_cache` is omitted, training falls back to the original v11 behavior.

## Hyperparameter Search

This repository includes three search entrypoints for the recursive ITHP variant. All of them launch `train.py`, write per-trial logs, and rank trials with a user-selected primary metric.

### 1. Random Search

Use `scripts/random_search.py` for a simple baseline sweep over the discrete search space.

```bash
python scripts/random_search.py \
   --dataset mosi \
   --gpu 0 \
   --n_trials 50 \
   --primary_metric acc2_no_zero \
   --selection_metric mae \
   --output_dir search_results_acc2_no_zero_50trial
```

Outputs are written to `<output_dir>/<dataset>/`, including one `trial_*.log` file per run and a `summary.jsonl` file that can be reused by later Optuna stages.

### 2. Broad Optuna Search

Use `scripts/optuna_search.py` for the original two-phase Optuna workflow:

- Phase 1: `RandomSampler` for broad exploration.
- Phase 2: `TPESampler` on the same study for global refinement.

```bash
python scripts/optuna_search.py \
   --dataset mosei \
   --gpu 0 \
   --random_trials 50 \
   --tpe_trials 100 \
   --primary_metric mae \
   --selection_metric mae \
   --output_dir optuna_results_mae
```

Artifacts are written to `<output_dir>/<dataset>/`, including `optuna_study.sqlite3`, `study_summary.json`, and `trial_logs/`.

### 3. Local Refine Optuna Search

Use `scripts/optuna_local_refine_search.py` when you want the second stage to stay near the best phase-1 region instead of continuing a global search.

- Phase 1 can import completed random-search results from `summary.jsonl`, or run a fresh random phase.
- Phase 2 builds a narrowed local neighborhood around the best phase-1 configuration and runs TPE only inside that neighborhood.
- For `mosi`, the script includes a prior anchor configuration and a narrower phase-1 search space based on previous Optuna results.

```bash
python scripts/optuna_local_refine_search.py \
   --dataset mosi \
   --gpu 0 \
   --random_trials 60 \
   --local_trials 60 \
   --primary_metric mae \
   --selection_metric mae \
   --output_dir optuna_results_local_refine_mae \
   --study_prefix ithp_local_refine \
   --force_random_phase1
```

By default, if `--force_random_phase1` is not set, the script will try to import phase-1 results from:

- `search_results_acc2_no_zero_50trial/<dataset>/summary.jsonl`
- `search_results/<dataset>/summary.jsonl`

Useful flags:

- `--phase1_summary`: explicitly point to a previous `summary.jsonl` file.
- `--local_radius`: control how many neighboring categorical choices are kept around the phase-1 best configuration.
- `--tpe_startup_trials`: number of startup trials before local TPE begins modeling.

The local-refine workflow also writes `optuna_study.sqlite3`, `study_summary.json`, and per-phase trial logs under `<output_dir>/<dataset>/`.

## Citation

Please cite the following paper if this model assists your research:

```bibtex
@inproceedings{
xiao2024neuroinspired,
title={Neuro-Inspired Information-Theoretic Hierarchical Perception for Multimodal Learning},
author={Xiongye Xiao and Gengshuo Liu and Gaurav Gupta and Defu Cao and Shixuan Li and Yaxing Li and Tianqing Fang and Mingxi Cheng and Paul Bogdan},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=Z9AZsU1Tju}
}
```

Experiment with the model in Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/joshuaxiao98/58265fa4270f19e5eea307b5cf56448f/ithp_test.ipynb)
