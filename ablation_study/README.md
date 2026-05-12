# ITHP ablation study (MOSI)

This directory holds **MOSI component ablations** using the same stack as [`fixed_experiment/`](ITHP/recursive_ITHP/fixed_experiment/README.md) (`run_fixed.py` → `fixed_training.train`). Paths are rooted at:

`/root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study`

**Default [`manifest.json`](ITHP/recursive_ITHP/ablation_study/manifest.json)** runs **four** MOSI jobs only (full + A1 + A3 + A4). JSON presets for **MOSEI** and **UR-FUNNY** remain under [`configs/mosei/`](ITHP/recursive_ITHP/ablation_study/configs/mosei) and [`configs/urfunny/`](ITHP/recursive_ITHP/ablation_study/configs/urfunny) if you want to launch them manually; they are **not** queued by the default manifest.

## Runner (default manifest)

| Dataset | Runner | Entry point |
|--------|--------|-------------|
| MOSI | `fixed` | [`ITHP/recursive_ITHP/fixed_experiment/run_fixed.py`](ITHP/recursive_ITHP/fixed_experiment/run_fixed.py) → `python -m fixed_training.train` |

**Comparable to Optuna + `fixed_experiment` repro:** [`scripts/run_manifest_repro.sh`](ITHP/recursive_ITHP/ablation_study/scripts/run_manifest_repro.sh) runs [`run_manifest_parallel.py`](ITHP/recursive_ITHP/ablation_study/scripts/run_manifest_parallel.py) with **`--profile optuna -j 1`**, which appends **`--train_batch_size 8 --gradient_accumulation_step 1`** to each `run_fixed.py` invocation (same defaults as `scripts/optuna_mosi_refine.py` / `train.py`).

[`run_manifest_parallel.py`](ITHP/recursive_ITHP/ablation_study/scripts/run_manifest_parallel.py) defaults to **`--profile optuna`**. **Throughput / VRAM:** [`run_manifest_32g2.sh`](ITHP/recursive_ITHP/ablation_study/scripts/run_manifest_32g2.sh) uses **`--profile 32g2`** (4×2 batch for `fixed`); that **breaks strict match** to [`repro_mosi_refine2_t278.log`](ITHP/recursive_ITHP/fixed_experiment/repro_mosi_refine2_t278.log) unless you restore 8/1.

Sequential (no parallelism): [`scripts/run_manifest.sh`](ITHP/recursive_ITHP/ablation_study/scripts/run_manifest.sh) → `run_manifest.py` (no profile injection; configs still carry `extra_cli` batch 8/1 for MOSI).

## Ablation grid (MOSI, `fixed_training`)

| Label | Config | What changes vs full (trial 278) |
|-------|--------|-------------------------------------|
| Full | [`configs/mosi/full_t278.json`](ITHP/recursive_ITHP/ablation_study/configs/mosi/full_t278.json) | Baseline: `extra_cli` fixes **`acc7_loss_weight` 0.2** (train default), **`train_batch_size` 8**, **`gradient_accumulation_step` 1**. |
| A1 | `A1_no_syntax_sup.json` | `silver_span_loss_weight: 0` (syntax **supervision** off; fusion still on). |
| A3 | `A3_no_recursion_refine.json` | `max_recursion_depth: 1`. |
| A4 | `A4_ib_no_stage2.json` | `p_lambda: 0` (IB stage-2 not in combined IB objective). |

**Acc7 auxiliary loss:** not ablated. All MOSI configs keep **`acc7_loss_weight: 0.2`** in `extra_cli` (aligned with repro / Optuna recipe). Turning off Acc7 is treated as a training trick, not a component slot for this study.

**Note:** A paper-style “remove syntax structure entirely” would require model/code changes in `deberta_ITHP` forward; this study only toggles **losses** and depth as above.

[`configs/mosi/A2_no_acc7.json`](ITHP/recursive_ITHP/ablation_study/configs/mosi/A2_no_acc7.json) is **not** in the manifest (legacy / optional manual run).

### Other datasets (deferred)

- **MOSEI:** [`configs/mosei/*.json`](ITHP/recursive_ITHP/ablation_study/configs/mosei) — run with [`run_one_mmsa.sh`](ITHP/recursive_ITHP/ablation_study/scripts/run_one_mmsa.sh) or a custom manifest.
- **UR-FUNNY (HKT):** [`configs/urfunny/*.json`](ITHP/recursive_ITHP/ablation_study/configs/urfunny), [`launch_hkt_json.py`](ITHP/recursive_ITHP/ablation_study/scripts/launch_hkt_json.py) → [`train_hkt_binary.py`](ITHP/recursive_ITHP/train_hkt_binary.py). Baseline aligned to Optuna TPE trial 29 lives in [`full_best_trial29.json`](ITHP/recursive_ITHP/ablation_study/configs/urfunny/full_best_trial29.json) (source: [`trial_0029/result.json`](ITHP/recursive_ITHP/log/4080_restart/urfunny/trial_logs/tpe/trial_0029/result.json)).

## Run one experiment (MOSI)

```bash
bash /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/scripts/run_one_mmsa.sh \
  /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/configs/mosi/A1_no_syntax_sup.json
```

Dry-run (print `train` command):

```bash
python3 /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/fixed_experiment/run_fixed.py \
  --dry-run --config /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/configs/mosi/full_t278.json
```

## Run the full manifest (MOSI)

```bash
bash /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/scripts/run_manifest.sh
```

Dry-run (print all commands; expect **four** lines):

```bash
python3 /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/scripts/run_manifest_parallel.py --dry-run
```

Optional slicing (example: skip full, run next two):

```bash
python3 /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/scripts/run_manifest.py --start 1 --limit 2
```

### Parallel manifest (repro-default)

**Paper-comparable (recommended):** single concurrent job, batch 8/1.

```bash
bash /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/scripts/run_manifest_repro.sh
# same as:
# bash .../run_manifest_parallel.sh --profile optuna -j 1
```

Dry-run first three entries:

```bash
python3 /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/scripts/run_manifest_parallel.py --dry-run --limit 3
```

Expect `run_fixed.py ... --train_batch_size 8 --gradient_accumulation_step 1` on each line. **`--jobs` default is 2** on `run_manifest_parallel.py`; use **`-j 1`** when VRAM is tight.

### Optuna / repro verification

After a MOSI full run, compare the final `TEST:` line to [`fixed_experiment/repro_mosi_refine2_t278.log`](ITHP/recursive_ITHP/fixed_experiment/repro_mosi_refine2_t278.log).

```bash
python3 /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/fixed_experiment/run_fixed.py --dry-run \
  --config /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/configs/mosi/full_t278.json
```

## 32GB 单卡双路（吞吐 / 省显存，非严格复现）

**`--profile 32g2`** injects **`--train_batch_size 4 --gradient_accumulation_step 2`** for `fixed` runs when extras are empty. That **does not** match the default MOSI repro; use only if you need two concurrent jobs on one GPU.

```bash
bash /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/scripts/run_manifest_32g2.sh
```

**更高并发**：自行 `run_manifest_parallel.sh -j N` 并调低 `--fixed-extra` 里的 batch，用 `nvidia-smi` 实测。

### 手写 extra（覆盖 profile 或 `profile off`）

- `--fixed-extra '...'`：追加到每条 `run_fixed.py`（在 `--config` 之后，**覆盖** JSON / `extra_cli` 同名键）。
- `--hkt-extra '...'`：仅当你用 `launch_hkt_json.py` / UR-FUNNY JSON 时 relevant（默认 manifest 不含 HKT）。
- `--continue-on-error`：失败后仍继续排队后续任务。

示例（显式低 VRAM，**非** repro 默认）：

```bash
bash /root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP/ablation_study/scripts/run_manifest_parallel.sh \
  --profile off -j 2 \
  --fixed-extra '--train_batch_size 4 --gradient_accumulation_step 2'
```

**`launch_hkt_json.py`**（UR-FUNNY manual）：`--dry-run` 须放在最前，例如  
`python3 .../launch_hkt_json.py --dry-run configs/urfunny/full_best_trial29.json`.

## Layout

（本目录位于 `ITHP/recursive_ITHP/ablation_study/`。）

```
recursive_ITHP/ablation_study/
  README.md
  manifest.json          # default: 4 MOSI entries
  configs/
    mosi/                # full_t278, A1, A3, A4 (+ optional A2_no_acc7.json, not in manifest)
    mosei/               # optional presets (not in default manifest)
    urfunny/             # optional HKT argv JSONs (not in default manifest)
  scripts/
    run_one_mmsa.sh
    run_one_urfunny.sh
    launch_hkt_json.py
    run_manifest.py
    run_manifest.sh
    run_manifest_parallel.py
    run_manifest_parallel.sh
    run_manifest_repro.sh
    run_manifest_32g2.sh
    start_ablation_no_nohup.sh
```
