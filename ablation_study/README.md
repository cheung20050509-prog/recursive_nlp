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

## t-SNE 可视化（MOSI ablation）

表征取 **DeBERTa 骨干融合后的 `pooled_output`**（`model.dberta(...)` 的首个返回值，与分类头前的句向量一致）。流程：**训练时保存 best checkpoint → 导出测试集 embedding → sklearn t-SNE → matplotlib 出图**。

### 1. 训练并保存权重

在原有 `run_fixed.py` / manifest 命令后追加 **`--save_checkpoint`**（路径自定；目录不存在会自动创建）。每次验证集指标刷新 best 时都会覆盖写入该文件。

```bash
cd /path/to/recursive_ITHP/fixed_experiment
python3 run_fixed.py --config ../ablation_study/configs/mosi/full_t278.json \
  --train_batch_size 8 --gradient_accumulation_step 1 \
  --save_checkpoint ../ablation_study/checkpoints/mosi_full_t278.pt
```

或在实验 JSON 的 **`extra_cli`** 里加 `"save_checkpoint": "../ablation_study/checkpoints/<id>.pt"`（相对路径相对于 **`fixed_experiment`** 当前工作目录）。

**批量（manifest）**：`scripts/run_manifest_parallel.py` 支持 **`--save-checkpoints`**，会为每条 `fixed` 任务自动追加  
`--save_checkpoint <ablation_study>/checkpoints/<manifest id>.pt`（绝对路径）。可选 **`--checkpoint-dir DIR`** 覆盖保存目录。例如两条并发、四任务全开：

```bash
cd /path/to/recursive_ITHP/ablation_study
PY=/path/to/env/bin/python3
nohup "$PY" -u scripts/run_manifest_parallel.py --profile optuna -j 2 --save-checkpoints >>logs/mosi_ablation_ckpt.log 2>&1 &
```

Checkpoint 内容：`{"model": state_dict, "args": vars(args)}`，供导出脚本复现数据与模型配置。

### 2. 导出测试集 embeddings

```bash
cd /path/to/recursive_ITHP/fixed_experiment
python3 ../ablation_study/scripts/export_mosi_embeddings.py \
  --checkpoint ../ablation_study/checkpoints/mosi_full_t278.pt \
  --output ../ablation_study/figures/embeddings_full.npz \
  --variant mosi_full_t278
```

会生成 **`embeddings_full.npz`**（`embeddings`、`labels`）和同名的 **`embeddings_full.meta.json`**（`variant`、`dataset`、`n_samples` 等）。测试集 **`DataLoader` 使用 `shuffle=False`**，保证各 ablation 的样本行顺序一致，便于 `--combined` 对比。

### 3. 画 t-SNE

默认：**每个 npz 一个子图**，颜色为 **情感分 7 档分箱**（边界默认 `linspace(-3, 3, 8)`，适用于 MOSI 连续标签）。依赖 **`matplotlib`**（已写入仓库根目录 [`requirements.txt`](ITHP/recursive_ITHP/requirements.txt)）。

```bash
python3 ../ablation_study/scripts/plot_tsne_mosi.py \
  ../ablation_study/figures/embeddings_full.npz \
  ../ablation_study/figures/embeddings_A1.npz \
  ../ablation_study/figures/embeddings_A3.npz \
  ../ablation_study/figures/embeddings_A4.npz \
  --out ../ablation_study/figures/tsne_mosi_ablation.png
```

- **`--combined`**：把所有模型的 embedding **纵向拼接**（要求各文件 **样本数相同**），`StandardScaler` 后跑一次 t-SNE，按 **variant 名称** 上色；输出额外一张 `*_combined.png`。
- **`--perplexity`** / **`--seed`**：调节 t-SNE；小样本时 perplexity 会自动上限为 `n-1`。
- **`--edges`**：自定义分箱边界（逗号分隔浮点数），覆盖默认 MOSI 七档。

**论文式「序关系 / 三分类」图**（参考 *without vs with ordinal* 那种 **+ / o / x** 分色）：

- **`--color-mode trinary`**：按连续标签划 **负 / 中 / 正** 三类（默认 `label < -0.5` 为负，`> 0.5` 为正，中间为中性；可用 **`--trinary-neg-th`** / **`--trinary-pos-th`** 改阈值）。
- **`--panel-tags '(a),(b)'`**：逗号分隔、与子图数量一致，会拼进子图标题（用于左右对比）。

```bash
python3 ../ablation_study/scripts/plot_tsne_mosi.py \
  ../ablation_study/figures/embeddings_mosi_full_t278.npz \
  ../ablation_study/figures/embeddings_mosi_A3_no_recursion_refine.npz \
  --out ../ablation_study/figures/tsne_compare_full_vs_A3.png \
  --color-mode trinary --panel-tags '(a),(b)'
```

四路 ablation 的 trinary 总览示例：`--out .../tsne_mosi_ablation_trinary.png --color-mode trinary`（已生成于 `figures/` 时可复用同一命令）。

**让消融「看得出差别」（重要）**

- **问题**：默认 `--layout separate` 对每个模型**单独**做 t-SNE，**横纵轴不可跨子图比较**，四张图往往很像，难以支撑 ablation 叙事。
- **推荐**：**`--layout joint`** — 把所有模型的 embedding **按行对齐后纵向拼接**，只做 **一次** t-SNE（`4N` 个点），再按模型切回 `N` 个点画子图，并 **统一 xlim/ylim**。这样四个面板处在 **同一 2D 坐标系**，云团的平移/形变更可比。示例：

```bash
python3 ../ablation_study/scripts/plot_tsne_mosi.py \
  ../ablation_study/figures/embeddings_mosi_full_t278.npz \
  ../ablation_study/figures/embeddings_mosi_A1_no_syntax_sup.npz \
  ../ablation_study/figures/embeddings_mosi_A3_no_recursion_refine.npz \
  ../ablation_study/figures/embeddings_mosi_A4_ib_no_stage2.npz \
  --out ../ablation_study/figures/tsne_mosi_ablation_joint_trinary.png \
  --color-mode trinary --layout joint
```

- **数值辅助**（与图一起写进附录最稳）：[`scripts/embedding_drift_metrics.py`](ITHP/recursive_ITHP/ablation_study/scripts/embedding_drift_metrics.py) 以 **第一个 npz 为 reference**，对每个 ablation 打印相对 full 的 **逐样本 L2 / cosine 距离**均值，以及 **trinary 标签上的 silhouette**（越高表示三类在欧氏空间里可分性越好，仅作辅助指标）。

```bash
python3 ../ablation_study/scripts/embedding_drift_metrics.py \
  ../ablation_study/figures/embeddings_mosi_full_t278.npz \
  ../ablation_study/figures/embeddings_mosi_A1_no_syntax_sup.npz \
  ...
```

- **仍建议保留**：`--combined` 单图按 **variant** 上色（看模型整体是否占不同区域）；与 **joint + trinary** 分工不同，可二选一或并列。

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
    export_mosi_embeddings.py   # checkpoint → .npz (+ .meta.json)
    plot_tsne_mosi.py            # .npz → t-SNE PNG(s); use --layout joint for ablation compare
    embedding_drift_metrics.py # L2/cosine vs reference + silhouette (trinary)
```
