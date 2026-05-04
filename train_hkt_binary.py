"""HKT-style binary classification trainer for MUStARD and UR-FUNNY.

- **Text backbone:** ``--backbone {auto,albert,deberta}``; ``--model`` may point
  to ``albert-base-v2`` (Recursive **ITHP** + ALBERT) or DeBERTa (default path).
- **HKT paper data/metrics (matalvepu):** ``--hkt_paper_style`` = 60+36 feature
  subset, train z-score, HCF on, binary F1; mutually exclusive with ``--github_style``.
- **F1:** ``--primary_f1 {binary,weighted}``; ``f1_hkt_paper`` in JSON mirrors
  sklearn binary F1 (comparable to README Accuracy / F-score columns).

HCF (4-dim Humor-Centric Features) flows through the multimodal encoder. Per
dimension z-score (unless ``--skip_normalize``). Model selection via
``--selection_metric``; ``--fold`` for MUStARD k-fold. Writes ``result.json`` per
run.

**Decision threshold:** Default is a fixed **0.5** on sigmoid probabilities (``best``
in ``result.json``). With ``--decision_threshold_mode tune_on_valid``, a scalar
``T`` is chosen on the **dev** split to maximize ``--threshold_tune_objective``,
then applied to **test**; results live under ``threshold_tuning`` (dev-based
selection can look optimistic on test vs strict 0.5 / paper protocols).

GitHub-style MHD/MSD compatibility knobs (match ``My_creation@MHD_MSD_optuna``
``data_humor.py`` + ``train_classify.py`` processing; disabled by default):

- ``--feature_dim_mode full``  keep every HKT feature column (acoustic=81,
  visual=91) instead of the default ``subset`` (60/36) matching that repo's
  ``global_configs``.
- ``--skip_normalize``  bypass z-score; pass raw per-word features through.
- ``--primary_f1 weighted``  report ``f1`` as weighted-F1 (also used for
  selection when ``--selection_metric f1``).
- ``--github_style``  single switch that implies all three plus
  ``--disable_hcf`` to mirror ``data_humor.py`` exactly.

Examples:

    # default HKT/ITHP behaviour
    python train_hkt_binary.py --dataset mustard --fold 0 --n_epochs 15 \\
        --output_dir log/4080_restart_hkt/mustard/fold_0

    # GitHub MHD_MSD_optuna-aligned processing
    python train_hkt_binary.py --dataset mustard --github_style \\
        --output_dir log/4080_restart_hkt/mustard_github

Paths assume the repo root ``/root/autodl-tmp/recursive_nlp/ITHP/recursive_ITHP``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AlbertConfig,
    AutoConfig,
    AutoTokenizer,
    DebertaV2Tokenizer,
    get_linear_schedule_with_warmup,
)

from deberta_ITHP import ITHP_DeBertaForBinaryClassification
from albert_ITHP import ITHP_AlbertForBinaryClassification
import global_configs
from global_configs import DEVICE
from hkt_data import (
    ACOUSTIC_DIM_ALL,
    HCF_DIM,
    HKT_ACOUSTIC_FEATURES,
    HKT_DEFAULT_MAX_SEQ_LENGTH,
    HKT_VISUAL_FEATURES,
    MUSTARD_NUM_FOLDS,
    VISUAL_DIM_ALL,
    load_hkt_dataset,
    normalize_dataset_name,
)
from silver_span_utils import (
    build_pair_target_syntax_span_mask,
    get_silver_span_record,
    load_hkt_silver_span_cache,
    resolve_hkt_silver_span_cache_path,
    target_words_from_p_field,
    tokenize_word_list,
)

# argparse: omit --syntax_loss_weight to use auto rule in main (see below).
_SYNTAX_LOSS_WEIGHT_ARG_DEFAULT = object()
_SILVER_SPAN_SYNTAX_DEFAULT = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/root/autodl-tmp/recursive_nlp/deberta-v3-base")
    parser.add_argument("--dataset", type=str, choices=["mustard", "urfunny", "sarcasm", "humor"], default="mustard")
    parser.add_argument("--dataset_cache", type=str, default="")
    parser.add_argument("--fold", type=int, default=-1,
                        help="MUStARD speaker-independent fold index (0..K-1). -1 uses the default HKT pickle split.")
    parser.add_argument("--dev_ratio", type=float, default=0.1,
                        help="Size of the dev hold-out carved out of the k-fold train split (mustard only).")
    parser.add_argument("--num_folds", type=int, default=MUSTARD_NUM_FOLDS)
    parser.add_argument("--max_seq_length", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--dev_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=5149)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_weight", type=str, default="")
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["valid_loss", "accuracy", "f1"],
        default="accuracy",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--inter_dim", default=256, type=int)
    parser.add_argument("--drop_prob", default=0.3, type=float)
    parser.add_argument("--p_lambda", default=0.3, type=float)
    parser.add_argument("--p_beta", default=8.0, type=float)
    parser.add_argument("--p_gamma", default=32.0, type=float)
    parser.add_argument("--beta_shift", default=1.0, type=float)
    parser.add_argument("--B0_dim", default=128, type=int)
    parser.add_argument("--B1_dim", default=64, type=int)
    parser.add_argument("--max_recursion_depth", default=3, type=int)
    parser.add_argument("--halting_threshold", default=0.0285, type=float)
    parser.add_argument("--syntax_temperature", default=1.0, type=float)
    parser.add_argument("--ib_loss_weight", default=-1.0, type=float,
                        help="If negative, defaults to 2/(p_beta+p_gamma).")
    parser.add_argument(
        "--syntax_loss_weight",
        default=_SYNTAX_LOSS_WEIGHT_ARG_DEFAULT,
        type=float,
        help=(
            "Weight on constituency span supervision (same as train.py silver_span_loss_weight). "
            "If omitted: 0.1 when datasets/<dataset>_silver_spans.pkl exists for urfunny, or for "
            "mustard with --fold -1 (official pickle split); else 0.0. MUStARD k-fold (--fold>=0) "
            "stays 0.0 unless you pass this flag and a fold-aligned silver cache."
        ),
    )
    parser.add_argument(
        "--silver_span_cache",
        type=str,
        default="",
        help=(
            "Path to a benepar silver span pickle (train/dev/test). "
            "When empty, uses datasets/<dataset>_silver_spans.pkl if present. "
            "Same role as train.py --silver_span_cache; pairs with --syntax_loss_weight "
            "(MOSI naming: silver_span_loss_weight)."
        ),
    )
    parser.add_argument("--hcf_dim", default=HCF_DIM, type=int)
    parser.add_argument("--disable_hcf", action="store_true",
                        help="Ignore HCF features even if present in the pickle.")
    parser.add_argument(
        "--feature_dim_mode",
        type=str,
        choices=["subset", "full"],
        default="subset",
        help=(
            "'subset' keeps HKT's original 60/36 columns (ITHP default). "
            "'full' keeps every HKT feature (81/91), matching the GitHub "
            "My_creation@MHD_MSD_optuna data_humor.py pipeline."
        ),
    )
    parser.add_argument(
        "--skip_normalize",
        action="store_true",
        help="Skip train-fit z-score on visual/acoustic/HCF (matches GitHub data_humor.py).",
    )
    parser.add_argument(
        "--primary_f1",
        type=str,
        choices=["binary", "weighted"],
        default="binary",
        help=(
            "Which F1 variant goes into the 'f1' metric field. 'binary' matches "
            "HKT-style reporting; 'weighted' matches the GitHub MHD_MSD_optuna "
            "score() helper."
        ),
    )
    parser.add_argument(
        "--github_style",
        action="store_true",
        help=(
            "Convenience preset: enables --feature_dim_mode full, --skip_normalize, "
            "--primary_f1 weighted, and --disable_hcf to mirror the GitHub "
            "MHD_MSD_optuna data processing 1:1."
        ),
    )
    parser.add_argument(
        "--hkt_paper_style",
        action="store_true",
        help=(
            "Align with matalvepu/HKT data protocol: 60+36 feature subset, train-fitted z-score, "
            "HCF on, binary F1 as primary. Mutually exclusive with --github_style."
        ),
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="auto",
        choices=["auto", "deberta", "albert"],
        help=(
            "Text encoder: 'albert' uses ITHP_AlbertForBinaryClassification + ALBERT; "
            "'deberta' uses DeBERTa. 'auto' uses AutoConfig on --model to pick."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="log/4080_restart_hkt")
    parser.add_argument("--run_name", type=str, default="",
                        help="Optional subdirectory under output_dir; defaults to <dataset>[_fold_k].")
    parser.add_argument(
        "--decision_threshold_mode",
        type=str,
        choices=["fixed", "tune_on_valid"],
        default="fixed",
        help=(
            "fixed: classify with probability >= 0.5 (HKT-style). "
            "tune_on_valid: after training, pick T on dev to maximize --threshold_tune_objective, "
            "then report valid/test metrics at T under result.json 'threshold_tuning'."
        ),
    )
    parser.add_argument(
        "--threshold_tune_objective",
        type=str,
        choices=["accuracy", "f1"],
        default="accuracy",
        help="When decision_threshold_mode=tune_on_valid: which dev metric to maximize when scanning T.",
    )
    parser.add_argument(
        "--threshold_grid_size",
        type=int,
        default=91,
        help="Number of equally-spaced thresholds in [0.05, 0.95]; 0.5 is always included.",
    )
    args = parser.parse_args()

    if args.github_style and args.hkt_paper_style:
        raise ValueError("Use either --github_style or --hkt_paper_style, not both")
    if args.github_style:
        args.feature_dim_mode = "full"
        args.skip_normalize = True
        args.primary_f1 = "weighted"
        args.disable_hcf = True
    if args.hkt_paper_style:
        # matalvepu HKT: subset features, HCF, z-score (train only); binary F1 like README F-score
        args.feature_dim_mode = "subset"
        args.skip_normalize = False
        args.primary_f1 = "binary"
        args.disable_hcf = False

    args.dataset = normalize_dataset_name(args.dataset)
    if args.max_seq_length <= 0:
        args.max_seq_length = HKT_DEFAULT_MAX_SEQ_LENGTH[args.dataset]
    if args.max_recursion_depth < 1:
        raise ValueError("max_recursion_depth must be at least 1")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if args.threshold_grid_size < 3:
        raise ValueError("threshold_grid_size must be at least 3")
    if args.ib_loss_weight < 0:
        args.ib_loss_weight = 2.0 / (args.p_beta + args.p_gamma)
    if args.hcf_dim < 1:
        raise ValueError("hcf_dim must be >= 1")
    if args.fold >= 0 and args.dataset != "mustard":
        raise ValueError("--fold is only meaningful for dataset=mustard")
    args.resolved_backbone = _resolve_backbone_name(args)
    return args


def _resolve_backbone_name(args):
    if args.backbone in ("albert", "deberta"):
        return args.backbone
    try:
        cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
        model_type = getattr(cfg, "model_type", None)
        if model_type == "albert":
            return "albert"
        if model_type in ("deberta", "deberta-v2"):
            return "deberta"
    except OSError:
        try:
            cfg = AutoConfig.from_pretrained(args.model, local_files_only=False)
            model_type = getattr(cfg, "model_type", None)
            if model_type == "albert":
                return "albert"
            if model_type in ("deberta", "deberta-v2"):
                return "deberta"
        except Exception:
            pass
    except Exception:
        pass
    p = (args.model or "").lower()
    if "albert" in p:
        return "albert"
    return "deberta"


args = parse_args()

_resolved_silver = resolve_hkt_silver_span_cache_path(args.dataset, args.silver_span_cache)
if args.syntax_loss_weight is _SYNTAX_LOSS_WEIGHT_ARG_DEFAULT:
    _silver_ok = bool(_resolved_silver and os.path.isfile(_resolved_silver))
    if args.dataset == "urfunny" and _silver_ok:
        args.syntax_loss_weight = _SILVER_SPAN_SYNTAX_DEFAULT
    elif args.dataset == "mustard" and args.fold < 0 and _silver_ok:
        args.syntax_loss_weight = _SILVER_SPAN_SYNTAX_DEFAULT
    else:
        args.syntax_loss_weight = 0.0
    if args.syntax_loss_weight > 0.0:
        print(
            "HKT_SILVER_SPAN: auto-enabled --syntax_loss_weight={} (cache={})".format(
                args.syntax_loss_weight, _resolved_silver
            ),
            flush=True,
        )
if args.syntax_loss_weight > 0.0 and (not _resolved_silver or not os.path.isfile(_resolved_silver)):
    raise ValueError(
        "When --syntax_loss_weight > 0, a readable silver span cache is required. "
        "Build with:\n  python scripts/build_hkt_silver_span_cache.py --dataset "
        f"{args.dataset}\nor pass --silver_span_cache to an existing pickle."
    )

global_configs.MAX_RECURSION_DEPTH = args.max_recursion_depth
dataset_config_key = args.dataset if args.feature_dim_mode == "subset" else f"{args.dataset}_full"
global_configs.set_dataset_config(dataset_config_key)
ACOUSTIC_DIM, VISUAL_DIM = global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM


class InputFeatures(object):
    def __init__(self, input_ids, visual, acoustic, hcf, input_mask, segment_ids, label_id, syntax_span_mask):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.hcf = hcf
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.syntax_span_mask = syntax_span_mask


def set_random_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}", flush=True)


def _truncate_context_prefix(tokens_a, tokens_b, max_length):
    """Drop context tokens from the front until [tokens_a][tokens_b] fits."""
    pop_count = 0
    while len(tokens_a) + len(tokens_b) > max_length:
        if len(tokens_a) == 0:
            tokens_b.pop()
        else:
            tokens_a.pop(0)
            pop_count += 1
    return pop_count


def get_inversion(tokens, spiece_marker="\u2581"):
    """Map each sub-token to its parent word index."""
    inversion_index = -1
    inversions = []
    for token in tokens:
        if spiece_marker in token:
            inversion_index += 1
        inversions.append(max(inversion_index, 0))
    return inversions


def align_word_features(word_features, inversions, feature_dim):
    if len(inversions) == 0:
        return np.zeros((0, feature_dim), dtype=np.float32)
    if word_features.shape[0] == 0:
        return np.zeros((len(inversions), feature_dim), dtype=np.float32)

    indices = np.asarray(inversions, dtype=np.int64)
    indices = np.clip(indices, 0, word_features.shape[0] - 1)
    return word_features[indices]


def prepare_deberta_pair_input(
        tokens_a, tokens_b,
        visual_a, visual_b,
        acoustic_a, acoustic_b,
        hcf_a, hcf_b,
        tokenizer,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]

    acoustic_zero = np.zeros((1, ACOUSTIC_DIM_ALL), dtype=np.float32)
    acoustic = np.concatenate((acoustic_zero, acoustic_a, acoustic_zero, acoustic_b, acoustic_zero), axis=0)
    if args.feature_dim_mode == "subset":
        acoustic = acoustic[:, HKT_ACOUSTIC_FEATURES]

    visual_zero = np.zeros((1, VISUAL_DIM_ALL), dtype=np.float32)
    visual = np.concatenate((visual_zero, visual_a, visual_zero, visual_b, visual_zero), axis=0)
    if args.feature_dim_mode == "subset":
        visual = visual[:, HKT_VISUAL_FEATURES]

    hcf_zero = np.zeros((1, args.hcf_dim), dtype=np.float32)
    hcf = np.concatenate((hcf_zero, hcf_a, hcf_zero, hcf_b, hcf_zero), axis=0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)
    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM), dtype=np.float32)
    visual_padding = np.zeros((pad_length, VISUAL_DIM), dtype=np.float32)
    hcf_padding = np.zeros((pad_length, args.hcf_dim), dtype=np.float32)
    acoustic = np.concatenate((acoustic, acoustic_padding), axis=0)
    visual = np.concatenate((visual, visual_padding), axis=0)
    hcf = np.concatenate((hcf, hcf_padding), axis=0)

    padding = [0] * pad_length
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, hcf, input_mask, segment_ids


def convert_hkt_examples_to_features(examples, tokenizer, silver_split_records=None):
    features = []

    for ex_index, example in enumerate(examples):
        (
            (p_words, p_visual, p_acoustic, p_hcf),
            (c_words, c_visual, c_acoustic, c_hcf),
            hid,
            label,
        ) = example

        text_a = ". ".join(c_words) if isinstance(c_words, (list, tuple)) else str(c_words)
        tokens_a = tokenizer.tokenize(text_a)

        use_silver = silver_split_records is not None
        if use_silver:
            words_tgt = target_words_from_p_field(p_words)
            tokens_b, inversions_b = tokenize_word_list(words_tgt, tokenizer)
        else:
            text_b = (p_words + ".") if isinstance(p_words, str) else (" ".join(p_words) + ".")
            tokens_b = tokenizer.tokenize(text_b)
            inversions_b = get_inversion(tokens_b)

        inversions_a = get_inversion(tokens_a)
        pop_count = _truncate_context_prefix(tokens_a, tokens_b, args.max_seq_length - 3)
        # Pop the front of inversions_a so it aligns with the shorter token_a.
        inversions_a = inversions_a[pop_count:]
        inversions_b = inversions_b[: len(tokens_b)]

        visual_a = align_word_features(np.asarray(c_visual, dtype=np.float32), inversions_a, VISUAL_DIM_ALL)
        visual_b = align_word_features(np.asarray(p_visual, dtype=np.float32), inversions_b, VISUAL_DIM_ALL)
        acoustic_a = align_word_features(np.asarray(c_acoustic, dtype=np.float32), inversions_a, ACOUSTIC_DIM_ALL)
        acoustic_b = align_word_features(np.asarray(p_acoustic, dtype=np.float32), inversions_b, ACOUSTIC_DIM_ALL)
        hcf_a = align_word_features(np.asarray(c_hcf, dtype=np.float32), inversions_a, args.hcf_dim)
        hcf_b = align_word_features(np.asarray(p_hcf, dtype=np.float32), inversions_b, args.hcf_dim)

        input_ids, visual, acoustic, hcf, input_mask, segment_ids = prepare_deberta_pair_input(
            tokens_a,
            tokens_b,
            visual_a,
            visual_b,
            acoustic_a,
            acoustic_b,
            hcf_a,
            hcf_b,
            tokenizer,
        )

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert visual.shape == (args.max_seq_length, VISUAL_DIM)
        assert acoustic.shape == (args.max_seq_length, ACOUSTIC_DIM)
        assert hcf.shape == (args.max_seq_length, args.hcf_dim)

        if use_silver:
            silver_record = get_silver_span_record(
                silver_split_records, ex_index, words_tgt, str(hid)
            )
            offset_b = 1 + len(tokens_a) + 1
            syntax_span_mask = build_pair_target_syntax_span_mask(
                silver_record, inversions_b, offset_b, args.max_seq_length
            )
        else:
            syntax_span_mask = np.zeros((args.max_seq_length, args.max_seq_length), dtype=np.float32)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                visual=visual,
                acoustic=acoustic,
                hcf=hcf,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=float(label),
                syntax_span_mask=syntax_span_mask,
            )
        )

    return features


def stack_features(features):
    return {
        "input_ids": np.array([f.input_ids for f in features], dtype=np.int64),
        "visual": np.array([f.visual for f in features], dtype=np.float32),
        "acoustic": np.array([f.acoustic for f in features], dtype=np.float32),
        "hcf": np.array([f.hcf for f in features], dtype=np.float32),
        "input_mask": np.array([f.input_mask for f in features], dtype=np.int64),
        "label_id": np.array([f.label_id for f in features], dtype=np.float32),
        "syntax_span_mask": np.array([f.syntax_span_mask for f in features], dtype=np.float32),
    }


def fit_zscore(stack):
    """Fit per-dim mean/std on non-padding rows only."""
    mask = stack["input_mask"].astype(bool).reshape(-1)

    def _fit(tensor, default_dim):
        flat = tensor.reshape(-1, default_dim)
        flat = flat[mask]
        if flat.shape[0] == 0:
            return np.zeros(default_dim, dtype=np.float32), np.ones(default_dim, dtype=np.float32)
        mean = flat.mean(axis=0).astype(np.float32)
        std = flat.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return mean, std

    return {
        "visual": _fit(stack["visual"], VISUAL_DIM),
        "acoustic": _fit(stack["acoustic"], ACOUSTIC_DIM),
        "hcf": _fit(stack["hcf"], args.hcf_dim),
    }


def apply_zscore(stack, scaler):
    stack = dict(stack)
    mask_float = stack["input_mask"].astype(np.float32)[..., None]
    for key in ("visual", "acoustic", "hcf"):
        mean, std = scaler[key]
        normalised = (stack[key] - mean) / std
        # Zero-out pad rows to keep the model's masking clean.
        stack[key] = (normalised * mask_float).astype(np.float32)
    return stack


def build_tensor_dataset(stack):
    return TensorDataset(
        torch.from_numpy(stack["input_ids"]).long(),
        torch.from_numpy(stack["visual"]).float(),
        torch.from_numpy(stack["acoustic"]).float(),
        torch.from_numpy(stack["hcf"]).float(),
        torch.from_numpy(stack["label_id"]).float(),
        torch.from_numpy(stack["syntax_span_mask"]).float(),
    )


def maybe_limit_examples(examples):
    if args.max_examples <= 0:
        return examples
    return list(examples)[: args.max_examples]


def set_up_data_loader():
    fold = None if args.fold < 0 else int(args.fold)
    split_payload, split_source = load_hkt_dataset(
        args.dataset,
        cache_path=args.dataset_cache,
        seed=args.seed,
        fold=fold,
        dev_ratio=args.dev_ratio,
    )
    if args.resolved_backbone == "albert":
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    else:
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.model)

    train_examples = maybe_limit_examples(split_payload["train"])
    dev_examples = maybe_limit_examples(split_payload["dev"])
    test_examples = maybe_limit_examples(split_payload["test"])
    if not train_examples:
        raise RuntimeError("Empty training split; aborting")
    if not test_examples:
        raise RuntimeError("Empty test split; aborting")

    print(
        "HKT_DATASET_SOURCE: {} | train={} dev={} test={}".format(
            split_source,
            len(train_examples),
            len(dev_examples),
            len(test_examples),
        ),
        flush=True,
    )
    print(
        "HKT_PROCESSING_STYLE: backbone={} feature_dim_mode={} skip_normalize={} primary_f1={} "
        "disable_hcf={} hcf_dim={} github_style={} hkt_paper_style={} "
        "dims=(acoustic={},visual={})".format(
            args.resolved_backbone,
            args.feature_dim_mode,
            bool(args.skip_normalize),
            args.primary_f1,
            bool(args.disable_hcf),
            args.hcf_dim,
            bool(args.github_style),
            bool(getattr(args, "hkt_paper_style", False)),
            ACOUSTIC_DIM,
            VISUAL_DIM,
        ),
        flush=True,
    )

    cache_path = resolve_hkt_silver_span_cache_path(args.dataset, args.silver_span_cache)
    silver_cache = (
        load_hkt_silver_span_cache(cache_path)
        if cache_path and os.path.isfile(cache_path)
        else None
    )
    if silver_cache is None and cache_path:
        print(f"HKT_SILVER_SPAN_CACHE: path not found, skipping: {cache_path}", flush=True)

    def _silver_for_split(split_name: str, examples_list):
        if silver_cache is None:
            return None
        records = silver_cache[split_name]
        if isinstance(records, list):
            return records[: len(examples_list)]
        return records

    train_features = convert_hkt_examples_to_features(
        train_examples,
        tokenizer,
        silver_split_records=_silver_for_split("train", train_examples),
    )
    dev_features = (
        convert_hkt_examples_to_features(
            dev_examples,
            tokenizer,
            silver_split_records=_silver_for_split("dev", dev_examples),
        )
        if dev_examples
        else []
    )
    test_features = convert_hkt_examples_to_features(
        test_examples,
        tokenizer,
        silver_split_records=_silver_for_split("test", test_examples),
    )

    train_stack = stack_features(train_features)
    if args.skip_normalize:
        scaler = None
        dev_stack = stack_features(dev_features) if dev_features else None
        test_stack = stack_features(test_features)
    else:
        scaler = fit_zscore(train_stack)
        train_stack = apply_zscore(train_stack, scaler)
        dev_stack = apply_zscore(stack_features(dev_features), scaler) if dev_features else None
        test_stack = apply_zscore(stack_features(test_features), scaler)

    train_dataset = build_tensor_dataset(train_stack)
    test_dataset = build_tensor_dataset(test_stack)
    dev_dataset = build_tensor_dataset(dev_stack) if dev_stack is not None else test_dataset

    steps_per_epoch = max(1, len(train_dataset) // max(args.train_batch_size, 1) // max(args.gradient_accumulation_step, 1))
    num_train_optimization_steps = steps_per_epoch * max(args.n_epochs, 1)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    meta = {
        "split_source": split_source,
        "sizes": {"train": len(train_dataset), "dev": len(dev_dataset), "test": len(test_dataset)},
        "scaler_is_dev_same_as_test": dev_stack is None,
    }
    return train_loader, dev_loader, test_loader, num_train_optimization_steps, meta


def prep_for_training(num_train_optimization_steps: int):
    if args.resolved_backbone == "albert":
        # Composite model: do not use PreTrainedModel.from_pretrained (key prefix mismatch); the
        # inner ITHP_AlbertModel already runs ``AlbertModel.from_pretrained`` in __init__.
        try:
            albert_cfg = AlbertConfig.from_pretrained(args.model, local_files_only=True)
        except OSError:
            albert_cfg = AlbertConfig.from_pretrained(args.model, local_files_only=False)
        model = ITHP_AlbertForBinaryClassification(albert_cfg, args)
    else:
        model = ITHP_DeBertaForBinaryClassification.from_pretrained(
            args.model,
            multimodal_config=args,
            num_labels=1,
        )
    model.to(DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [parameter for name, parameter in param_optimizer if not any(nd in name for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [parameter for name, parameter in param_optimizer if any(nd in name for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * num_train_optimization_steps),
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def compute_binary_metrics(probabilities, labels, threshold=0.5):
    """Binary predictions via ``probs >= threshold`` (threshold 0.5 = standard logistic cut)."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    predictions = (probs >= float(threshold)).astype(np.int64)
    f1_binary = float(f1_score(labels, predictions, average="binary", zero_division=0))
    f1_weighted = float(f1_score(labels, predictions, average="weighted", zero_division=0))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": f1_weighted if args.primary_f1 == "weighted" else f1_binary,
        "f1_binary": f1_binary,
        "f1_weighted": f1_weighted,
        "f1_hkt_paper": f1_binary,
    }


def tune_threshold_on_valid(dev_probs, dev_labels, objective: str, grid_size: int) -> tuple[float, dict]:
    """Pick T in [0.05, 0.95] maximizing objective on dev; ties broken by smaller T."""
    probs = np.asarray(dev_probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(dev_labels).reshape(-1).astype(np.int64)
    if probs.size == 0:
        print("THRESHOLD_TUNE: empty dev set; falling back to T=0.5", flush=True)
        return 0.5, compute_binary_metrics(probs, labels, threshold=0.5)

    if len(np.unique(probs)) <= 1:
        print("THRESHOLD_TUNE: constant dev probabilities; falling back to T=0.5", flush=True)
        return 0.5, compute_binary_metrics(probs, labels, threshold=0.5)

    lo, hi = 0.05, 0.95
    grid = np.unique(np.concatenate([np.linspace(lo, hi, max(3, int(grid_size))), [0.5]]))
    grid.sort()

    metric_key = "accuracy" if objective == "accuracy" else "f1"
    best_t = 0.5
    best_score = float("-inf")
    for t in grid:
        m = compute_binary_metrics(probs, labels, threshold=float(t))
        score = float(m[metric_key])
        if score > best_score + 1e-15:
            best_score = score
            best_t = float(t)
        elif abs(score - best_score) <= 1e-15 and float(t) < best_t:
            best_t = float(t)

    tuned = compute_binary_metrics(probs, labels, threshold=best_t)
    return best_t, tuned


def merge_tuned_classification_head(base_metrics: dict, tuned_cls: dict) -> dict:
    """Keep loss / recursion stats from ``base_metrics``; overwrite accuracy and F1 fields."""
    out = dict(base_metrics)
    for key in ("accuracy", "f1", "f1_binary", "f1_weighted", "f1_hkt_paper"):
        if key in tuned_cls:
            out[key] = tuned_cls[key]
    return out


def run_epoch(model, dataloader, loss_fct, optimizer=None, scheduler=None, desc="Eval", return_prob_arrays=False):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    recursion_steps_total = 0.0
    max_depth_hits = 0
    sample_count = 0
    probabilities = []
    labels = []

    iterator = tqdm(dataloader, desc=desc, leave=False)
    for step, batch in enumerate(iterator, start=1):
        batch = tuple(tensor.to(DEVICE, non_blocking=True) for tensor in batch)
        input_ids, visual, acoustic, hcf, label_ids, syntax_span_masks = batch

        with torch.set_grad_enabled(training):
            logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss = model(
                input_ids,
                visual,
                acoustic,
                hcf=None if args.disable_hcf else hcf,
                syntax_span_masks=syntax_span_masks,
            )
            classification_loss = loss_fct(logits.view(-1), label_ids.view(-1))
            loss = classification_loss + args.ib_loss_weight * IB_loss + args.syntax_loss_weight * syntax_loss

            if training and not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss encountered: cls={} ib={} syntax={}".format(
                        classification_loss.detach().item(),
                        IB_loss.detach().item(),
                        syntax_loss.detach().item(),
                    )
                )

            if training:
                scaled_loss = loss / max(1, args.gradient_accumulation_step)
                scaled_loss.backward()
                if step % args.gradient_accumulation_step == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

        total_loss += loss.detach().item()
        recursion_steps_total += recursive_steps.float().sum().item()
        max_depth_hits += (recursive_steps == global_configs.MAX_RECURSION_DEPTH).sum().item()
        sample_count += recursive_steps.size(0)
        probabilities.extend(torch.sigmoid(logits).view(-1).detach().cpu().numpy().tolist())
        labels.extend(label_ids.view(-1).detach().cpu().numpy().tolist())

    if training and len(dataloader) % args.gradient_accumulation_step != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    average_loss = total_loss / max(len(dataloader), 1)
    probs_arr = np.asarray(probabilities, dtype=np.float64)
    labels_arr = np.asarray(labels).astype(np.int64)
    metrics = compute_binary_metrics(probs_arr, labels_arr)
    metrics["avg_recursion_steps"] = recursion_steps_total / max(sample_count, 1)
    metrics["max_depth_hit_rate"] = max_depth_hits / max(sample_count, 1)
    metrics["loss"] = average_loss
    if return_prob_arrays:
        return metrics, probs_arr.copy(), labels_arr.copy()
    return metrics


def is_better_result(best_result, current_valid_metrics):
    if best_result is None:
        return True

    prev = best_result["valid"]
    if args.selection_metric == "valid_loss":
        return current_valid_metrics["loss"] < prev["loss"]
    return current_valid_metrics[args.selection_metric] > prev[args.selection_metric]


def resolve_run_dir():
    if args.run_name:
        return os.path.join(args.output_dir, args.run_name)
    name = args.dataset
    if args.fold >= 0:
        name = f"{args.dataset}/fold_{args.fold}"
    return os.path.join(args.output_dir, name)


def main():
    run_dir = resolve_run_dir()
    os.makedirs(run_dir, exist_ok=True)

    set_random_seed(args.seed)
    train_loader, dev_loader, test_loader, num_train_optimization_steps, meta = set_up_data_loader()
    model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)
    loss_fct = BCEWithLogitsLoss()

    if args.dry_run:
        batch = next(iter(train_loader))
        batch = tuple(tensor.to(DEVICE) for tensor in batch)
        input_ids, visual, acoustic, hcf, label_ids, syntax_span_masks = batch
        logits, *_ = model(
            input_ids,
            visual,
            acoustic,
            hcf=None if args.disable_hcf else hcf,
            syntax_span_masks=syntax_span_masks,
        )
        print(f"DRY_RUN: batch={input_ids.size(0)} logits_shape={tuple(logits.shape)}", flush=True)
        return

    best_result = None
    best_state_dict = None
    best_dev_probs = best_dev_labels = best_test_probs = best_test_labels = None
    tune_mode = args.decision_threshold_mode == "tune_on_valid"
    stale_epochs = 0
    started_at = time.time()

    for epoch_index in range(args.n_epochs):
        train_metrics = run_epoch(model, train_loader, loss_fct, optimizer=optimizer, scheduler=scheduler, desc=f"Train[{epoch_index}]")
        if tune_mode:
            valid_metrics, dev_p, dev_l = run_epoch(
                model, dev_loader, loss_fct, desc=f"Dev[{epoch_index}]", return_prob_arrays=True
            )
            test_metrics, test_p, test_l = run_epoch(
                model, test_loader, loss_fct, desc=f"Test[{epoch_index}]", return_prob_arrays=True
            )
        else:
            valid_metrics = run_epoch(model, dev_loader, loss_fct, desc=f"Dev[{epoch_index}]")
            test_metrics = run_epoch(model, test_loader, loss_fct, desc=f"Test[{epoch_index}]")

        print(
            "epoch:{} train_loss:{:.6f} valid_loss:{:.6f} valid_acc:{:.4f} valid_f1:{:.4f} test_acc:{:.4f} test_f1:{:.4f} avg_steps:{:.2f}".format(
                epoch_index,
                train_metrics["loss"],
                valid_metrics["loss"],
                valid_metrics["accuracy"],
                valid_metrics["f1"],
                test_metrics["accuracy"],
                test_metrics["f1"],
                train_metrics["avg_recursion_steps"],
            ),
            flush=True,
        )

        if is_better_result(best_result, valid_metrics):
            best_result = {
                "epoch": int(epoch_index),
                "train": train_metrics,
                "valid": valid_metrics,
                "test": test_metrics,
            }
            if tune_mode:
                best_dev_probs, best_dev_labels = dev_p, dev_l
                best_test_probs, best_test_labels = test_p, test_l
            if args.save_weight:
                best_state_dict = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if args.early_stopping_patience and stale_epochs >= args.early_stopping_patience:
            print(f"EARLY_STOPPING at epoch {epoch_index}", flush=True)
            break

    if best_result is None:
        raise RuntimeError("Training finished without a valid epoch result")

    elapsed = time.time() - started_at

    if best_state_dict is not None and args.save_weight:
        model.load_state_dict(best_state_dict)
        save_path = args.save_weight if os.path.isabs(args.save_weight) else os.path.join(run_dir, args.save_weight)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved best checkpoint to {save_path}", flush=True)

    print(
        "BEST_RESULT: epoch={} valid_loss={:.6f} valid_acc={:.4f} valid_f1={:.4f} test_acc={:.4f} test_f1={:.4f} selection={}".format(
            best_result["epoch"],
            best_result["valid"]["loss"],
            best_result["valid"]["accuracy"],
            best_result["valid"]["f1"],
            best_result["test"]["accuracy"],
            best_result["test"]["f1"],
            args.selection_metric,
        ),
        flush=True,
    )

    threshold_tuning = None
    if tune_mode:
        if best_dev_probs is None or best_test_probs is None:
            raise RuntimeError("tune_on_valid enabled but dev/test probability caches are missing")
        t_star, _dev_tuned_cls = tune_threshold_on_valid(
            best_dev_probs,
            best_dev_labels,
            args.threshold_tune_objective,
            args.threshold_grid_size,
        )
        dev_tuned_cls = compute_binary_metrics(best_dev_probs, best_dev_labels, threshold=t_star)
        test_tuned_cls = compute_binary_metrics(best_test_probs, best_test_labels, threshold=t_star)
        threshold_tuning = {
            "mode": "tune_on_valid",
            "objective": args.threshold_tune_objective,
            "t_star": float(t_star),
            "grid_size": int(args.threshold_grid_size),
            "valid": merge_tuned_classification_head(best_result["valid"], dev_tuned_cls),
            "test": merge_tuned_classification_head(best_result["test"], test_tuned_cls),
        }
        print(
            "THRESHOLD_TUNING: T_star={:.4f} objective={} valid_acc_t={:.4f} valid_f1_t={:.4f} test_acc_t={:.4f} test_f1_t={:.4f}".format(
                t_star,
                args.threshold_tune_objective,
                threshold_tuning["valid"]["accuracy"],
                threshold_tuning["valid"]["f1"],
                threshold_tuning["test"]["accuracy"],
                threshold_tuning["test"]["f1"],
            ),
            flush=True,
        )

    cli_snapshot = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (int, float, str, bool, list))
    }
    result_payload = {
        "dataset": args.dataset,
        "fold": int(args.fold) if args.fold >= 0 else None,
        "selection_metric": args.selection_metric,
        "backbone": args.resolved_backbone,
        "hkt_paper_style": bool(getattr(args, "hkt_paper_style", False)),
        "metric_definitions": {
            "best_block": (
                "Epoch chosen by --selection_metric on dev using a fixed 0.5 decision threshold "
                "(probability >= 0.5); train/valid/test metrics at 0.5."
            ),
            "f1_hkt_paper": "sklearn f1 average=binary at the stated decision threshold.",
            "accuracy": "Fraction correct at the stated decision threshold (0.5 for 'best').",
            "threshold_tuning_block": (
                "When key 'threshold_tuning' is present: T_star chosen on dev to maximize "
                "--threshold_tune_objective over a grid in [0.05, 0.95]; valid/test reuse loss and "
                "recursion fields from 'best' but replace accuracy/F1 with metrics at T_star. "
                "Dev-based threshold selection can look optimistic on test vs strict 0.5 / paper protocols."
            ),
        },
        "elapsed_seconds": elapsed,
        "data": meta,
        "best": best_result,
        "args": cli_snapshot,
    }
    if threshold_tuning is not None:
        result_payload["threshold_tuning"] = threshold_tuning
    result_path = os.path.join(run_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
