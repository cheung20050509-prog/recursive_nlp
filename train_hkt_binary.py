"""HKT-style binary classification trainer for MUStARD and UR-FUNNY.

The pipeline is a ground-up rewrite over the legacy ``sibling`` version; it
fixes several real bugs:

- HCF (4-dim Humor-Centric Features) now flow through ``ITHP_DeBerta`` instead
  of being silently dropped before the tensor dataset.
- Per-dimension z-score normalisation for visual/acoustic (and HCF) fit on
  train only, then applied to dev/test; replaces the old batch-wise min-max.
- F1 metric switched to ``average="binary"`` (HKT-style).
- Model selection defaults to ``accuracy`` (``--selection_metric``).
- CLI exposes ``--fold`` for MUStARD speaker-independent k-fold CV and writes
  a structured ``result.json`` per run under ``--output_dir``.

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
from transformers import DebertaV2Tokenizer, get_linear_schedule_with_warmup

from deberta_ITHP import ITHP_DeBertaForBinaryClassification
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
    parser.add_argument("--syntax_loss_weight", default=0.0, type=float)
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
    parser.add_argument("--output_dir", type=str, default="log/4080_restart_hkt")
    parser.add_argument("--run_name", type=str, default="",
                        help="Optional subdirectory under output_dir; defaults to <dataset>[_fold_k].")
    args = parser.parse_args()

    if args.github_style:
        args.feature_dim_mode = "full"
        args.skip_normalize = True
        args.primary_f1 = "weighted"
        args.disable_hcf = True

    args.dataset = normalize_dataset_name(args.dataset)
    if args.max_seq_length <= 0:
        args.max_seq_length = HKT_DEFAULT_MAX_SEQ_LENGTH[args.dataset]
    if args.max_recursion_depth < 1:
        raise ValueError("max_recursion_depth must be at least 1")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if args.ib_loss_weight < 0:
        args.ib_loss_weight = 2.0 / (args.p_beta + args.p_gamma)
    if args.hcf_dim < 1:
        raise ValueError("hcf_dim must be >= 1")
    if args.fold >= 0 and args.dataset != "mustard":
        raise ValueError("--fold is only meaningful for dataset=mustard")
    return args


args = parse_args()

global_configs.MAX_RECURSION_DEPTH = args.max_recursion_depth
dataset_config_key = args.dataset if args.feature_dim_mode == "subset" else f"{args.dataset}_full"
global_configs.set_dataset_config(dataset_config_key)
ACOUSTIC_DIM, VISUAL_DIM = global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM


class InputFeatures(object):
    def __init__(self, input_ids, visual, acoustic, hcf, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.hcf = hcf
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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


def convert_hkt_examples_to_features(examples, tokenizer):
    features = []

    for example in examples:
        (
            (p_words, p_visual, p_acoustic, p_hcf),
            (c_words, c_visual, c_acoustic, c_hcf),
            hid,
            label,
        ) = example

        text_a = ". ".join(c_words) if isinstance(c_words, (list, tuple)) else str(c_words)
        text_b = (p_words + ".") if isinstance(p_words, str) else (" ".join(p_words) + ".")
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)

        inversions_a = get_inversion(tokens_a)
        inversions_b = get_inversion(tokens_b)
        pop_count = _truncate_context_prefix(tokens_a, tokens_b, args.max_seq_length - 3)
        # Pop the front of inversions_a so it aligns with the shorter token_a.
        inversions_a = inversions_a[pop_count:]
        inversions_b = inversions_b[:len(tokens_b)]

        visual_a = align_word_features(np.asarray(c_visual, dtype=np.float32), inversions_a, VISUAL_DIM_ALL)
        visual_b = align_word_features(np.asarray(p_visual, dtype=np.float32), inversions_b, VISUAL_DIM_ALL)
        acoustic_a = align_word_features(np.asarray(c_acoustic, dtype=np.float32), inversions_a, ACOUSTIC_DIM_ALL)
        acoustic_b = align_word_features(np.asarray(p_acoustic, dtype=np.float32), inversions_b, ACOUSTIC_DIM_ALL)
        hcf_a = align_word_features(np.asarray(c_hcf, dtype=np.float32), inversions_a, args.hcf_dim)
        hcf_b = align_word_features(np.asarray(p_hcf, dtype=np.float32), inversions_b, args.hcf_dim)

        input_ids, visual, acoustic, hcf, input_mask, segment_ids = prepare_deberta_pair_input(
            tokens_a, tokens_b,
            visual_a, visual_b,
            acoustic_a, acoustic_b,
            hcf_a, hcf_b,
            tokenizer,
        )

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert visual.shape == (args.max_seq_length, VISUAL_DIM)
        assert acoustic.shape == (args.max_seq_length, ACOUSTIC_DIM)
        assert hcf.shape == (args.max_seq_length, args.hcf_dim)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                visual=visual,
                acoustic=acoustic,
                hcf=hcf,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=float(label),
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
        "HKT_PROCESSING_STYLE: feature_dim_mode={} skip_normalize={} primary_f1={} "
        "disable_hcf={} hcf_dim={} github_style={} dims=(acoustic={},visual={})".format(
            args.feature_dim_mode,
            bool(args.skip_normalize),
            args.primary_f1,
            bool(args.disable_hcf),
            args.hcf_dim,
            bool(args.github_style),
            ACOUSTIC_DIM,
            VISUAL_DIM,
        ),
        flush=True,
    )

    train_features = convert_hkt_examples_to_features(train_examples, tokenizer)
    dev_features = convert_hkt_examples_to_features(dev_examples, tokenizer) if dev_examples else []
    test_features = convert_hkt_examples_to_features(test_examples, tokenizer)

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


def compute_binary_metrics(probabilities, labels):
    predictions = probabilities.round()
    f1_binary = float(f1_score(labels, predictions, average="binary", zero_division=0))
    f1_weighted = float(f1_score(labels, predictions, average="weighted", zero_division=0))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": f1_weighted if args.primary_f1 == "weighted" else f1_binary,
        "f1_binary": f1_binary,
        "f1_weighted": f1_weighted,
    }


def run_epoch(model, dataloader, loss_fct, optimizer=None, scheduler=None, desc="Eval"):
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
        input_ids, visual, acoustic, hcf, label_ids = batch

        with torch.set_grad_enabled(training):
            logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss = model(
                input_ids,
                visual,
                acoustic,
                hcf=None if args.disable_hcf else hcf,
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
    metrics = compute_binary_metrics(np.array(probabilities), np.array(labels))
    metrics["avg_recursion_steps"] = recursion_steps_total / max(sample_count, 1)
    metrics["max_depth_hit_rate"] = max_depth_hits / max(sample_count, 1)
    metrics["loss"] = average_loss
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
        input_ids, visual, acoustic, hcf, label_ids = batch
        logits, *_ = model(input_ids, visual, acoustic, hcf=None if args.disable_hcf else hcf)
        print(f"DRY_RUN: batch={input_ids.size(0)} logits_shape={tuple(logits.shape)}", flush=True)
        return

    best_result = None
    best_state_dict = None
    stale_epochs = 0
    started_at = time.time()

    for epoch_index in range(args.n_epochs):
        train_metrics = run_epoch(model, train_loader, loss_fct, optimizer=optimizer, scheduler=scheduler, desc=f"Train[{epoch_index}]")
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

    cli_snapshot = {
        key: value for key, value in vars(args).items() if isinstance(value, (int, float, str, bool, list))
    }
    result_payload = {
        "dataset": args.dataset,
        "fold": int(args.fold) if args.fold >= 0 else None,
        "selection_metric": args.selection_metric,
        "elapsed_seconds": elapsed,
        "data": meta,
        "best": best_result,
        "args": cli_snapshot,
    }
    result_path = os.path.join(run_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
