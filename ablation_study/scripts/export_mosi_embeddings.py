#!/usr/bin/env python3
"""
Export pooled backbone embeddings (pre-head) from a fixed_training checkpoint.

Requires a checkpoint saved with ``--save_checkpoint`` from ``fixed_training.train``,
which stores ``{"model": state_dict, "args": vars(args)}``.

Run from anywhere; the script adds ``fixed_experiment`` to ``sys.path`` and chdirs
to ``recursive_ITHP`` via ``configure_runtime`` (same as training).

Example::

    cd /path/to/recursive_ITHP/fixed_experiment
    python ../ablation_study/scripts/export_mosi_embeddings.py \\
        --checkpoint ../ablation_study/checkpoints/mosi_full_t278.pt \\
        --output ../ablation_study/figures/embeddings_full.npz \\
        --variant mosi_full_t278
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_ABLATION_ROOT = _SCRIPT_DIR.parent
_RECURSIVE_ITHP = _ABLATION_ROOT.parent
_FIXED_EXP = _RECURSIVE_ITHP / "fixed_experiment"


def _ensure_import_path() -> None:
    fe = str(_FIXED_EXP.resolve())
    if fe not in sys.path:
        sys.path.insert(0, fe)


def _backbone_forward(model, input_ids, visual_norm, acoustic_norm, syntax_span_masks):
    """First tensor from backbone (pooled), before dropout + classifier."""
    if hasattr(model, "dberta"):
        out = model.dberta(
            input_ids,
            visual_norm,
            acoustic_norm,
            syntax_span_masks=syntax_span_masks,
            return_syntax_info=False,
        )
        return out[0]
    if hasattr(model, "bert"):
        out = model.bert(
            input_ids,
            visual_norm,
            acoustic_norm,
            syntax_span_masks=syntax_span_masks,
            return_syntax_info=False,
        )
        return out[0]
    raise ValueError("Model has neither .dberta nor .bert; unsupported for embedding export.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to .pt from --save_checkpoint")
    ap.add_argument("--output", type=str, required=True, help="Output .npz path (embeddings + labels)")
    ap.add_argument("--variant", type=str, default="", help="Label stored in sidecar JSON for plotting")
    args_cli = ap.parse_args()

    ckpt_path = Path(args_cli.checkpoint).resolve()
    out_path = Path(args_cli.output).resolve()
    variant = args_cli.variant or ckpt_path.stem

    _ensure_import_path()
    from fixed_training.train import (  # noqa: E402
        configure_runtime,
        prep_for_training,
        safe_min_max_normalize,
        set_random_seed,
        set_up_data_loader,
    )
    from fixed_training.global_configs import DEVICE  # noqa: E402

    bundle = torch.load(ckpt_path, map_location="cpu")
    if "model" not in bundle or "args" not in bundle:
        raise KeyError("Checkpoint must contain 'model' and 'args' keys (use train.py --save_checkpoint).")

    ns = argparse.Namespace(**bundle["args"])
    configure_runtime(ns)
    set_random_seed(ns.seed)

    train_dl, dev_dl, test_dl, n_steps = set_up_data_loader(test_shuffle=False)
    model, _, _ = prep_for_training(n_steps)
    model.load_state_dict(bundle["model"])
    model.to(DEVICE)
    model.eval()

    emb_chunks: list[np.ndarray] = []
    lab_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="export_embeddings"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids, syntax_span_masks = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            visual_norm = safe_min_max_normalize(visual)
            acoustic_norm = safe_min_max_normalize(acoustic)

            pooled = _backbone_forward(model, input_ids, visual_norm, acoustic_norm, syntax_span_masks)
            emb_chunks.append(pooled.detach().float().cpu().numpy())
            lab_chunks.append(label_ids.detach().float().cpu().numpy().reshape(-1))

    embeddings = np.concatenate(emb_chunks, axis=0)
    labels = np.concatenate(lab_chunks, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, embeddings=embeddings, labels=labels)

    meta = {
        "variant": variant,
        "checkpoint": str(ckpt_path),
        "dataset": ns.dataset,
        "n_samples": int(embeddings.shape[0]),
        "hidden_size": int(embeddings.shape[1]),
        "npz": str(out_path),
    }
    meta_path = out_path.with_name(out_path.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} shape={embeddings.shape} meta={meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
