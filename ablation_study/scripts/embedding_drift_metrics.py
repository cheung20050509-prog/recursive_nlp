#!/usr/bin/env python3
"""
Quantify how much each ablation moves test-set embeddings vs a reference (.npz).

Reads ``export_mosi_embeddings.py`` outputs (same row order, same N).

Prints per variant:
  - mean / std L2 distance to reference per sample
  - mean cosine distance (1 - cosine similarity)
  - (optional) sklearn silhouette score for trinary sentiment classes on raw embeddings

Example::

    python embedding_drift_metrics.py \\
        figures/embeddings_mosi_full_t278.npz \\
        figures/embeddings_mosi_A1_no_syntax_sup.npz \\
        figures/embeddings_mosi_A3_no_recursion_refine.npz \\
        figures/embeddings_mosi_A4_ib_no_stage2.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import silhouette_score
except ImportError:
    silhouette_score = None  # type: ignore


def _trinary(y: np.ndarray, neg_th: float, pos_th: float) -> np.ndarray:
    y = y.astype(np.float64).reshape(-1)
    out = np.ones(len(y), dtype=np.int64)
    out[y < neg_th] = 0
    out[y > pos_th] = 2
    return out


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b_n = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    cos = (a * b).sum(axis=1) / (a_n.ravel() * b_n.ravel())
    cos = np.clip(cos, -1.0, 1.0)
    return 1.0 - cos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz_files", nargs="+", help="Reference first, then ablations (same N, same order)")
    ap.add_argument("--trinary-neg-th", type=float, default=-0.5)
    ap.add_argument("--trinary-pos-th", type=float, default=0.5)
    ap.add_argument("--json-out", type=str, default="", help="Optional path to write metrics JSON")
    args = ap.parse_args()

    paths = [Path(p).resolve() for p in args.npz_files]
    bundles: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in paths:
        data = np.load(p)
        meta = p.with_name(p.stem + ".meta.json")
        name = p.stem
        if meta.is_file():
            try:
                name = str(json.loads(meta.read_text(encoding="utf-8")).get("variant", name))
            except (json.JSONDecodeError, OSError):
                pass
        bundles.append((name, data["embeddings"].astype(np.float64), data["labels"].astype(np.float64)))

    ref_name, ref_e, ref_y = bundles[0]
    n0 = ref_e.shape[0]
    rows = []

    for name, emb, y in bundles:
        if emb.shape[0] != n0:
            print(f"error: row count mismatch {name} has {emb.shape[0]}, ref has {n0}", file=sys.stderr)
            return 2
        if not np.allclose(y, ref_y):
            print(f"warning: labels differ from reference for {name} (expected identical test order)", file=sys.stderr)

    for name, emb, y in bundles:
        tri = _trinary(y, args.trinary_neg_th, args.trinary_pos_th)
        sil = None
        if silhouette_score is not None and len(np.unique(tri)) >= 2:
            try:
                sil = float(silhouette_score(emb, tri, metric="euclidean"))
            except ValueError:
                sil = None

        if name == ref_name:
            rows.append(
                {
                    "variant": name,
                    "role": "reference",
                    "l2_to_ref_mean": 0.0,
                    "l2_to_ref_std": 0.0,
                    "cosdist_to_ref_mean": 0.0,
                    "silhouette_trinary": sil,
                }
            )
            continue

        d_l2 = np.linalg.norm(emb - ref_e, axis=1)
        d_cos = _cosine_dist(emb, ref_e)
        rows.append(
            {
                "variant": name,
                "role": "ablation",
                "l2_to_ref_mean": float(d_l2.mean()),
                "l2_to_ref_std": float(d_l2.std()),
                "cosdist_to_ref_mean": float(d_cos.mean()),
                "silhouette_trinary": sil,
            }
        )

    print(f"reference: {ref_name}  N={n0}  dim={ref_e.shape[1]}")
    print(f"trinary thresholds: neg<{args.trinary_neg_th}, pos>{args.trinary_pos_th}")
    print()
    for r in rows:
        sil_s = f"{r['silhouette_trinary']:.4f}" if r["silhouette_trinary"] is not None else "n/a"
        print(
            f"{r['variant']}: L2_to_ref mean={r['l2_to_ref_mean']:.4f} std={r['l2_to_ref_std']:.4f} | "
            f"cosdist_to_ref mean={r['cosdist_to_ref_mean']:.4f} | silhouette(trinary)={sil_s}"
        )

    if args.json_out:
        out_p = Path(args.json_out).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps({"reference": ref_name, "metrics": rows}, indent=2), encoding="utf-8")
        print(f"\nWrote {out_p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
