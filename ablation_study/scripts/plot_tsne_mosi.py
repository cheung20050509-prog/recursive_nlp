#!/usr/bin/env python3
"""
t-SNE plots from ``export_mosi_embeddings.py`` outputs (.npz).

**Color modes**

- ``bins7`` (default): continuous MOSI-style score → 7 bins on [-3, 3], colormap + colorbar.
- ``trinary``: ordinal-style **negative / neutral / positive** (thresholds on raw labels),
  distinct **markers** (+ / o / x) and colors similar to common sentiment figures.

Optional ``--combined``: stacked embeddings, one TSNE, hue=variant (extra PNG).

**Ablation visibility:** use ``--layout joint`` — one t-SNE fit on **vertically stacked**
embeddings from all models (same test order, equal ``N``). Each subplot shows one model’s
``N`` points in the **same 2D coordinate system** (shared x/y limits), so clouds can be
compared directly. Per-model ``--layout separate`` (default) is **not** cross-comparable.

Example (ordinal-style compare, two models)::

    python plot_tsne_mosi.py \\
        figures/embeddings_baseline.npz figures/embeddings_ordinal.npz \\
        --out figures/compare_ordinal.png \\
        --color-mode trinary --panel-tags "(a),(b)"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

DEFAULT_SENTIMENT_EDGES = np.linspace(-3.0, 3.0, 8)  # 7 bins for MOSI-style scores

# Reference-style trinary (negative / neutral / positive)
_TRINARY_SPECS: list[tuple[int, str, str, str, int]] = [
    (0, "negative", "#17becf", "+", 52),   # teal-ish
    (1, "neutral", "#1f77b4", "o", 22),   # blue dot
    (2, "positive", "#9467bd", "x", 44),  # purple x
]


def _load_variant(npz_path: Path) -> str:
    meta = npz_path.with_name(npz_path.stem + ".meta.json")
    if meta.is_file():
        try:
            return str(json.loads(meta.read_text(encoding="utf-8")).get("variant", npz_path.stem))
        except (json.JSONDecodeError, OSError):
            pass
    return npz_path.stem


def _sentiment_bin_colors(labels: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each label to a bin index in ``0 .. len(edges)-2``."""
    clipped = np.clip(labels.astype(np.float64), edges[0], edges[-1])
    idx = np.searchsorted(edges, clipped, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return idx, centers


def _trinary_class(labels: np.ndarray, neg_th: float, pos_th: float) -> np.ndarray:
    """0 = negative, 1 = neutral, 2 = positive (ordinal on continuous MOSI label)."""
    y = labels.astype(np.float64).reshape(-1)
    out = np.ones(len(y), dtype=np.int8)
    out[y < neg_th] = 0
    out[y > pos_th] = 2
    return out


def _run_tsne(emb: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    n_samples = emb.shape[0]
    perp = min(perplexity, max(5, n_samples - 1))
    z = StandardScaler().fit_transform(emb)
    return TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    ).fit_transform(z)


def _scatter_trinary(
    ax,
    xy: np.ndarray,
    labels: np.ndarray,
    neg_th: float,
    pos_th: float,
    *,
    show_legend: bool = True,
) -> None:
    cls = _trinary_class(labels, neg_th, pos_th)
    for code, name, color, marker, ms in _TRINARY_SPECS:
        m = cls == code
        if not np.any(m):
            continue
        lbl = name if show_legend else None
        if marker == "o":
            kw = {
                "c": color,
                "marker": marker,
                "s": ms,
                "alpha": 0.78,
                "label": lbl,
                "edgecolors": "k",
                "linewidths": 0.2,
            }
        else:
            kw = {
                "c": color,
                "marker": marker,
                "s": ms,
                "alpha": 0.78,
                "label": lbl,
                "linewidths": 0.65,
            }
        ax.scatter(xy[m, 0], xy[m, 1], **kw)
    if show_legend:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "npz_files",
        nargs="+",
        type=str,
        help="One or more .npz files from export_mosi_embeddings.py",
    )
    ap.add_argument("--out", type=str, required=True, help="Output PNG path")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (capped vs n)")
    ap.add_argument("--seed", type=int, default=42, help="t-SNE random_state")
    ap.add_argument(
        "--color-mode",
        choices=("bins7", "trinary"),
        default="bins7",
        help="bins7: 7-bin colormap (default). trinary: neg/neu/pos with +/o/x markers.",
    )
    ap.add_argument(
        "--trinary-neg-th",
        type=float,
        default=-0.5,
        help="Labels < this → negative (trinary mode).",
    )
    ap.add_argument(
        "--trinary-pos-th",
        type=float,
        default=0.5,
        help="Labels > this → positive (trinary mode).",
    )
    ap.add_argument(
        "--panel-tags",
        type=str,
        default="",
        help='Comma-separated tags prepended to titles, e.g. \'(a),(b)\' for first two panels.',
    )
    ap.add_argument(
        "--layout",
        choices=("separate", "joint"),
        default="separate",
        help="separate: one TSNE per model (not comparable across panels). "
        "joint: one TSNE on stacked rows, split into panels with shared x/y limits (recommended for ablation).",
    )
    ap.add_argument(
        "--combined",
        action="store_true",
        help="Second figure: stacked embeddings, one TSNE, hue=variant (requires equal n per file)",
    )
    ap.add_argument(
        "--edges",
        type=str,
        default="",
        help="Optional comma-separated bin edges for bins7 coloring (default MOSI 7 bins in [-3,3])",
    )
    args = ap.parse_args()

    paths = [Path(p).resolve() for p in args.npz_files]
    bundles: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in paths:
        data = np.load(p)
        bundles.append((_load_variant(p), data["embeddings"], data["labels"]))

    tags: list[str] = []
    if args.panel_tags.strip():
        tags = [t.strip() for t in args.panel_tags.split(",")]
        if len(tags) != len(bundles):
            raise ValueError(f"--panel-tags: expected {len(bundles)} comma-separated tags, got {len(tags)}")

    edges = DEFAULT_SENTIMENT_EDGES
    if args.edges.strip():
        edges = np.array([float(x) for x in args.edges.split(",")], dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 3:
            raise ValueError("--edges must be at least 3 comma-separated floats")

    n = len(bundles)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.layout == "joint":
        sizes = [b[1].shape[0] for b in bundles]
        if len(set(sizes)) != 1:
            raise ValueError("--layout joint requires the same N in every .npz (same test order).")
        n_per = sizes[0]
        names, mats, labs = zip(*bundles)
        X = np.vstack(mats)
        z = StandardScaler().fit_transform(X)
        n_all = z.shape[0]
        perp = min(args.perplexity, max(5, n_all - 1))
        xy_all = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate="auto",
            init="pca",
            random_state=args.seed,
        ).fit_transform(z)

        ncols = 2 if n > 1 else 1
        nrows = (n + ncols - 1) // ncols
        fig_w = 4.8 * ncols if args.color_mode == "trinary" else 4.2 * ncols
        fig_h = 4.2 * nrows if args.color_mode == "trinary" else 3.8 * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        flat_ax = axes.ravel()

        all_xy = []
        for j, (name, _emb, lab) in enumerate(bundles):
            sl = slice(j * n_per, (j + 1) * n_per)
            xy_j = xy_all[sl]
            all_xy.append(xy_j)
            tag = tags[j] if tags else ""
            title = f"{tag} {name}".strip() if tag else name
            ax = flat_ax[j]
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            show_leg = j == 0
            if args.color_mode == "trinary":
                _scatter_trinary(ax, xy_j, lab, args.trinary_neg_th, args.trinary_pos_th, show_legend=show_leg)
            else:
                bin_idx, _centers = _sentiment_bin_colors(lab, edges)
                sc = ax.scatter(xy_j[:, 0], xy_j[:, 1], c=bin_idx, cmap="viridis", s=12, alpha=0.85)
                if j == len(bundles) - 1:
                    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="sentiment bin")

        for j in range(len(bundles), len(flat_ax)):
            flat_ax[j].set_visible(False)

        stacked = np.vstack(all_xy)
        pad = 0.05 * max(stacked[:, 0].ptp(), stacked[:, 1].ptp(), 1e-6)
        x0, x1 = stacked[:, 0].min() - pad, stacked[:, 0].max() + pad
        y0, y1 = stacked[:, 1].min() - pad, stacked[:, 1].max() + pad
        for ax in flat_ax[: len(bundles)]:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

        fig.suptitle("Joint t-SNE (shared 2D space; comparable across panels)", fontsize=10, y=1.02)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote joint-layout figure: {out_path}")
    else:
        ncols = 2 if n > 1 else 1
        nrows = (n + ncols - 1) // ncols
        fig_w = 4.8 * ncols if args.color_mode == "trinary" else 4.2 * ncols
        fig_h = 4.2 * nrows if args.color_mode == "trinary" else 3.8 * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        flat_ax = axes.ravel()

        for ax, (name, emb, lab), tag in zip(
            flat_ax,
            bundles,
            tags if tags else [""] * len(bundles),
        ):
            xy = _run_tsne(emb, args.perplexity, args.seed)
            title = f"{tag} {name}".strip() if tag else name
            ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

            if args.color_mode == "trinary":
                _scatter_trinary(ax, xy, lab, args.trinary_neg_th, args.trinary_pos_th)
            else:
                bin_idx, _centers = _sentiment_bin_colors(lab, edges)
                sc = ax.scatter(xy[:, 0], xy[:, 1], c=bin_idx, cmap="viridis", s=12, alpha=0.85)
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="sentiment bin")

        for j in range(len(bundles), len(flat_ax)):
            flat_ax[j].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Wrote grid figure: {out_path}")

    if args.combined:
        sizes = [b[1].shape[0] for b in bundles]
        if len(set(sizes)) != 1:
            raise ValueError("--combined requires the same number of rows in every .npz (same test order).")
        names, mats, _labs = zip(*bundles)
        X = np.vstack(mats)
        y_name = np.repeat(np.array(names), sizes[0])
        z = StandardScaler().fit_transform(X)
        n_samples = z.shape[0]
        perp = min(args.perplexity, max(5, n_samples - 1))
        xy = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate="auto",
            init="pca",
            random_state=args.seed,
        ).fit_transform(z)

        fig2, ax2 = plt.subplots(figsize=(6.5, 5.0))
        uniq = list(dict.fromkeys(names))
        cmap = plt.get_cmap("tab10")
        name_to_color = {u: cmap(i % 10) for i, u in enumerate(uniq)}
        for u in uniq:
            m = y_name == u
            ax2.scatter(
                xy[m, 0],
                xy[m, 1],
                s=10,
                alpha=0.75,
                label=u,
                color=name_to_color[u],
            )
        ax2.legend(markerscale=2, fontsize=8, loc="best")
        ax2.set_title("Combined t-SNE (stacked embeddings)")
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig2.tight_layout()
        comb_path = out_path.with_name(out_path.stem + "_combined" + out_path.suffix)
        fig2.savefig(comb_path, dpi=160)
        plt.close(fig2)
        print(f"Wrote combined figure: {comb_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
