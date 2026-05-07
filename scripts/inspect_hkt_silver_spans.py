#!/usr/bin/env python3
"""MUStARD / UR-FUNNY (HKT) silver span cache QA: meta, stratified spot-check, struct metrics.

Same record schema as ``build_silver_span_cache.py`` / ``build_hkt_silver_span_cache.py``:
``words``, ``word_spans``, ``parse_tree``, ``parse_error``, ``segment`` (sample id).

English utterances: span phrases are joined with spaces for readability.

Example::

    cd ITHP/recursive_ITHP
    python scripts/inspect_hkt_silver_spans.py \\
        --pickle datasets/mustard_silver_spans.pkl \\
        --n-per-split 30 --seed 11 \\
        --out-meta log/mustard_silver_meta.json \\
        --out-spotcheck log/mustard_silver_manual_spotcheck.txt \\
        --out-struct log/mustard_silver_struct_metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import statistics
from pathlib import Path


def _span_phrases_english(words: list, word_spans: list) -> list[str]:
    out = []
    n = len(words)
    for span in word_spans or []:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        a, b = int(span[0]), int(span[1])
        if a < 0 or b > n or a >= b:
            continue
        piece = [str(w) for w in words[a:b]]
        out.append(" ".join(piece))
    return out


def _struct_one_split(records: list) -> dict:
    span_lens: list[int] = []
    cover_counts: list[float] = []
    n_words_list: list[int] = []

    for rec in records:
        words = list(rec.get("words") or [])
        n_words_list.append(len(words))
        spans = rec.get("word_spans") or []
        covered: set[int] = set()
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            a, b = int(span[0]), int(span[1])
            span_lens.append(b - a)
            for i in range(a, b):
                covered.add(i)
        lw = len(words)
        cover_counts.append(len(covered) / lw if lw else 0.0)

    def pct(xs: list[int], p: float) -> float:
        if not xs:
            return 0.0
        xs_sorted = sorted(xs)
        k = min(len(xs_sorted) - 1, max(0, int(round((len(xs_sorted) - 1) * p))))
        return round(xs_sorted[k], 6)

    return {
        "n_records": len(records),
        "span_length_words": {
            "mean": round(statistics.mean(span_lens), 4) if span_lens else 0.0,
            "median": round(statistics.median(span_lens), 4) if span_lens else 0.0,
            "p25": pct(span_lens, 0.25) if span_lens else 0.0,
            "p75": pct(span_lens, 0.75) if span_lens else 0.0,
            "max": max(span_lens) if span_lens else 0,
            "count_spans": len(span_lens),
        },
        "word_coverage_by_span_union": {
            "mean": round(statistics.mean(cover_counts), 4) if cover_counts else 0.0,
            "median": round(statistics.median(cover_counts), 4) if cover_counts else 0.0,
        },
        "sentence_word_count": {
            "mean": round(statistics.mean(n_words_list), 4) if n_words_list else 0.0,
            "median": round(statistics.median(n_words_list), 4) if n_words_list else 0.0,
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="Inspect HKT (mustard/urfunny) silver span cache.")
    p.add_argument("--pickle", required=True, help="e.g. datasets/mustard_silver_spans.pkl")
    p.add_argument("--n-per-split", type=int, default=30)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--parse-tree-chars", type=int, default=220)
    p.add_argument("--out-meta", default="", help="JSON path under repo root (optional).")
    p.add_argument("--out-spotcheck", default="", help="Text report path under repo root (optional).")
    p.add_argument("--out-struct", default="", help="JSON struct metrics path (optional).")
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parent.parent
    pickle_path = repo / args.pickle
    if not pickle_path.is_file():
        raise FileNotFoundError(pickle_path)

    with open(pickle_path, "rb") as handle:
        data = pickle.load(handle)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}

    stem = pickle_path.stem.replace("_silver_spans", "").replace("silver_spans", "hkt_silver")
    default_base = f"log/{stem}_inspect"

    out_meta = repo / (args.out_meta or f"{default_base}_meta.json")
    out_spot = repo / (args.out_spotcheck or f"{default_base}_spotcheck.txt")
    out_struct = repo / (args.out_struct or f"{default_base}_struct.json")

    meta_out = {"pickle": str(pickle_path.resolve()), "meta": meta}
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_meta}")

    rng = random.Random(args.seed)
    lines: list[str] = []
    lines.append(f"HKT silver spotcheck pickle={pickle_path.name} seed={args.seed} n_per_split={args.n_per_split}")
    lines.append(f"meta.parser_model={meta.get('parser_model')!r} dataset={meta.get('dataset')!r}")
    lines.append("")

    for split in ("train", "dev", "test"):
        recs = data.get(split)
        if not isinstance(recs, list) or not recs:
            lines.append(f"=== {split}: (empty) ===\n")
            continue
        k = min(args.n_per_split, len(recs))
        idxs = rng.sample(range(len(recs)), k=k)
        idxs.sort()
        lines.append(f"=== {split} (showing {k} of {len(recs)}) ===")
        for j in idxs:
            rec = recs[j]
            words = list(rec.get("words") or [])
            spans = rec.get("word_spans") or []
            seg = rec.get("segment", "")
            joined = " ".join(str(w) for w in words)
            phrases = _span_phrases_english(words, spans)
            pt = rec.get("parse_tree")
            pt_snip = ""
            if isinstance(pt, str) and pt:
                pt_snip = pt[: args.parse_tree_chars].replace("\n", " ")
                if len(pt) > args.parse_tree_chars:
                    pt_snip += "..."
            lines.append(
                f"--- idx={j} segment={seg!r} n_words={len(words)} n_spans={len(spans)} parse_error={rec.get('parse_error')!r} ---"
            )
            head = joined[:240] + ("..." if len(joined) > 240 else "")
            lines.append(f"text: {head}")
            lines.append(f"span_phrases ({len(phrases)}): {phrases[:14]}{' ...' if len(phrases) > 14 else ''}")
            if pt_snip:
                lines.append(f"parse_tree_snip: {pt_snip}")
            lines.append("")
        lines.append("")

    out_spot.parent.mkdir(parents=True, exist_ok=True)
    out_spot.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_spot}")

    struct: dict = {"pickle": str(pickle_path.resolve()), "meta": meta, "splits": {}}
    for split in ("train", "dev", "test"):
        recs = data.get(split)
        if isinstance(recs, list) and recs:
            struct["splits"][split] = _struct_one_split(recs)

    out_struct.parent.mkdir(parents=True, exist_ok=True)
    out_struct.write_text(json.dumps(struct, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_struct}")


if __name__ == "__main__":
    main()
