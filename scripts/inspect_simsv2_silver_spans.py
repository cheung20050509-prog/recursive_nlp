#!/usr/bin/env python3
"""SIMSv2 silver span cache QA: meta dump, stratified spot-check, struct metrics, alignment.

Reads ``datasets/simsv2_silver_spans.pkl`` (or ``--pickle``). Optionally compares
``words`` to ``datasets/simsv2.pkl`` after ``normalize_simsv2_pickled_data``.

Example::

    cd ITHP/recursive_ITHP
    python scripts/inspect_simsv2_silver_spans.py \\
        --pickle datasets/simsv2_silver_spans.pkl \\
        --simsv2 datasets/simsv2.pkl \\
        --n-per-split 35 --seed 7 \\
        --out-meta log/simsv2_silver_meta.json \\
        --out-spotcheck log/simsv2_silver_manual_spotcheck.txt \\
        --out-struct log/simsv2_silver_struct_metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import statistics
from pathlib import Path


def _is_cjk_char(ch: str) -> bool:
    if len(ch) != 1:
        return False
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF


def _word_cjk_ratio(words: list) -> float:
    if not words:
        return 0.0
    cjk = sum(1 for w in words for ch in str(w) if _is_cjk_char(ch))
    tot = sum(len(str(w)) for w in words)
    return round(cjk / tot, 6) if tot else 0.0


def _span_phrases(words: list, word_spans: list) -> list[str]:
    out = []
    n = len(words)
    for span in word_spans or []:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        a, b = int(span[0]), int(span[1])
        if a < 0 or b > n or a >= b:
            continue
        piece = words[a:b]
        out.append("".join(str(w) for w in piece))
    return out


def _struct_one_split(records: list) -> dict:
    span_lens: list[int] = []
    cover_counts: list[int] = []
    n_words_list: list[int] = []
    cjk_ratios: list[float] = []

    for rec in records:
        words = list(rec.get("words") or [])
        n_words_list.append(len(words))
        spans = rec.get("word_spans") or []
        covered = set()
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                continue
            a, b = int(span[0]), int(span[1])
            span_lens.append(b - a)
            for i in range(a, b):
                covered.add(i)
        lw = len(words)
        cover_counts.append(len(covered) / lw if lw else 0.0)
        cjk_ratios.append(_word_cjk_ratio(words))

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * p))))
        return round(xs[k], 6)

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
        "cjk_char_ratio_in_words_mean": round(statistics.mean(cjk_ratios), 4) if cjk_ratios else 0.0,
    }


def _align_split(
    split: str,
    cache_recs: list,
    examples: list,
) -> dict:
    mismatches = 0
    first: list[dict] = []
    n = min(len(cache_recs), len(examples))
    if len(cache_recs) != len(examples):
        return {
            "error": "length_mismatch",
            "cache_len": len(cache_recs),
            "dataset_len": len(examples),
        }
    for i in range(n):
        (words, _, _), _, _seg = examples[i]
        words = list(words)
        cw = list(cache_recs[i].get("words") or [])
        if cw != words:
            mismatches += 1
            if len(first) < 5:
                first.append({"index": i, "cache_words_head": cw[:12], "data_words_head": words[:12]})
    return {"split": split, "n": n, "mismatches": mismatches, "mismatch_rate": round(mismatches / n, 8) if n else 0.0, "examples": first}


def parse_args():
    p = argparse.ArgumentParser(description="Inspect SIMSv2 silver span cache quality.")
    p.add_argument("--pickle", default="datasets/simsv2_silver_spans.pkl")
    p.add_argument("--simsv2", default="datasets/simsv2.pkl", help="For full words alignment (optional).")
    p.add_argument("--no-align", action="store_true", help="Skip alignment vs simsv2.pkl.")
    p.add_argument("--n-per-split", type=int, default=35)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--parse-tree-chars", type=int, default=220)
    p.add_argument("--out-meta", default="log/simsv2_silver_meta.json")
    p.add_argument("--out-spotcheck", default="log/simsv2_silver_manual_spotcheck.txt")
    p.add_argument("--out-struct", default="log/simsv2_silver_struct_metrics.json")
    p.add_argument("--out-recommendation", default="log/simsv2_silver_rebuild_recommendation.md")
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

    meta_out = {"pickle": str(pickle_path.resolve()), "meta": meta}
    if args.out_meta:
        out_meta = repo / args.out_meta
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        out_meta.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out_meta}")

    rng = random.Random(args.seed)
    lines: list[str] = []
    lines.append(f"SIMSv2 silver spotcheck seed={args.seed} n_per_split={args.n_per_split}")
    lines.append(f"pickle={pickle_path}")
    lines.append(f"meta.parser_model={meta.get('parser_model')!r} device={meta.get('device')!r}")
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
            joined = "".join(str(w) for w in words)
            phrases = _span_phrases(words, spans)
            pt = rec.get("parse_tree")
            pt_snip = ""
            if isinstance(pt, str) and pt:
                pt_snip = pt[: args.parse_tree_chars].replace("\n", " ") + ("..." if len(pt) > args.parse_tree_chars else "")
            lines.append(f"--- idx={j} n_words={len(words)} n_spans={len(spans)} parse_error={rec.get('parse_error')!r} ---")
            lines.append(f"text_joined: {joined[:200]}{'...' if len(joined) > 200 else ''}")
            lines.append(f"cjk_ratio(chars): {_word_cjk_ratio(words)}")
            lines.append(f"span_phrases ({len(phrases)}): {phrases[:12]}{' ...' if len(phrases) > 12 else ''}")
            if pt_snip:
                lines.append(f"parse_tree_snip: {pt_snip}")
            lines.append("")
        lines.append("")

    spot_path = repo / args.out_spotcheck
    spot_path.parent.mkdir(parents=True, exist_ok=True)
    spot_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {spot_path}")

    struct: dict = {"pickle": str(pickle_path), "meta": meta, "splits": {}}
    for split in ("train", "dev", "test"):
        recs = data.get(split)
        if isinstance(recs, list) and recs:
            struct["splits"][split] = _struct_one_split(recs)

    align_block: dict | None = None
    if args.no_align:
        struct["alignment_vs_simsv2_pkl"] = {"skipped": True}
    else:
        sim_path = repo / args.simsv2
        if sim_path.is_file():
            import sys

            sys.path.insert(0, str(repo))
            from simsv2_data import normalize_simsv2_pickled_data

            with open(sim_path, "rb") as handle:
                raw = pickle.load(handle)
            norm = normalize_simsv2_pickled_data(raw)
            align_block = {}
            for split, key in ("train", "train"), ("dev", "dev"), ("test", "test"):
                ex = norm.get(key)
                cr = data.get(split)
                if isinstance(ex, list) and isinstance(cr, list):
                    align_block[split] = _align_split(split, cr, ex)
            struct["alignment_vs_simsv2_pkl"] = align_block
        else:
            align_block = {"error": "file_not_found", "path": str(sim_path)}
            struct["alignment_vs_simsv2_pkl"] = align_block

    parser = str(meta.get("parser_model", "")).lower()
    cjk_mean = statistics.mean(
        struct["splits"][s]["cjk_char_ratio_in_words_mean"] for s in struct["splits"] if s in struct["splits"]
    ) if struct["splits"] else 0.0

    rec_lines = [
        "# SIMSv2 silver cache — rebuild / parser recommendation",
        "",
        f"- **Current parser** (from pickle meta): `{meta.get('parser_model')}` on **device** `{meta.get('device')}`.",
        f"- **Mean CJK character ratio** (char-based, averaged over per-example ratios): **{cjk_mean:.4f}**.",
        "",
    ]
    if "benepar_en" in parser or parser == "benepar_en3":
        rec_lines.append(
            "- **Risk**: English benepar models are a weak fit for **Chinese-dominant** `words` lists; "
            "parses may be syntactically odd even with `parse_errors: 0`."
        )
    if cjk_mean > 0.25:
        rec_lines.append(
            "- **Suggestion**: Prefer a **Chinese-capable constituency parser** (or word-segment first) "
            "and rebuild with [`scripts/build_simsv2_silver_span_cache.py`](scripts/build_simsv2_silver_span_cache.py) "
            "`--parser-model ...` (see benepar / spaCy Chinese docs)."
        )
    mis_any = False
    if isinstance(align_block, dict) and "error" not in align_block:
        mis_any = any(align_block.get(s, {}).get("mismatches", 0) for s in ("train", "dev", "test"))
    if args.no_align:
        rec_lines.append("- **Alignment**: skipped (`--no-align`).")
    elif mis_any:
        rec_lines.append("- **Critical**: Word alignment mismatches vs `simsv2.pkl` — **rebuild cache** after fixing normalization.")
    elif isinstance(align_block, dict) and align_block.get("error") == "file_not_found":
        rec_lines.append("- **Alignment**: `simsv2.pkl` not found; alignment skipped.")
    else:
        rec_lines.append("- **Alignment**: No word-list mismatches vs normalized `simsv2.pkl`.")

    rec_lines.extend(
        [
            "",
            "## Rebuild command (full; long-running)",
            "",
            "```bash",
            "cd ITHP/recursive_ITHP",
            "# Example only — pick a Chinese-appropriate benepar model if available in your env:",
            "python scripts/build_simsv2_silver_span_cache.py \\",
            "  --input datasets/simsv2.pkl \\",
            "  --output datasets/simsv2_silver_spans.pkl \\",
            "  --parser-model <your_model> --device cuda --overwrite",
            "```",
            "",
        ]
    )

    struct["recommendation_bullets"] = [ln.lstrip("- ").strip() for ln in rec_lines if ln.startswith("- **")]
    rec_path = repo / args.out_recommendation
    rec_path.write_text("\n".join(rec_lines), encoding="utf-8")
    print(f"Wrote {rec_path}")

    struct_path = repo / args.out_struct
    struct_path.write_text(json.dumps(struct, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {struct_path}")


if __name__ == "__main__":
    main()
