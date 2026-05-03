#!/usr/bin/env python3
"""Summarize silver span pickle caches (nonempty spans, parse errors, word lengths).

Usage (from repo root)::

    python scripts/audit_silver_span_caches.py
    python scripts/audit_silver_span_caches.py --pickles datasets/simsv2_silver_spans.pkl datasets/mustard_silver_spans.pkl

Writes JSON to ``--output-json`` (default: ``log/silver_cache_audit.json``).
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
from pathlib import Path


def summarize_split(records: list) -> dict:
    total = len(records)
    nonempty = sum(1 for r in records if r.get("word_spans"))
    parse_errors = sum(1 for r in records if r.get("parse_error"))
    lens = [len(r.get("words") or []) for r in records]
    span_counts = [len(r.get("word_spans") or []) for r in records]
    return {
        "total": total,
        "nonempty_word_spans": nonempty,
        "nonempty_rate": round(nonempty / total, 6) if total else 0.0,
        "parse_errors": parse_errors,
        "parse_error_rate": round(parse_errors / total, 6) if total else 0.0,
        "words_len_mean": round(statistics.mean(lens), 4) if lens else 0.0,
        "words_len_median": round(statistics.median(lens), 4) if lens else 0.0,
        "spans_per_record_mean": round(statistics.mean(span_counts), 4) if span_counts else 0.0,
    }


def audit_pickle(path: Path) -> dict:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    out = {"path": str(path.resolve()), "meta": meta, "splits": {}}
    for split in ("train", "dev", "test"):
        if split not in data:
            continue
        recs = data[split]
        if not isinstance(recs, list):
            continue
        out["splits"][split] = summarize_split(recs)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Audit silver span cache pickles.")
    p.add_argument(
        "--pickles",
        nargs="*",
        default=[
            "datasets/simsv2_silver_spans.pkl",
            "datasets/mustard_silver_spans.pkl",
        ],
        help="Relative paths under repo root.",
    )
    p.add_argument(
        "--output-json",
        default="log/silver_cache_audit.json",
        help="Relative path under repo root for JSON report.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parent.parent
    report = {"repo": str(repo), "caches": []}
    for rel in args.pickles:
        path = repo / rel
        if not path.is_file():
            report["caches"].append({"path": str(path), "error": "file_not_found"})
            continue
        report["caches"].append(audit_pickle(path))
    out_path = repo / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
