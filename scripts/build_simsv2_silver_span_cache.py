#!/usr/bin/env python3
"""Build ``datasets/simsv2_silver_spans.pkl`` for RITHM / train.py (same layout as MOSI).

CH-SIMS v2 pickles are often MMSA-style dicts (``train`` / ``valid`` / ``test``). This
script normalizes them with ``simsv2_data.normalize_simsv2_pickled_data`` so each
example is ``((words, V, A), label, segment)`` — the same layout expected by
``build_silver_span_cache.py`` — then runs the benepar-based span extraction.

**Chinese text:** default ``--parser-model benepar_en3`` is English-oriented. For
Chinese clauses you should pass a benepar model that supports your tokenization
(e.g. word-segmented ``words`` lists), or accept sparse/empty spans for ablation.
See README (SIMSv2 silver spans).

Example::

    python scripts/build_simsv2_silver_span_cache.py \\
        --input datasets/simsv2.pkl \\
        --output datasets/simsv2_silver_spans.pkl \\
        --parser-model benepar_en3 --device cpu --download-model
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Normalize CH-SIMS v2 pickle and build silver span cache.")
    p.add_argument("--input", default="datasets/simsv2.pkl", help="Source pickle (MMSA or ITHP list format).")
    p.add_argument("--output", default="datasets/simsv2_silver_spans.pkl", help="Written silver cache path.")
    p.add_argument("--parser-model", default="benepar_en3", help="benepar parser name (see benepar docs).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--download-model", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Parse only first N examples per split (debug).")
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from simsv2_data import normalize_simsv2_pickled_data

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input pickle not found: {input_path.resolve()}")

    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {out_path}")

    with open(input_path, "rb") as handle:
        raw = pickle.load(handle)

    data = normalize_simsv2_pickled_data(raw)
    for k in ("train", "dev", "test"):
        if k not in data:
            raise KeyError(f"After normalize, expected key {k!r} in data")

    tmp = repo / "datasets" / ".simsv2_ithp_normalized_for_silver.pkl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as handle:
        pickle.dump(data, handle)

    build_script = repo / "scripts" / "build_silver_span_cache.py"
    cmd = [
        sys.executable,
        str(build_script),
        "--dataset-path",
        str(tmp),
        "--output-path",
        str(out_path),
        "--parser-model",
        args.parser_model,
        "--device",
        args.device,
    ]
    if args.download_model:
        cmd.append("--download-model")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    try:
        tmp.unlink()
    except OSError:
        pass
    print(f"Done. Silver cache: {out_path.resolve()}")


if __name__ == "__main__":
    main()
