mnnnnnnnnnnn,.......#!/usr/bin/env python3
"""Convert MMSA/KuDA-style CH-SIMSv2 pickle to ITHP train.py list format."""

from __future__ import annotations

import argparse
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from simsv2_data import convert_mmsa_split_to_ithp_list  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="MMSA/KuDA-style pickle path")
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "datasets", "simsv2.pkl"),
        help="Output ITHP-format pickle path",
    )
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        src = pickle.load(handle)

    required = ("train", "valid", "test")
    missing = [k for k in required if k not in src]
    if missing:
        if "dev" in src and "valid" not in src:
            src["valid"] = src["dev"]
            missing = [k for k in required if k not in src]
    if missing:
        print(f"Input missing splits {missing}; keys={list(src.keys())}", file=sys.stderr)
        sys.exit(1)

    for split in required:
        d = src[split]
        need = ("raw_text", "vision", "audio", "regression_labels")
        bad = [k for k in need if k not in d]
        if bad:
            print(f"Split {split} missing keys {bad}; have {list(d.keys())[:40]}", file=sys.stderr)
            sys.exit(1)

    out = {
        "train": convert_mmsa_split_to_ithp_list(src["train"]),
        "dev": convert_mmsa_split_to_ithp_list(src["valid"]),
        "test": convert_mmsa_split_to_ithp_list(src["test"]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.output} sizes train={len(out['train'])} dev={len(out['dev'])} test={len(out['test'])}")


if __name__ == "__main__":
    main()
