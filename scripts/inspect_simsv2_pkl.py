#!/usr/bin/env python3
"""Inspect datasets/simsv2.pkl structure (ITHP list format vs MMSA dict format)."""

import argparse
import os
import pickle
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=os.path.join(REPO_ROOT, "datasets", "simsv2.pkl"),
        help="Path to simsv2 pickle",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.path):
        print(f"Missing: {args.path}", file=sys.stderr)
        sys.exit(1)

    with open(args.path, "rb") as handle:
        data = pickle.load(handle)

    print("top_keys:", list(data.keys()) if isinstance(data, dict) else type(data))

    if not isinstance(data, dict) or "train" not in data:
        return

    train = data["train"]
    print("train type:", type(train), "len:", len(train) if hasattr(train, "__len__") else "NA")

    if isinstance(train, list) and len(train) > 0:
        ex = train[0]
        print("train[0] type:", type(ex), "len:", len(ex) if isinstance(ex, (list, tuple)) else "")
        if isinstance(ex, (list, tuple)) and len(ex) >= 2:
            inner, label = ex[0], ex[1]
            print("  label:", label)
            if isinstance(inner, tuple) and len(inner) == 3:
                words, vis, ac = inner
                print("  words (len):", len(words) if hasattr(words, "__len__") else words)
                print("  visual shape:", getattr(vis, "shape", None))
                print("  acoustic shape:", getattr(ac, "shape", None))
    elif isinstance(train, dict):
        print("train dict keys:", list(train.keys())[:30])
        for k in ("vision", "audio", "raw_text", "regression_labels", "text_bert"):
            if k in train:
                v = train[k]
                print(f"  {k} type:", type(v), "shape:", getattr(v, "shape", None))


if __name__ == "__main__":
    main()
