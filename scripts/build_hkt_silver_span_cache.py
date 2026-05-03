#!/usr/bin/env python3
"""Build benepar silver constituency caches for MUStARD / UR-FUNNY (HKT pickles).

Writes ``datasets/{mustard,urfunny}_silver_spans.pkl`` in the same schema as
``scripts/build_silver_span_cache.py`` (train/dev/test lists of records with
``words``, ``word_spans``, ``segment`` = sample id ``hid``).

**MUStARD k-fold:** pass the same ``--fold`` / ``--seed`` / ``--dev_ratio`` /
``--dataset_cache`` as ``train_hkt_binary.py`` so cache rows align with the
training split.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hkt_data import (  # noqa: E402
    REPO_ROOT as HKT_REPO_ROOT,
    load_hkt_dataset,
    normalize_dataset_name,
)
from scripts.build_silver_span_cache import (  # noqa: E402
    build_empty_record,
    build_record,
    configure_runtime,
    load_parser,
)
from silver_span_utils import target_words_from_p_field  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build HKT silver span caches (benepar on target utterance).")
    parser.add_argument("--dataset", required=True, choices=["mustard", "urfunny", "sarcasm", "humor"])
    parser.add_argument("--repo-root", type=str, default=str(HKT_REPO_ROOT))
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--dataset-cache", type=str, default="", help="Same as train_hkt_binary --dataset_cache.")
    parser.add_argument("--fold", type=int, default=-1, help="MUStARD SI fold (only when no official pickle).")
    parser.add_argument("--seed", type=int, default=5149)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--parser-model", default="benepar_en3", type=str)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], type=str)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    return parser.parse_args()


def _fake_example_for_record(words: list, hid: str):
    return ((words, None, None), 0.0, str(hid))


def build_split_cache(split_name, examples, benepar_module, parser, limit: int):
    split_cache = []
    parse_limit = limit if limit > 0 else len(examples)

    for example_index, example in enumerate(tqdm(examples, desc=f"Parsing {split_name}")):
        (
            (p_words, _pv, _pa, _ph),
            (_cw, _cv, _ca, _ch),
            hid,
            _label,
        ) = example
        words = target_words_from_p_field(p_words)
        fake = _fake_example_for_record(words, str(hid))

        if example_index >= parse_limit:
            split_cache.append(build_empty_record(fake))
            continue

        rec = build_record(fake, benepar_module, parser)
        rec["segment"] = str(hid)
        split_cache.append(rec)

    return split_cache


def summarize_split_cache(split_name, split_cache):
    total = len(split_cache)
    nonempty = sum(1 for record in split_cache if record.get("word_spans"))
    parse_errors = sum(1 for record in split_cache if record.get("parse_error"))
    print(
        f"CACHE_SUMMARY[{split_name}]: total={total}, nonempty={nonempty}, parse_errors={parse_errors}",
        flush=True,
    )
    return {"total": total, "nonempty": nonempty, "parse_errors": parse_errors}


def main():
    args = parse_args()
    dataset = normalize_dataset_name(args.dataset)
    fold = None if args.fold < 0 else int(args.fold)

    configure_runtime(args.device)

    output_path = Path(
        args.output_path
        if args.output_path
        else os.path.join(args.repo_root, "datasets", f"{dataset}_silver_spans.pkl")
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (use --overwrite)")

    split_payload, split_source = load_hkt_dataset(
        dataset,
        repo_root=args.repo_root,
        seed=args.seed,
        cache_path=args.dataset_cache,
        fold=fold,
        dev_ratio=args.dev_ratio,
    )
    print(f"HKT_SILVER_BUILD: data_source={split_source}", flush=True)

    benepar_module, parser = load_parser(args.parser_model, download_model=args.download_model)

    train_cache = build_split_cache("train", split_payload["train"], benepar_module, parser, args.limit)
    dev_cache = build_split_cache("dev", split_payload.get("dev") or [], benepar_module, parser, args.limit)
    test_cache = build_split_cache("test", split_payload["test"], benepar_module, parser, args.limit)

    summary = {
        "train": summarize_split_cache("train", train_cache),
        "dev": summarize_split_cache("dev", dev_cache),
        "test": summarize_split_cache("test", test_cache),
    }

    cache = {
        "meta": {
            "dataset": dataset,
            "split_source": split_source,
            "parser_model": args.parser_model,
            "device": args.device,
            "limit": args.limit,
            "fold": fold,
            "seed": args.seed,
            "dev_ratio": args.dev_ratio,
            "dataset_cache": args.dataset_cache or None,
            "summary": summary,
        },
        "train": train_cache,
        "dev": dev_cache,
        "test": test_cache,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(cache, handle)
    print(f"Wrote HKT silver span cache to {output_path}", flush=True)


if __name__ == "__main__":
    main()
