import argparse
import pickle
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Build silver constituency span caches for ITHP datasets.")
    parser.add_argument("--dataset-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--parser-model", default="benepar_en3", type=str)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    return parser.parse_args()


def load_parser(model_name, download_model=False):
    try:
        import benepar
    except ImportError as exc:
        raise RuntimeError(
            "benepar is required to build silver span caches. Install optional parser deps first."
        ) from exc

    if download_model:
        benepar.download(model_name)

    try:
        parser = benepar.Parser(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load benepar model '{model_name}'. Run with --download-model or pre-download it manually."
        ) from exc

    return benepar, parser


def extract_word_spans(tree):
    spans = set()

    def visit(node, start_index):
        if isinstance(node, str):
            return start_index + 1

        cursor = start_index
        for child in node:
            cursor = visit(child, cursor)

        end_index = cursor
        if end_index - start_index >= 2:
            spans.add((start_index, end_index))
        return end_index

    visit(tree, 0)
    return [list(span) for span in sorted(spans)]


def build_empty_record(example):
    (words, _, _), _, segment = example
    return {
        "segment": segment,
        "words": list(words),
        "word_spans": [],
        "parse_tree": None,
        "parse_error": None,
    }


def build_record(example, benepar_module, parser):
    (words, _, _), _, segment = example
    record = build_empty_record(example)

    if len(words) < 2:
        return record

    try:
        tree = parser.parse(benepar_module.InputSentence(words=list(words)))
        record["word_spans"] = extract_word_spans(tree)
        record["parse_tree"] = tree.pformat(margin=10**9)
    except Exception as exc:
        record["parse_error"] = f"{type(exc).__name__}: {exc}"

    return record


def build_split_cache(split_name, examples, benepar_module, parser, limit=0):
    split_cache = []
    parse_limit = limit if limit > 0 else len(examples)

    for example_index, example in enumerate(tqdm(examples, desc=f"Parsing {split_name}")):
        if example_index >= parse_limit:
            split_cache.append(build_empty_record(example))
            continue
        split_cache.append(build_record(example, benepar_module, parser))

    return split_cache


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    with open(args.dataset_path, "rb") as handle:
        data = pickle.load(handle)

    missing_splits = [split_name for split_name in ("train", "dev", "test") if split_name not in data]
    if missing_splits:
        raise ValueError(f"Dataset pickle missing splits: {missing_splits}")

    benepar_module, parser = load_parser(args.parser_model, download_model=args.download_model)

    cache = {
        "meta": {
            "dataset_path": args.dataset_path,
            "parser_model": args.parser_model,
            "limit": args.limit,
        },
        "train": build_split_cache("train", data["train"], benepar_module, parser, limit=args.limit),
        "dev": build_split_cache("dev", data["dev"], benepar_module, parser, limit=args.limit),
        "test": build_split_cache("test", data["test"], benepar_module, parser, limit=args.limit),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(cache, handle)

    print(f"Wrote silver span cache to {output_path}")


if __name__ == "__main__":
    main()