"""Shared helpers for silver constituency span caches (MOSI-style records + token masks)."""

from __future__ import annotations

import os
import pickle
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from hkt_data import heuristic_word_tokenize


def get_silver_span_record(
    split_records: Any,
    example_index: int,
    words: Sequence[str],
    segment: str,
) -> Optional[dict]:
    if split_records is None:
        return None

    if isinstance(split_records, list):
        if example_index >= len(split_records):
            raise IndexError(
                f"Silver span cache is shorter than dataset: idx={example_index}, cache_size={len(split_records)}"
            )
        record = split_records[example_index]
    elif isinstance(split_records, dict):
        record = split_records.get(segment)
        if record is None:
            record = split_records.get(str(example_index))
    else:
        raise TypeError("Silver span cache split must be either a list or a dict")

    if record is None:
        raise KeyError(f"Missing silver span record for segment={segment}, index={example_index}")

    record_words = record.get("words") if isinstance(record, dict) else None
    if record_words is not None and list(record_words) != list(words):
        raise ValueError(f"Silver span cache word mismatch for segment={segment}, index={example_index}")

    return record


def build_silver_span_mask(
    silver_record: Optional[dict],
    inversions: Sequence[int],
    max_seq_length: int,
    token_index_offset: int = 1,
) -> np.ndarray:
    """Token-level span corner mask (same convention as ``train.py``).

    ``token_index_offset`` defaults to ``1`` for a single segment ``[CLS] + tokens``.
    Use ``0`` when ``inversions`` already indexes a local segment without a leading CLS.
    """
    span_mask = np.zeros((max_seq_length, max_seq_length), dtype=np.float32)
    if silver_record is None or len(inversions) == 0:
        return span_mask

    word_spans = silver_record.get("word_spans") or silver_record.get("spans") or []
    if not word_spans:
        return span_mask

    word_to_token_start: dict[int, int] = {}
    word_to_token_end: dict[int, int] = {}
    for token_idx, word_idx in enumerate(inversions):
        word_to_token_start.setdefault(word_idx, token_idx)
        word_to_token_end[word_idx] = token_idx + 1

    for word_start, word_end in word_spans:
        word_start = int(word_start)
        word_end = int(word_end)

        if word_end - word_start < 2:
            continue

        if word_start not in word_to_token_start or (word_end - 1) not in word_to_token_end:
            continue

        token_start = word_to_token_start[word_start] + token_index_offset
        token_end = word_to_token_end[word_end - 1] + token_index_offset

        if token_end - token_start < 2:
            continue

        if token_start >= max_seq_length or token_end - 1 >= max_seq_length:
            continue

        span_mask[token_start, token_end - 1] = 1.0

    return span_mask


def target_words_from_p_field(p_words: Any) -> List[str]:
    """Target-side word list aligned with HKT pickles (p_* = current / punchline)."""
    if isinstance(p_words, str):
        return heuristic_word_tokenize(p_words)
    return [str(w) for w in p_words]


def tokenize_word_list(words: Sequence[str], tokenizer) -> Tuple[list, list]:
    """MOSI-style per-word subword expansion + inversion indices."""
    tokens: list = []
    inversions: list = []
    for idx, word in enumerate(words):
        for piece in tokenizer.tokenize(word):
            tokens.append(piece)
            inversions.append(idx)
    return tokens, inversions


def build_pair_target_syntax_span_mask(
    silver_record: Optional[dict],
    inversions_b: Sequence[int],
    offset_b: int,
    max_seq_length: int,
) -> np.ndarray:
    """Place a target-segment span mask into the full pair sequence (DeBERTa pair layout)."""
    full = np.zeros((max_seq_length, max_seq_length), dtype=np.float32)
    lb = len(inversions_b)
    if lb == 0 or offset_b < 0:
        return full

    local_max = min(lb, max(0, max_seq_length - offset_b))
    if local_max <= 0:
        return full

    inv_trim = list(inversions_b[:local_max])
    local_mask = build_silver_span_mask(silver_record, inv_trim, local_max, token_index_offset=0)
    end = offset_b + local_max
    if end > max_seq_length:
        trim = max_seq_length - offset_b
        local_mask = local_mask[:trim, :trim]
        end = max_seq_length
    full[offset_b:end, offset_b:end] = local_mask
    return full


def load_hkt_silver_span_cache(cache_path: str) -> Optional[dict]:
    if not cache_path:
        return None
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Silver span cache not found: {cache_path}")

    with open(cache_path, "rb") as handle:
        cache = pickle.load(handle)

    if not isinstance(cache, dict):
        raise ValueError("Silver span cache must be a dict with train/dev/test splits")

    missing_splits = [split_name for split_name in ("train", "dev", "test") if split_name not in cache]
    if missing_splits:
        raise ValueError(f"Silver span cache missing splits: {missing_splits}")

    for split_name in ("train", "dev", "test"):
        split_records = cache[split_name]
        if isinstance(split_records, dict):
            iterable = split_records.values()
        else:
            iterable = split_records

        nonempty = sum(1 for record in iterable if record.get("word_spans") or record.get("spans"))
        parse_errors = sum(1 for record in iterable if record.get("parse_error"))
        total = len(split_records) if isinstance(split_records, list) else len(split_records)
        print(
            f"HKT_SILVER_SPAN_CACHE[{split_name}]: total={total}, nonempty={nonempty}, parse_errors={parse_errors}",
            flush=True,
        )

    print(f"HKT_SILVER_SPAN_CACHE: loaded {cache_path}", flush=True)
    return cache


def resolve_hkt_silver_span_cache_path(dataset_name: str, explicit_path: str) -> str:
    if explicit_path:
        return explicit_path
    default_cache_path = os.path.join("datasets", f"{dataset_name}_silver_spans.pkl")
    if os.path.exists(default_cache_path):
        return default_cache_path
    return ""
