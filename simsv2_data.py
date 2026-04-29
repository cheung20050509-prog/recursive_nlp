"""CH-SIMSv2: MMSA/KuDA batch-dict format -> ITHP list format (shared by train.py and build script)."""

from __future__ import annotations

import numpy as np


def _clean_modal_array(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    x[np.isneginf(x)] = 0.0
    x[np.isposinf(x)] = 0.0
    x[np.isnan(x)] = 0.0
    return x


def _resample_time(seq: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    t = seq.shape[0]
    if t == 0:
        return np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    if t == target_len:
        return seq.astype(np.float32)
    idx = np.linspace(0, t - 1, target_len).round().astype(np.int64)
    idx = np.clip(idx, 0, t - 1)
    return seq[idx].astype(np.float32)


def _is_mmsa_simsv2_split_dict(split_obj) -> bool:
    if not isinstance(split_obj, dict):
        return False
    need = ("raw_text", "vision", "audio", "regression_labels")
    return all(k in split_obj for k in need)


def convert_mmsa_split_to_ithp_list(split_dict: dict) -> list:
    """One split: stacked arrays -> list of ``((words, V, A), label, segment_id)``."""
    raw_texts = split_dict["raw_text"]
    vision = _clean_modal_array(split_dict["vision"])
    audio = _clean_modal_array(split_dict["audio"])
    labels = np.asarray(split_dict["regression_labels"], dtype=np.float32).reshape(-1)
    ids = split_dict.get("id")
    n = len(labels)
    if vision.shape[0] != n or audio.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, vision {vision.shape}, audio {audio.shape}")

    out = []
    for i in range(n):
        text = raw_texts[i]
        if isinstance(text, (list, tuple)):
            words = [str(w) for w in text]
        else:
            s = str(text).strip()
            words = list(s) if s else [""]

        vis_i = vision[i]
        aud_i = audio[i]
        if vis_i.ndim != 2 or aud_i.ndim != 2:
            raise ValueError(f"Sample {i}: expected 2D vision/audio, got {vis_i.shape}, {aud_i.shape}")

        L = len(words)
        vis_w = _resample_time(vis_i, L)
        aud_w = _resample_time(aud_i, L)
        seg = str(ids[i]) if ids is not None else str(i)
        out.append(((words, vis_w, aud_w), float(labels[i]), seg))
    return out


def normalize_simsv2_pickled_data(data: dict) -> dict:
    """Return ``{train, dev, test}`` with ITHP-style lists (MMSA dicts converted in-memory)."""
    train = data["train"]
    if _is_mmsa_simsv2_split_dict(train):
        if "valid" not in data or "test" not in data:
            raise ValueError("MMSA-style CH-SIMSv2 pickle must contain train, valid, and test splits")
        return {
            "train": convert_mmsa_split_to_ithp_list(data["train"]),
            "dev": convert_mmsa_split_to_ithp_list(data["valid"]),
            "test": convert_mmsa_split_to_ithp_list(data["test"]),
        }

    out = {"train": train, "test": data["test"]}
    if "dev" in data:
        out["dev"] = data["dev"]
    elif "valid" in data:
        va = data["valid"]
        out["dev"] = convert_mmsa_split_to_ithp_list(va) if _is_mmsa_simsv2_split_dict(va) else va
    else:
        raise KeyError("simsv2 pickle must have 'dev' or 'valid' split")
    return out
