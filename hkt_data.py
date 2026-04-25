import json
import os
import pickle
import re
import sys
import zipfile

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold


DATASET_ALIASES = {
    "humor": "urfunny",
    "sarcasm": "mustard",
    "ur_funny": "urfunny",
}

HKT_DEFAULT_MAX_SEQ_LENGTH = {
    "mustard": 77,
    "urfunny": 64,
}

HKT_VISUAL_FEATURES = list(range(55, 91))
HKT_ACOUSTIC_FEATURES = list(range(0, 60))
VISUAL_DIM_ALL = 91
ACOUSTIC_DIM_ALL = 81
HCF_DIM = 4

MUSTARD_NUM_FOLDS = 5

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MUSTARD_METADATA_PATH = "/root/autodl-tmp/datasets/MUStARD/sarcasm_data.json"
DEFAULT_URFUNNY_ARCHIVE_PATH = "/root/autodl-tmp/datasets/UR-FUNNY-v2/urfunny_v2_features.zip"


def normalize_dataset_name(dataset_name):
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def get_hkt_filename(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    return "ur_funny.pkl" if dataset_name == "urfunny" else "mustard.pkl"


def get_official_hkt_path(repo_root, dataset_name):
    return os.path.join(repo_root, "datasets", "hkt", get_hkt_filename(dataset_name))


def get_candidate_hkt_paths(repo_root, dataset_name):
    """Ordered list of likely locations for a HKT 4-tuple split pickle.

    - ``datasets/hkt/<ds>.pkl`` matches matalvepu/HKT's layout.
    - ``datasets/<ds>.pkl`` is where the newly transferred "standard" HKT
      pickles live in this repo (same 4-tuple schema, different directory).
    """
    return [
        get_official_hkt_path(repo_root, dataset_name),
        os.path.join(repo_root, "datasets", get_hkt_filename(dataset_name)),
    ]


def load_pickle(path):
    # Older pickles may reference numpy._core while this environment exposes numpy.core.
    if "numpy._core" not in sys.modules:
        sys.modules["numpy._core"] = np.core
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickle(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def load_hkt_pickle_if_valid(path):
    if not path or not os.path.exists(path):
        return None

    try:
        payload = load_pickle(path)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    if not all(split_name in payload for split_name in ("train", "dev", "test")):
        return None

    return payload


def extract_split_ids_from_hkt_pickle(path):
    payload = load_hkt_pickle_if_valid(path)
    if payload is None:
        return None

    split_ids = {}
    for split_name in ("train", "dev", "test"):
        split_ids[split_name] = [str(example[2]) for example in payload[split_name]]

    return split_ids


def heuristic_word_tokenize(text):
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
    if tokens:
        return tokens

    text = text.strip()
    return [text] if text else []


def flatten_sentence_word_lists(sentence_word_lists):
    words = []
    for sentence_words in sentence_word_lists:
        words.extend(sentence_words)
    return words


def nearest_resample(sequence, target_length):
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2:
        raise ValueError(f"Expected a 2D sequence, got shape {sequence.shape}")

    if target_length == 0:
        return np.zeros((0, sequence.shape[1]), dtype=np.float32)

    if sequence.shape[0] == 0:
        return np.zeros((target_length, sequence.shape[1]), dtype=np.float32)

    if sequence.shape[0] == target_length:
        return sequence.astype(np.float32, copy=False)

    indices = np.rint(np.linspace(0, sequence.shape[0] - 1, num=target_length)).astype(int)
    indices = np.clip(indices, 0, sequence.shape[0] - 1)
    return sequence[indices]


def repeat_hcf_vector(hcf_vector, target_length):
    hcf_vector = np.asarray(hcf_vector, dtype=np.float32).reshape(1, -1)

    if target_length == 0:
        return np.zeros((0, hcf_vector.shape[1]), dtype=np.float32)

    return np.repeat(hcf_vector, target_length, axis=0)


def split_feature_store_record(record, context_word_count, target_word_count):
    total_word_count = context_word_count + target_word_count
    if total_word_count <= 0:
        raise ValueError("Target and context cannot both be empty")

    visual = nearest_resample(record["vision"], total_word_count)
    acoustic = nearest_resample(record["audio"], total_word_count)
    hcf = np.asarray(record["hcf"], dtype=np.float32)

    context_visual = visual[:context_word_count]
    target_visual = visual[context_word_count:]
    context_acoustic = acoustic[:context_word_count]
    target_acoustic = acoustic[context_word_count:]
    context_hcf = repeat_hcf_vector(hcf, context_word_count)
    target_hcf = repeat_hcf_vector(hcf, target_word_count)

    return {
        "context_visual": context_visual,
        "target_visual": target_visual,
        "context_acoustic": context_acoustic,
        "target_acoustic": target_acoustic,
        "context_hcf": context_hcf,
        "target_hcf": target_hcf,
    }


def build_hkt_example(sample_id, label, context_sentences, target_text, context_word_count, target_word_count, feature_store_record):
    split_modalities = split_feature_store_record(
        feature_store_record,
        context_word_count=context_word_count,
        target_word_count=target_word_count,
    )

    return (
        (
            target_text,
            split_modalities["target_visual"],
            split_modalities["target_acoustic"],
            split_modalities["target_hcf"],
        ),
        (
            list(context_sentences),
            split_modalities["context_visual"],
            split_modalities["context_acoustic"],
            split_modalities["context_hcf"],
        ),
        str(sample_id),
        float(label),
    )


def build_stratified_split_ids(sample_ids, labels, seed):
    sample_ids = np.asarray(sample_ids)
    labels = np.asarray(labels)

    if len(sample_ids) < 3 or len(set(labels.tolist())) < 2:
        shuffled_indices = np.arange(len(sample_ids))
        rng = np.random.default_rng(seed)
        rng.shuffle(shuffled_indices)

        dev_size = max(1, len(shuffled_indices) // 10)
        test_size = max(1, len(shuffled_indices) // 10)
        train_size = max(1, len(shuffled_indices) - dev_size - test_size)

        train_indices = shuffled_indices[:train_size]
        dev_indices = shuffled_indices[train_size:train_size + dev_size]
        test_indices = shuffled_indices[train_size + dev_size:]
    else:
        first_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_indices, temp_indices = next(first_split.split(sample_ids, labels))

        temp_labels = labels[temp_indices]
        if len(temp_indices) < 2 or len(set(temp_labels.tolist())) < 2:
            midpoint = len(temp_indices) // 2
            dev_indices = temp_indices[:midpoint]
            test_indices = temp_indices[midpoint:]
        else:
            second_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
            dev_relative, test_relative = next(second_split.split(sample_ids[temp_indices], temp_labels))
            dev_indices = temp_indices[dev_relative]
            test_indices = temp_indices[test_relative]

    return {
        "train": sample_ids[train_indices].tolist(),
        "dev": sample_ids[dev_indices].tolist(),
        "test": sample_ids[test_indices].tolist(),
    }


def rebuild_mustard_hkt_dataset(repo_root=REPO_ROOT, seed=5149, metadata_path=DEFAULT_MUSTARD_METADATA_PATH):
    feature_store = load_pickle(os.path.join(repo_root, "datasets", "mustard.pkl"))
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    split_ids = extract_split_ids_from_hkt_pickle(get_official_hkt_path(repo_root, "mustard"))
    split_source = "official_hkt_pickle" if split_ids is not None else "seeded_stratified_split"

    available_ids = sorted(sample_id for sample_id in metadata if sample_id in feature_store)
    if split_ids is None:
        labels = [int(bool(metadata[sample_id].get("sarcasm", 0))) for sample_id in available_ids]
        split_ids = build_stratified_split_ids(available_ids, labels, seed)

    split_payload = {}
    for split_name, sample_ids in split_ids.items():
        examples = []
        for sample_id in sample_ids:
            sample_id = str(sample_id)
            if sample_id not in metadata or sample_id not in feature_store:
                continue

            record = metadata[sample_id]
            context_sentences = record.get("context") or []
            target_text = record.get("utterance", "")
            label = int(bool(record.get("sarcasm", 0)))

            context_word_count = sum(len(heuristic_word_tokenize(sentence)) for sentence in context_sentences)
            target_word_count = len(heuristic_word_tokenize(target_text))
            if target_word_count == 0:
                continue

            examples.append(
                build_hkt_example(
                    sample_id=sample_id,
                    label=label,
                    context_sentences=context_sentences,
                    target_text=target_text,
                    context_word_count=context_word_count,
                    target_word_count=target_word_count,
                    feature_store_record=feature_store[sample_id],
                )
            )

        split_payload[split_name] = examples

    return split_payload, split_source


def _build_examples_from_feature_store(metadata, feature_store):
    examples = []
    for sample_id in sorted(metadata.keys()):
        sample_id = str(sample_id)
        if sample_id not in feature_store:
            continue

        record = metadata[sample_id]
        context_sentences = record.get("context") or []
        target_text = record.get("utterance", "")
        label = int(bool(record.get("sarcasm", 0)))
        speaker = str(record.get("speaker", "UNKNOWN"))

        context_word_count = sum(len(heuristic_word_tokenize(sentence)) for sentence in context_sentences)
        target_word_count = len(heuristic_word_tokenize(target_text))
        if target_word_count == 0:
            continue

        hkt_example = build_hkt_example(
            sample_id=sample_id,
            label=label,
            context_sentences=context_sentences,
            target_text=target_text,
            context_word_count=context_word_count,
            target_word_count=target_word_count,
            feature_store_record=feature_store[sample_id],
        )
        examples.append({"example": hkt_example, "label": label, "speaker": speaker, "sample_id": sample_id})
    return examples


def _collect_examples_from_hkt_split(hkt_payload, metadata):
    """Pool train+dev+test from an HKT 4-tuple pickle and attach speakers."""
    entries = []
    for split_name in ("train", "dev", "test"):
        split_examples = hkt_payload.get(split_name) or []
        for example in split_examples:
            sample_id = str(example[2])
            label = int(bool(example[3]))
            speaker = str(metadata.get(sample_id, {}).get("speaker", "UNKNOWN"))
            entries.append({"example": example, "label": label, "speaker": speaker, "sample_id": sample_id})
    return entries


def _collect_mustard_entries(repo_root, metadata_path):
    """Return list of entries {example, label, speaker, sample_id} for MUStARD.

    Priority order:
    1. Pooled HKT 4-tuple pickle at ``datasets/hkt/mustard.pkl`` (matches
       matalvepu/HKT's dataset format).
    2. Pooled HKT 4-tuple pickle at ``datasets/mustard.pkl`` when that file is
       itself in 4-tuple split form rather than the legacy feature store dict.
    3. Raw feature store ``datasets/mustard.pkl`` combined with ``sarcasm_data.json``
       via ``_build_examples_from_feature_store``.
    """
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    candidate_hkt_paths = [
        os.path.join(repo_root, "datasets", "hkt", "mustard.pkl"),
        os.path.join(repo_root, "datasets", "mustard.pkl"),
    ]
    for candidate in candidate_hkt_paths:
        payload = load_hkt_pickle_if_valid(candidate)
        if payload is not None:
            entries = _collect_examples_from_hkt_split(payload, metadata)
            if entries:
                return entries, f"hkt_pickle:{candidate}", metadata

    feature_store = load_pickle(os.path.join(repo_root, "datasets", "mustard.pkl"))
    entries = _build_examples_from_feature_store(metadata, feature_store)
    return entries, "feature_store", metadata


def build_mustard_speaker_independent_folds(entries, num_folds=MUSTARD_NUM_FOLDS, seed=5149):
    """Stratified K-fold grouped by speaker.

    Every speaker group lives in exactly one fold (test = that fold). Labels
    are kept roughly balanced inside each fold via
    ``sklearn.model_selection.StratifiedGroupKFold``.
    """
    if not entries:
        raise RuntimeError("No MUStARD examples available for split; check feature store and metadata")

    labels = np.array([entry["label"] for entry in entries], dtype=np.int64)
    groups = np.array([entry["speaker"] for entry in entries])
    splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = []
    for train_indices, test_indices in splitter.split(np.zeros(len(entries)), labels, groups):
        folds.append((train_indices.tolist(), test_indices.tolist()))
    return folds


def rebuild_mustard_hkt_dataset_5fold(
    fold_index,
    repo_root=REPO_ROOT,
    metadata_path=DEFAULT_MUSTARD_METADATA_PATH,
    seed=5149,
    dev_ratio=0.1,
    num_folds=MUSTARD_NUM_FOLDS,
):
    """Speaker-independent k-fold CV payload for MUStARD.

    The returned payload mirrors the HKT 4-tuple schema used by
    ``rebuild_mustard_hkt_dataset``: ``{"train": [...], "dev": [...], "test": [...]}``.
    Fold ``fold_index`` (0..num_folds-1) provides the test split; the remaining
    folds are further shuffled and carved into train/dev via a stratified
    hold-out (``dev_ratio``) that respects the sarcasm label distribution.
    """
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(f"fold_index must be in [0,{num_folds}); got {fold_index}")

    entries, source, _metadata = _collect_mustard_entries(repo_root, metadata_path)
    folds = build_mustard_speaker_independent_folds(entries, num_folds=num_folds, seed=seed)
    train_indices, test_indices = folds[fold_index]

    train_entries = [entries[i] for i in train_indices]
    test_entries = [entries[i] for i in test_indices]

    train_labels = np.array([entry["label"] for entry in train_entries], dtype=np.int64)
    dev_fraction = max(0.0, min(0.5, float(dev_ratio)))
    if dev_fraction == 0.0 or len(train_entries) < 4 or len(set(train_labels.tolist())) < 2:
        dev_entries = []
        final_train_entries = train_entries
    else:
        holdout = StratifiedShuffleSplit(n_splits=1, test_size=dev_fraction, random_state=seed + fold_index)
        rel_train, rel_dev = next(holdout.split(np.zeros(len(train_entries)), train_labels))
        final_train_entries = [train_entries[i] for i in rel_train]
        dev_entries = [train_entries[i] for i in rel_dev]

    split_payload = {
        "train": [entry["example"] for entry in final_train_entries],
        "dev": [entry["example"] for entry in dev_entries],
        "test": [entry["example"] for entry in test_entries],
    }
    split_source = f"mustard_si_kfold(k={num_folds},fold={fold_index},seed={seed},source={source})"
    return split_payload, split_source


def rebuild_urfunny_hkt_dataset(repo_root=REPO_ROOT, archive_path=DEFAULT_URFUNNY_ARCHIVE_PATH):
    feature_store = load_pickle(os.path.join(repo_root, "datasets", "urfunny.pkl"))

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("data_folds.pkl") as handle:
            split_ids = pickle.load(handle)
        with archive.open("language_sdk.pkl") as handle:
            language_sdk = pickle.load(handle)
        with archive.open("humor_label_sdk.pkl") as handle:
            humor_labels = pickle.load(handle)

    split_payload = {}
    for split_name in ("train", "dev", "test"):
        examples = []
        for sample_id in split_ids[split_name]:
            feature_key = str(sample_id)
            if feature_key not in feature_store or sample_id not in language_sdk or sample_id not in humor_labels:
                continue

            record = language_sdk[sample_id]
            context_sentences = record.get("context_sentences") or []
            target_text = record.get("punchline_sentence", "")
            context_words = flatten_sentence_word_lists(record.get("context_features") or [])
            target_words = list(record.get("punchline_features") or [])

            if not target_words:
                target_words = heuristic_word_tokenize(target_text)
            if not context_words:
                context_words = flatten_sentence_word_lists(
                    [heuristic_word_tokenize(sentence) for sentence in context_sentences]
                )
            if not target_words:
                continue

            examples.append(
                build_hkt_example(
                    sample_id=sample_id,
                    label=int(bool(humor_labels[sample_id])),
                    context_sentences=context_sentences,
                    target_text=target_text,
                    context_word_count=len(context_words),
                    target_word_count=len(target_words),
                    feature_store_record=feature_store[feature_key],
                )
            )

        split_payload[split_name] = examples

    return split_payload, "official_urfunny_folds"


def load_hkt_dataset(dataset_name, repo_root=REPO_ROOT, seed=5149, cache_path="", fold=None, dev_ratio=0.1):
    """Load a HKT-format split payload.

    Args:
        dataset_name: one of ``{mustard, urfunny, sarcasm, humor, ur_funny}``.
        repo_root: repo root used to locate ``datasets/`` contents.
        seed: base seed for stochastic splits (stratified hold-out, SI k-fold).
        cache_path: optional path to a preprocessed 4-tuple pickle; takes
            priority over everything else.
        fold: when ``dataset_name == mustard`` and no official/cache pickle is
            found, select fold ``k`` (0..MUSTARD_NUM_FOLDS-1) from the
            speaker-independent k-fold split.
        dev_ratio: size of the dev hold-out carved out of the training folds
            for MUStARD k-fold splits.
    """
    dataset_name = normalize_dataset_name(dataset_name)

    if cache_path:
        cached_payload = load_hkt_pickle_if_valid(cache_path)
        if cached_payload is None:
            raise ValueError(f"Invalid HKT dataset cache: {cache_path}")
        return cached_payload, f"user_cache:{cache_path}"

    if dataset_name == "mustard" and fold is not None:
        return rebuild_mustard_hkt_dataset_5fold(
            fold_index=int(fold),
            repo_root=repo_root,
            seed=seed,
            dev_ratio=dev_ratio,
        )

    for candidate in get_candidate_hkt_paths(repo_root, dataset_name):
        official_payload = load_hkt_pickle_if_valid(candidate)
        if official_payload is not None:
            return official_payload, f"hkt_pickle:{candidate}"

    if dataset_name == "mustard":
        return rebuild_mustard_hkt_dataset(repo_root=repo_root, seed=seed)
    if dataset_name == "urfunny":
        return rebuild_urfunny_hkt_dataset(repo_root=repo_root)

    raise ValueError(f"Unsupported HKT dataset: {dataset_name}")
