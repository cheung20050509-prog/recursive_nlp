import argparse
import os
import random
import pickle
import numpy as np
import copy

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.nn import MSELoss, CrossEntropyLoss

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from deberta_ITHP import ITHP_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/root/autodl-tmp/recursive_language/deberta-v3-base", )
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=128)
parser.add_argument('--inter_dim', default=256, help='dimension of inter layers', type=int)
parser.add_argument("--drop_prob", help='drop probability for dropout -- encoder', default=0.3,
                    type=float)  # Dropout for ITHP
parser.add_argument('--p_lambda', default=0.3, help='coefficient -- lambda', type=float)  # For IB2
parser.add_argument('--p_beta', default=8, help='coefficient -- beta', type=float)  # For IB1
parser.add_argument('--p_gamma', default=32, help='coefficient -- gamma', type=float)
parser.add_argument('--beta_shift', default=1.0, help='coefficient -- shift', type=float)
parser.add_argument('--IB_coef', default=10, type=float)
parser.add_argument('--B0_dim', default=128, type=int)
parser.add_argument('--B1_dim', default=64, type=int)
parser.add_argument('--halting_threshold', default=0.0285, type=float)
parser.add_argument('--acc7_loss_weight', default=0.2, type=float)
parser.add_argument('--silver_span_cache', default='', type=str)
parser.add_argument('--silver_span_loss_weight', default=0.1, type=float)
parser.add_argument('--syntax_temperature', default=1.0, type=float)
parser.add_argument('--merge_trace_samples', default=3, type=int)
parser.add_argument('--max_grad_norm', default=1.0, type=float)

args = parser.parse_args()

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM,
                                      global_configs.TEXT_DIM)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id, syntax_span_mask):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.syntax_span_mask = syntax_span_mask


def safe_min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    tensor_min = tensor.amin()
    tensor_range = (tensor.amax() - tensor_min).clamp_min(1e-6)
    return (tensor - tensor_min) / tensor_range


TREE_SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]"}
CONTRACTION_SUFFIXES = {"s", "t", "re", "ve", "m", "ll", "d"}


def render_token_phrase(tokens, include_special_tokens=False):
    words = []
    current_word = ""

    for token in tokens:
        if token in TREE_SPECIAL_TOKENS:
            if current_word:
                words.append(current_word)
                current_word = ""
            if include_special_tokens:
                words.append(token)
            continue

        if token.startswith("▁"):
            if current_word:
                words.append(current_word)
            current_word = token[1:]
            continue

        if not current_word:
            current_word = token
            continue

        if token in CONTRACTION_SUFFIXES and current_word[-1].isalnum():
            current_word = current_word + "'" + token
        else:
            current_word = current_word + token

    if current_word:
        words.append(current_word)

    if not words and not include_special_tokens:
        return render_token_phrase(tokens, include_special_tokens=True)

    return " ".join(words)


def parse_syntax_tree(tree_text):
    if tree_text is None:
        return None

    tree_tokens = tree_text.replace("(", " ( ").replace(")", " ) ").split()
    if not tree_tokens:
        return None

    def parse_node(cursor):
        token = tree_tokens[cursor]
        if token != "(":
            return int(token), cursor + 1

        left_node, cursor = parse_node(cursor + 1)
        right_node, cursor = parse_node(cursor)
        if tree_tokens[cursor] != ")":
            raise ValueError("Malformed syntax tree: missing closing parenthesis")
        return (left_node, right_node), cursor + 1

    syntax_tree, next_cursor = parse_node(0)
    if next_cursor != len(tree_tokens):
        raise ValueError("Malformed syntax tree: trailing tokens remain after parse")
    return syntax_tree


def build_readable_tree_views(tree_text, tokens):
    syntax_tree = parse_syntax_tree(tree_text)
    if syntax_tree is None or not tokens:
        return None, None

    def render_node(node):
        if isinstance(node, int):
            start = node
            end = node + 1
            span_tree = f"[{start}:{end}]"
            text_tree = render_token_phrase(tokens[start:end], include_special_tokens=True)
            return start, end, span_tree, text_tree

        left_start, left_end, left_span_tree, left_text_tree = render_node(node[0])
        right_start, right_end, right_span_tree, right_text_tree = render_node(node[1])
        start = left_start
        end = right_end
        span_tree = f"([{start}:{end}] {left_span_tree} {right_span_tree})"
        phrase_text = render_token_phrase(tokens[start:end], include_special_tokens=False)
        text_tree = f"({phrase_text}: {left_text_tree} {right_text_tree})"
        return start, end, span_tree, text_tree

    _, _, span_tree, text_tree = render_node(syntax_tree)
    return span_tree, text_tree


def resolve_silver_span_cache_path():
    if args.silver_span_cache:
        return args.silver_span_cache

    default_cache_path = os.path.join("datasets", f"{args.dataset}_silver_spans.pkl")
    if os.path.exists(default_cache_path):
        return default_cache_path

    return ""


def load_silver_span_cache():
    cache_path = resolve_silver_span_cache_path()
    if not cache_path:
        print("SILVER_SPAN_CACHE: disabled")
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
            split_records = list(split_records.values())

        nonempty = sum(1 for record in split_records if record.get("word_spans") or record.get("spans"))
        parse_errors = sum(1 for record in split_records if record.get("parse_error"))
        print(
            f"SILVER_SPAN_CACHE[{split_name}]: total={len(split_records)}, nonempty={nonempty}, parse_errors={parse_errors}"
        )
        if len(split_records) > 0 and nonempty == 0 and parse_errors > 0:
            raise ValueError(
                f"Silver span cache '{cache_path}' has no usable spans for split '{split_name}'. "
                "Regenerate the cache before training."
            )

    print(f"SILVER_SPAN_CACHE: loaded {cache_path}")
    return cache


def get_silver_span_record(split_records, example_index, words, segment):
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


def build_silver_span_mask(silver_record, inversions, max_seq_length):
    span_mask = np.zeros((max_seq_length, max_seq_length), dtype=np.float32)
    if silver_record is None or len(inversions) == 0:
        return span_mask

    word_spans = silver_record.get("word_spans") or silver_record.get("spans") or []
    if not word_spans:
        return span_mask

    word_to_token_start = {}
    word_to_token_end = {}
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

        token_start = word_to_token_start[word_start] + 1
        token_end = word_to_token_end[word_end - 1] + 1

        if token_end - token_start < 2:
            continue

        if token_start >= max_seq_length or token_end - 1 >= max_seq_length:
            continue

        span_mask[token_start, token_end - 1] = 1.0

    return span_mask


def convert_to_features(examples, max_seq_length, tokenizer, silver_records=None):
    features = []

    if isinstance(silver_records, list) and len(silver_records) != len(examples):
        raise ValueError(
            f"Silver span cache size mismatch: dataset={len(examples)}, cache={len(silver_records)}"
        )

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            inversions = inversions[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        silver_record = get_silver_span_record(silver_records, ex_index, words, segment) if silver_records is not None else None
        syntax_span_mask = build_silver_span_mask(silver_record, inversions, max_seq_length)

        prepare_input = prepare_deberta_input

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
                syntax_span_mask=syntax_span_mask,
            )
        )
    return features


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)


def get_appropriate_dataset(data, silver_records=None):
    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer, silver_records=silver_records)
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)
    all_syntax_span_masks = torch.tensor(np.array([f.syntax_span_mask for f in features]), dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_label_ids,
        all_syntax_span_masks,
    )
    return dataset


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    silver_span_cache = load_silver_span_cache()

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(
        train_data,
        silver_records=silver_span_cache["train"] if silver_span_cache is not None else None,
    )
    dev_dataset = get_appropriate_dataset(
        dev_data,
        silver_records=silver_span_cache["dev"] if silver_span_cache is not None else None,
    )
    test_dataset = get_appropriate_dataset(
        test_data,
        silver_records=silver_span_cache["test"] if silver_span_cache is not None else None,
    )

    num_train_optimization_steps = (
            int(
                len(train_dataset) / args.train_batch_size /
                args.gradient_accumulation_step
            )
            * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Seed: {}".format(seed))


def prep_for_training(num_train_optimization_steps: int):
    model = ITHP_DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1,
    )

    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    syntax_loss_total = 0.0
    recursion_steps_total = 0.0
    max_depth_hits = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    regression_loss_fct = MSELoss()
    acc7_loss_fct = CrossEntropyLoss()
    encountered_nonfinite = False
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids, syntax_span_masks = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        visual_norm = safe_min_max_normalize(visual)
        acoustic_norm = safe_min_max_normalize(acoustic)
        logits, acc7_logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss = model(
            input_ids,
            visual_norm,
            acoustic_norm,
            syntax_span_masks=syntax_span_masks,
        )
        regression_loss = regression_loss_fct(logits.view(-1), label_ids.view(-1))
        acc7_targets = build_acc7_targets(label_ids.view(-1))
        acc7_loss = acc7_loss_fct(acc7_logits.view(-1, 7), acc7_targets)
        loss = (
            regression_loss
            + args.acc7_loss_weight * acc7_loss
            + args.silver_span_loss_weight * syntax_loss
            + 2 / (args.p_beta + args.p_gamma) * IB_loss
        )

        if not torch.isfinite(loss):
            print(
                "NONFINITE_TRAIN: step:{}, regression_loss:{}, acc7_loss:{}, syntax_loss:{}, ib_loss:{}".format(
                    step + 1,
                    regression_loss.detach().item(),
                    acc7_loss.detach().item(),
                    syntax_loss.detach().item(),
                    IB_loss.detach().item(),
                )
            )
            encountered_nonfinite = True
            break

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        syntax_loss_total += syntax_loss.detach().item()
        recursion_steps_total += recursive_steps.float().sum().item()
        max_depth_hits += (recursive_steps == global_configs.MAX_RECURSION_DEPTH).sum().item()
        nb_tr_examples += recursive_steps.size(0)
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    if nb_tr_steps == 0 or nb_tr_examples == 0:
        return float("nan"), 0.0, 0.0, 0.0, encountered_nonfinite

    return (
        tr_loss / nb_tr_steps,
        recursion_steps_total / nb_tr_examples,
        max_depth_hits / nb_tr_examples,
        syntax_loss_total / nb_tr_steps,
        encountered_nonfinite,
    )


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    syntax_loss_total = 0.0
    recursion_steps_total = 0.0
    max_depth_hits = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    regression_loss_fct = MSELoss()
    acc7_loss_fct = CrossEntropyLoss()
    encountered_nonfinite = False
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids, syntax_span_masks = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = safe_min_max_normalize(visual)
            acoustic_norm = safe_min_max_normalize(acoustic)

            logits, acc7_logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                syntax_span_masks=syntax_span_masks,
            )
            regression_loss = regression_loss_fct(logits.view(-1), label_ids.view(-1))
            acc7_targets = build_acc7_targets(label_ids.view(-1))
            acc7_loss = acc7_loss_fct(acc7_logits.view(-1, 7), acc7_targets)
            loss = regression_loss + args.acc7_loss_weight * acc7_loss

            if not torch.isfinite(loss) or not torch.isfinite(syntax_loss):
                print(
                    "NONFINITE_VALID: step:{}, regression_loss:{}, acc7_loss:{}, syntax_loss:{}".format(
                        step + 1,
                        regression_loss.detach().item(),
                        acc7_loss.detach().item(),
                        syntax_loss.detach().item(),
                    )
                )
                encountered_nonfinite = True
                break

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            syntax_loss_total += syntax_loss.detach().item()
            recursion_steps_total += recursive_steps.float().sum().item()
            max_depth_hits += (recursive_steps == global_configs.MAX_RECURSION_DEPTH).sum().item()
            nb_dev_examples += recursive_steps.size(0)
            nb_dev_steps += 1

    if nb_dev_steps == 0 or nb_dev_examples == 0:
        return float("nan"), 0.0, 0.0, 0.0, encountered_nonfinite

    return (
        dev_loss / nb_dev_steps,
        recursion_steps_total / nb_dev_examples,
        max_depth_hits / nb_dev_examples,
        syntax_loss_total / nb_dev_steps,
        encountered_nonfinite,
    )


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    recursion_steps_total = 0.0
    max_depth_hits = 0
    sample_count = 0
    syntax_samples = []
    syntax_loss_total = 0.0
    syntax_loss_batches = 0
    tokenizer = get_tokenizer(args.model) if args.merge_trace_samples > 0 else None
    pad_token_id = tokenizer.pad_token_id if tokenizer is not None and tokenizer.pad_token_id is not None else 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, label_ids, syntax_span_masks = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = safe_min_max_normalize(visual)
            acoustic_norm = safe_min_max_normalize(acoustic)

            model_outputs = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                syntax_span_masks=syntax_span_masks,
                return_syntax_info=tokenizer is not None,
            )

            if tokenizer is not None:
                logits, acc7_logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss, syntax_info = model_outputs
            else:
                logits, acc7_logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_loss = model_outputs
                syntax_info = None

            recursion_steps_total += recursive_steps.float().sum().item()
            max_depth_hits += (recursive_steps == global_configs.MAX_RECURSION_DEPTH).sum().item()
            sample_count += recursive_steps.size(0)
            syntax_loss_total += syntax_loss.detach().item()
            syntax_loss_batches += 1

            if syntax_info is not None and len(syntax_samples) < args.merge_trace_samples:
                sample_logits = logits.view(-1).detach().cpu()
                sample_labels = label_ids.view(-1).detach().cpu()
                sample_input_ids = input_ids.detach().cpu()
                remaining_slots = args.merge_trace_samples - len(syntax_samples)

                for sample_idx in range(min(remaining_slots, sample_input_ids.size(0))):
                    valid_ids = sample_input_ids[sample_idx][sample_input_ids[sample_idx].ne(pad_token_id)].tolist()
                    token_list = tokenizer.convert_ids_to_tokens(valid_ids)
                    span_tree, text_tree = build_readable_tree_views(
                        syntax_info["syntax_trees"][sample_idx],
                        token_list,
                    )
                    syntax_samples.append({
                        "prediction": float(sample_logits[sample_idx].item()),
                        "label": float(sample_labels[sample_idx].item()),
                        "tokens": token_list,
                        "sentence": render_token_phrase(token_list, include_special_tokens=False),
                        "merge_trace": syntax_info["merge_traces"][sample_idx],
                        "syntax_tree": syntax_info["syntax_trees"][sample_idx],
                        "span_tree": span_tree,
                        "text_tree": text_tree,
                    })

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    avg_recursion_steps = recursion_steps_total / sample_count if sample_count > 0 else 0.0
    max_depth_hit_rate = max_depth_hits / sample_count if sample_count > 0 else 0.0
    avg_syntax_loss = syntax_loss_total / syntax_loss_batches if syntax_loss_batches > 0 else 0.0

    return preds, labels, avg_recursion_steps, max_depth_hit_rate, syntax_samples, avg_syntax_loss


def build_acc7_targets(labels):
    return torch.clamp(torch.round(labels), -3, 3).long() + 3


def multiclass_acc(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    if preds.size == 0:
        return 0.0

    return np.mean(np.round(preds) == np.round(labels))


def acc7_score(preds, labels):
    return multiclass_acc(np.clip(preds, -3.0, 3.0), np.clip(labels, -3.0, 3.0))


def safe_corrcoef(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    if preds.size < 2 or labels.size < 2:
        return 0.0

    corr = np.corrcoef(preds, labels)[0][1]
    if np.isnan(corr):
        return 0.0

    return float(corr)


def print_merge_trace_samples(syntax_samples):
    for sample_idx, sample in enumerate(syntax_samples, start=1):
        print(
            "MERGE_TRACE[{}]: pred:{:.4f}, label:{:.4f}, trace:{}, tree:{}".format(
                sample_idx,
                sample["prediction"],
                sample["label"],
                sample["merge_trace"],
                sample["syntax_tree"],
            )
        )
        print("MERGE_TRACE_SPAN[{}]: {}".format(sample_idx, sample["span_tree"]))
        print("MERGE_TRACE_TEXT[{}]: {}".format(sample_idx, sample["text_tree"]))
        print("MERGE_TRACE_TOKENS[{}]: {}".format(sample_idx, " ".join(sample["tokens"])))
        print("MERGE_TRACE_SENTENCE[{}]: {}".format(sample_idx, sample["sentence"]))


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False, return_acc7=False):
    preds, y_test, avg_recursion_steps, max_depth_hit_rate, syntax_samples, avg_syntax_loss = test_epoch(model, test_dataloader)
    non_zeros = np.array([index for index, label in enumerate(y_test) if label != 0 or use_zero])

    preds_a7 = np.clip(preds, a_min=-3.0, a_max=3.0)
    labels_a7 = np.clip(y_test, a_min=-3.0, a_max=3.0)
    acc7 = multiclass_acc(preds_a7, labels_a7)

    mae = float(np.mean(np.absolute(preds - y_test)))
    corr = safe_corrcoef(preds, y_test)

    preds_zero = preds >= 0
    labels_zero = y_test >= 0
    f1_score_zero = f1_score(labels_zero, preds_zero, average="weighted")
    acc2_zero = accuracy_score(labels_zero, preds_zero)

    preds_no_zero = preds[non_zeros]
    labels_no_zero = y_test[non_zeros]
    if preds_no_zero.size == 0:
        acc2_no_zero = 0.0
        f1_score_no_zero = 0.0
    else:
        preds_no_zero = preds_no_zero > 0
        labels_no_zero = labels_no_zero > 0
        f1_score_no_zero = f1_score(labels_no_zero, preds_no_zero, average="weighted")
        acc2_no_zero = accuracy_score(labels_no_zero, preds_no_zero)

    metrics = {
        "acc7": float(acc7),
        "acc2_zero": float(acc2_zero),
        "f1_score_zero": float(f1_score_zero),
        "acc2_no_zero": float(acc2_no_zero),
        "f1_score_no_zero": float(f1_score_no_zero),
        "mae": mae,
        "corr": corr,
    }

    if return_acc7:
        return metrics, avg_recursion_steps, max_depth_hit_rate, syntax_samples, avg_syntax_loss

    return metrics, avg_recursion_steps, max_depth_hit_rate, syntax_samples, avg_syntax_loss


def train(
        model,
        train_dataloader,
        validation_dataloader,
        test_data_loader,
        optimizer,
        scheduler,
):
    valid_losses = []
    test_accuracies = []
    mae_list = []
    corr_list = []
    f1_list = []
    best_valid_loss = float("inf")
    best_epoch = -1
    best_train_loss = None
    best_state_dict = None
    best_train_avg_steps = None
    best_valid_avg_steps = None
    best_train_hit_rate = None
    best_valid_hit_rate = None
    best_train_syntax_loss = None
    best_valid_syntax_loss = None

    for epoch_i in range(int(args.n_epochs)):
        train_loss, train_avg_steps, train_hit_rate, train_syntax_loss, train_nonfinite = train_epoch(
            model, train_dataloader, optimizer, scheduler
        )
        valid_loss, valid_avg_steps, valid_hit_rate, valid_syntax_loss, valid_nonfinite = eval_epoch(
            model, validation_dataloader
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch_i + 1
            best_train_loss = train_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_train_avg_steps = train_avg_steps
            best_valid_avg_steps = valid_avg_steps
            best_train_hit_rate = train_hit_rate
            best_valid_hit_rate = valid_hit_rate
            best_train_syntax_loss = train_syntax_loss
            best_valid_syntax_loss = valid_syntax_loss

        print(
            "TRAIN: epoch:{}, train_loss:{}, valid_loss:{}, train_syntax_loss:{}, valid_syntax_loss:{}, train_avg_steps:{}, valid_avg_steps:{}, train_hit_max_depth_rate:{}, valid_hit_max_depth_rate:{}".format(
                epoch_i + 1,
                train_loss,
                valid_loss,
                train_syntax_loss,
                valid_syntax_loss,
                train_avg_steps,
                valid_avg_steps,
                train_hit_rate,
                valid_hit_rate,
            )
        )

        if train_nonfinite or valid_nonfinite:
            print("EARLY_STOP: non-finite loss encountered at epoch:{}".format(epoch_i + 1))
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_metrics, test_avg_steps, test_hit_rate, syntax_samples, test_syntax_loss = test_score_model(
        model, test_data_loader, return_acc7=True
    )
    print(
        "TEST: best_epoch:{}, train_loss:{}, valid_loss:{}, train_syntax_loss:{}, valid_syntax_loss:{}, train_avg_steps:{}, valid_avg_steps:{}, train_hit_max_depth_rate:{}, valid_hit_max_depth_rate:{}, test_acc:{}, acc7:{}, acc2_zero:{}, f1_score_zero:{}, acc2_no_zero:{}, f1_score_no_zero:{}, mae:{}, corr:{}, f1_score:{}, test_avg_steps:{}, test_hit_max_depth_rate:{}, test_syntax_loss:{}".format(
            best_epoch,
            best_train_loss,
            best_valid_loss,
            best_train_syntax_loss,
            best_valid_syntax_loss,
            best_train_avg_steps,
            best_valid_avg_steps,
            best_train_hit_rate,
            best_valid_hit_rate,
            test_metrics["acc2_no_zero"],
            test_metrics["acc7"],
            test_metrics["acc2_zero"],
            test_metrics["f1_score_zero"],
            test_metrics["acc2_no_zero"],
            test_metrics["f1_score_no_zero"],
            test_metrics["mae"],
            test_metrics["corr"],
            test_metrics["f1_score_no_zero"],
            test_avg_steps,
            test_hit_rate,
            test_syntax_loss,
        )
    )
    print_merge_trace_samples(syntax_samples)
    return (
        best_train_loss,
        best_valid_loss,
        test_metrics["acc2_no_zero"],
        test_metrics["mae"],
        test_metrics["corr"],
        test_metrics["f1_score_no_zero"],
    )


def main():
    set_random_seed(args.seed)
    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == '__main__':
    main()
