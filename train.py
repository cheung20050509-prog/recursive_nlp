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
parser.add_argument('--syntax_temperature', default=1.0, type=float)
parser.add_argument('--merge_trace_samples', default=3, type=int)
parser.add_argument('--max_grad_norm', default=1.0, type=float)

args = parser.parse_args()

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM,
                                      global_configs.TEXT_DIM)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def safe_min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    tensor_min = tensor.amin()
    tensor_range = (tensor.amax() - tensor_min).clamp_min(1e-6)
    return (tensor - tensor_min) / tensor_range


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []

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
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

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


def get_appropriate_dataset(data):
    tokenizer = get_tokenizer(args.model)

    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

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
    recursion_steps_total = 0.0
    max_depth_hits = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    regression_loss_fct = MSELoss()
    acc7_loss_fct = CrossEntropyLoss()
    encountered_nonfinite = False
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        visual_norm = safe_min_max_normalize(visual)
        acoustic_norm = safe_min_max_normalize(acoustic)
        logits, acc7_logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps = model(
            input_ids,
            visual_norm,
            acoustic_norm,
        )
        regression_loss = regression_loss_fct(logits.view(-1), label_ids.view(-1))
        acc7_targets = build_acc7_targets(label_ids.view(-1))
        acc7_loss = acc7_loss_fct(acc7_logits.view(-1, 7), acc7_targets)
        loss = regression_loss + args.acc7_loss_weight * acc7_loss + 2 / (args.p_beta + args.p_gamma) * IB_loss

        if not torch.isfinite(loss):
            print(
                "NONFINITE_TRAIN: step:{}, regression_loss:{}, acc7_loss:{}, ib_loss:{}".format(
                    step + 1,
                    regression_loss.detach().item(),
                    acc7_loss.detach().item(),
                    IB_loss.detach().item(),
                )
            )
            encountered_nonfinite = True
            break

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
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
        return float("nan"), 0.0, 0.0, encountered_nonfinite

    return tr_loss / nb_tr_steps, recursion_steps_total / nb_tr_examples, max_depth_hits / nb_tr_examples, encountered_nonfinite


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    recursion_steps_total = 0.0
    max_depth_hits = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    regression_loss_fct = MSELoss()
    acc7_loss_fct = CrossEntropyLoss()
    encountered_nonfinite = False
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = safe_min_max_normalize(visual)
            acoustic_norm = safe_min_max_normalize(acoustic)

            logits, acc7_logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps = model(
                input_ids,
                visual_norm,
                acoustic_norm,
            )
            regression_loss = regression_loss_fct(logits.view(-1), label_ids.view(-1))
            acc7_targets = build_acc7_targets(label_ids.view(-1))
            acc7_loss = acc7_loss_fct(acc7_logits.view(-1, 7), acc7_targets)
            loss = regression_loss + args.acc7_loss_weight * acc7_loss

            if not torch.isfinite(loss):
                print(
                    "NONFINITE_VALID: step:{}, regression_loss:{}, acc7_loss:{}".format(
                        step + 1,
                        regression_loss.detach().item(),
                        acc7_loss.detach().item(),
                    )
                )
                encountered_nonfinite = True
                break

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            recursion_steps_total += recursive_steps.float().sum().item()
            max_depth_hits += (recursive_steps == global_configs.MAX_RECURSION_DEPTH).sum().item()
            nb_dev_examples += recursive_steps.size(0)
            nb_dev_steps += 1

    if nb_dev_steps == 0 or nb_dev_examples == 0:
        return float("nan"), 0.0, 0.0, encountered_nonfinite

    return dev_loss / nb_dev_steps, recursion_steps_total / nb_dev_examples, max_depth_hits / nb_dev_examples, encountered_nonfinite


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    recursion_steps_total = 0.0
    max_depth_hits = 0
    sample_count = 0
    syntax_samples = []
    tokenizer = get_tokenizer(args.model) if args.merge_trace_samples > 0 else None
    pad_token_id = tokenizer.pad_token_id if tokenizer is not None and tokenizer.pad_token_id is not None else 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = safe_min_max_normalize(visual)
            acoustic_norm = safe_min_max_normalize(acoustic)

            model_outputs = model(
                input_ids,
                visual_norm,
                acoustic_norm,
                return_syntax_info=tokenizer is not None,
            )

            if tokenizer is not None:
                logits, acc7_logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps, syntax_info = model_outputs
            else:
                logits, acc7_logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, recursive_steps = model_outputs
                syntax_info = None

            recursion_steps_total += recursive_steps.float().sum().item()
            max_depth_hits += (recursive_steps == global_configs.MAX_RECURSION_DEPTH).sum().item()
            sample_count += recursive_steps.size(0)

            if syntax_info is not None and len(syntax_samples) < args.merge_trace_samples:
                sample_logits = logits.view(-1).detach().cpu()
                sample_labels = label_ids.view(-1).detach().cpu()
                sample_input_ids = input_ids.detach().cpu()
                remaining_slots = args.merge_trace_samples - len(syntax_samples)

                for sample_idx in range(min(remaining_slots, sample_input_ids.size(0))):
                    valid_ids = sample_input_ids[sample_idx][sample_input_ids[sample_idx].ne(pad_token_id)].tolist()
                    syntax_samples.append({
                        "prediction": float(sample_logits[sample_idx].item()),
                        "label": float(sample_labels[sample_idx].item()),
                        "tokens": tokenizer.convert_ids_to_tokens(valid_ids),
                        "merge_trace": syntax_info["merge_traces"][sample_idx],
                        "syntax_tree": syntax_info["syntax_trees"][sample_idx],
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

    return preds, labels, avg_recursion_steps, max_depth_hit_rate, syntax_samples


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


def print_merge_trace_samples(syntax_samples):
    for sample_idx, sample in enumerate(syntax_samples, start=1):
        print(
            "MERGE_TRACE[{}]: pred:{:.4f}, label:{:.4f}, trace:{}, tree:{}, tokens:{}".format(
                sample_idx,
                sample["prediction"],
                sample["label"],
                sample["merge_trace"],
                sample["syntax_tree"],
                " ".join(sample["tokens"]),
            )
        )


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False, return_acc7=False):
    preds, y_test, avg_recursion_steps, max_depth_hit_rate, syntax_samples = test_epoch(model, test_dataloader)
    acc7 = acc7_score(preds, y_test)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)

    if return_acc7:
        return acc, acc7, mae, corr, f_score, avg_recursion_steps, max_depth_hit_rate, syntax_samples

    return acc, mae, corr, f_score, avg_recursion_steps, max_depth_hit_rate, syntax_samples


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

    for epoch_i in range(int(args.n_epochs)):
        train_loss, train_avg_steps, train_hit_rate, train_nonfinite = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss, valid_avg_steps, valid_hit_rate, valid_nonfinite = eval_epoch(model, validation_dataloader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch_i + 1
            best_train_loss = train_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_train_avg_steps = train_avg_steps
            best_valid_avg_steps = valid_avg_steps
            best_train_hit_rate = train_hit_rate
            best_valid_hit_rate = valid_hit_rate

        print(
            "TRAIN: epoch:{}, train_loss:{}, valid_loss:{}, train_avg_steps:{}, valid_avg_steps:{}, train_hit_max_depth_rate:{}, valid_hit_max_depth_rate:{}".format(
                epoch_i + 1, train_loss, valid_loss, train_avg_steps, valid_avg_steps, train_hit_rate, valid_hit_rate
            )
        )

        if train_nonfinite or valid_nonfinite:
            print("EARLY_STOP: non-finite loss encountered at epoch:{}".format(epoch_i + 1))
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_acc, test_acc7, test_mae, test_corr, test_f_score, test_avg_steps, test_hit_rate, syntax_samples = test_score_model(
        model, test_data_loader, return_acc7=True
    )
    print(
        "TEST: best_epoch:{}, train_loss:{}, valid_loss:{}, train_avg_steps:{}, valid_avg_steps:{}, train_hit_max_depth_rate:{}, valid_hit_max_depth_rate:{}, test_acc:{}, acc7:{}, mae:{}, corr:{}, f1_score:{}, test_avg_steps:{}, test_hit_max_depth_rate:{}".format(
            best_epoch, best_train_loss, best_valid_loss, best_train_avg_steps, best_valid_avg_steps,
            best_train_hit_rate, best_valid_hit_rate, test_acc, test_acc7, test_mae, test_corr, test_f_score,
            test_avg_steps, test_hit_rate
        )
    )
    print_merge_trace_samples(syntax_samples)
    return best_train_loss, best_valid_loss, test_acc, test_mae, test_corr, test_f_score


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
