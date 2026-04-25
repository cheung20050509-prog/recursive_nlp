import os
import torch

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEXT_DIM = 0
ACOUSTIC_DIM = 0
VISUAL_DIM = 0
MAX_RECURSION_DEPTH = 3
DEVICE = torch.device("cuda:0")

def set_dataset_config(dataset_name):
    global TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM

    dataset_configs = {
        "mosi": {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 47, "TEXT_DIM": 768},
        "mosei": {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 35, "TEXT_DIM": 768},
        "mustard": {"ACOUSTIC_DIM": 60, "VISUAL_DIM": 36, "TEXT_DIM": 768},
        "sarcasm": {"ACOUSTIC_DIM": 60, "VISUAL_DIM": 36, "TEXT_DIM": 768},
        "urfunny": {"ACOUSTIC_DIM": 60, "VISUAL_DIM": 36, "TEXT_DIM": 768},
        "humor": {"ACOUSTIC_DIM": 60, "VISUAL_DIM": 36, "TEXT_DIM": 768},
        # Full-dim variants mirror the GitHub MHD_MSD_optuna branch
        # (global_configs: mustard/ur_funny -> ACOUSTIC=81, VISUAL=91).
        # They keep every HKT feature column (no 0:60 / 55:91 subset) so the
        # recursive-ITHP forward matches the InfoGate baseline 1:1 on dims.
        "mustard_full": {"ACOUSTIC_DIM": 81, "VISUAL_DIM": 91, "TEXT_DIM": 768},
        "sarcasm_full": {"ACOUSTIC_DIM": 81, "VISUAL_DIM": 91, "TEXT_DIM": 768},
        "urfunny_full": {"ACOUSTIC_DIM": 81, "VISUAL_DIM": 91, "TEXT_DIM": 768},
        "humor_full": {"ACOUSTIC_DIM": 81, "VISUAL_DIM": 91, "TEXT_DIM": 768},
    }

    config = dataset_configs.get(dataset_name)
    if config:
        ACOUSTIC_DIM = config["ACOUSTIC_DIM"]
        VISUAL_DIM = config["VISUAL_DIM"]
        TEXT_DIM = config["TEXT_DIM"]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

