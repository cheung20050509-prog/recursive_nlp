import os
import torch

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Text sequence hidden size: 768 for both DeBERTa-v3-base and ALBERT-base (HKT)
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
        # CH-SIMSv2 (MMSA processed): placeholder dims; always overridden at load from the
        # first training example (see ``apply_simsv2_runtime_dims``), e.g. 25 x 177.
        "simsv2": {"ACOUSTIC_DIM": 33, "VISUAL_DIM": 709, "TEXT_DIM": 768},
    }

    config = dataset_configs.get(dataset_name)
    if config:
        ACOUSTIC_DIM = config["ACOUSTIC_DIM"]
        VISUAL_DIM = config["VISUAL_DIM"]
        TEXT_DIM = config["TEXT_DIM"]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")


def apply_simsv2_runtime_dims(acoustic_dim: int, visual_dim: int):
    """Override ``ACOUSTIC_DIM`` / ``VISUAL_DIM`` after inspecting ``datasets/simsv2.pkl``."""
    global ACOUSTIC_DIM, VISUAL_DIM
    ACOUSTIC_DIM = int(acoustic_dim)
    VISUAL_DIM = int(visual_dim)

