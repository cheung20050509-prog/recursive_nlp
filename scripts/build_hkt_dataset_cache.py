import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hkt_data import get_hkt_filename, load_hkt_dataset, normalize_dataset_name, save_pickle


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mustard", "urfunny", "sarcasm", "humor"], required=True)
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--seed", type=int, default=5149)
args = parser.parse_args()

dataset_name = normalize_dataset_name(args.dataset)
default_output_path = os.path.join(REPO_ROOT, "datasets", "hkt_rebuilt", get_hkt_filename(dataset_name))
output_path = args.output_path or default_output_path

payload, source = load_hkt_dataset(dataset_name, repo_root=REPO_ROOT, seed=args.seed)
save_pickle(output_path, payload)

print(
    "Wrote {} from {} | train={} dev={} test={}".format(
        output_path,
        source,
        len(payload["train"]),
        len(payload["dev"]),
        len(payload["test"]),
    )
)
