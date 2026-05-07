#!/usr/bin/env python3
"""Export ``datasets/ur_funny.pkl`` from UR-FUNNY SDK + local feature store.

Bypasses any existing HKT pickle on disk: calls ``rebuild_urfunny_hkt_dataset`` directly
so punchline ``p_words`` are **official word lists** from ``language_sdk`` ``punchline_features``
(aligned with multimodal features), not heuristic tokenization of a single string.

**Prerequisites**

- ``datasets/urfunny.pkl`` — id → ``{vision, audio, hcf}`` feature store (same as HKT pipeline).
- UR-FUNNY v2 zip (default: ``DEFAULT_URFUNNY_ARCHIVE_PATH`` in ``hkt_data``) containing
  ``data_folds.pkl``, ``language_sdk.pkl``, ``humor_label_sdk.pkl``.

**After export**

Rebuild silver spans so word boundaries match::

    python scripts/build_hkt_silver_span_cache.py --dataset urfunny --overwrite

Then train / Optuna as usual with ``--dataset urfunny``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hkt_data import (  # noqa: E402
    DEFAULT_URFUNNY_ARCHIVE_PATH,
    get_hkt_filename,
    rebuild_urfunny_hkt_dataset,
    save_pickle,
)


def _p_field_stats(payload: dict) -> tuple[int, int, int]:
    """Counts (list_p, str_p, total) over train+dev+test."""
    n_list, n_str, n_total = 0, 0, 0
    for split in ("train", "dev", "test"):
        for ex in payload.get(split) or []:
            p0 = ex[0][0]
            n_total += 1
            if isinstance(p0, list):
                n_list += 1
            elif isinstance(p0, str):
                n_str += 1
    return n_list, n_str, n_total


def main():
    parser = argparse.ArgumentParser(description="Export ur_funny HKT pickle from UR-FUNNY SDK (official word lists).")
    parser.add_argument("--repo-root", type=str, default=str(REPO_ROOT))
    parser.add_argument(
        "--archive",
        type=str,
        default=DEFAULT_URFUNNY_ARCHIVE_PATH,
        help="Path to urfunny_v2_features.zip (or equivalent).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output pickle path (default: <repo-root>/datasets/ur_funny.pkl).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="If output exists, copy it to output.<timestamp>.bak before overwriting.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output) if args.output else repo_root / "datasets" / get_hkt_filename("urfunny")
    output_path = output_path.resolve()

    if output_path.exists() and args.backup:
        bak = output_path.with_name(output_path.name + f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
        shutil.copy2(output_path, bak)
        print(f"Backed up existing file to {bak}", flush=True)

    payload, split_source = rebuild_urfunny_hkt_dataset(
        repo_root=str(repo_root),
        archive_path=args.archive,
    )
    save_pickle(str(output_path), payload)

    n_list, n_str, n_total = _p_field_stats(payload)
    print(
        f"Wrote {output_path} | source={split_source} | "
        f"train={len(payload['train'])} dev={len(payload['dev'])} test={len(payload['test'])}",
        flush=True,
    )
    print(
        f"p_words layout: list={n_list} str={n_str} total={n_total} "
        f"(UR-FUNNY SDK rebuild expects list for samples with punchline_features)",
        flush=True,
    )


if __name__ == "__main__":
    main()
