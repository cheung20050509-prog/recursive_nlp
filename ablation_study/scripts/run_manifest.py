#!/usr/bin/env python3
"""Run every entry in manifest.json (fixed vs hkt_urfunny)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run ablation_study manifest entries sequentially.")
    ap.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest.json (default: <ablation_study>/manifest.json)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    ap.add_argument("--start", type=int, default=0, help="Skip first N entries")
    ap.add_argument("--limit", type=int, default=0, help="Run at most N entries (0 = all)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    ablation_root = here.parent
    manifest_path = Path(args.manifest) if args.manifest else ablation_root / "manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        print("manifest must be a JSON array", file=sys.stderr)
        return 2

    slice_entries = entries[args.start :]
    if args.limit and args.limit > 0:
        slice_entries = slice_entries[: args.limit]

    run_fixed = ablation_root.parent / "fixed_experiment" / "run_fixed.py"
    launch_hkt = here / "launch_hkt_json.py"
    py = sys.executable

    for i, row in enumerate(slice_entries):
        rid = row.get("id", f"entry_{i}")
        runner = row.get("runner")
        cfg_rel = row.get("config")
        if not runner or not cfg_rel:
            print(f"skip {rid}: missing runner or config", file=sys.stderr)
            continue
        cfg_path = (ablation_root / cfg_rel).resolve()
        if not cfg_path.is_file():
            print(f"skip {rid}: missing file {cfg_path}", file=sys.stderr)
            continue

        if runner == "fixed":
            cmd = [py, "-u", str(run_fixed), "--config", str(cfg_path)]
        elif runner == "hkt_urfunny":
            cmd = [py, "-u", str(launch_hkt), str(cfg_path)]
        else:
            print(f"skip {rid}: unknown runner {runner}", file=sys.stderr)
            continue

        if args.dry_run:
            print(subprocess.list2cmdline(cmd))
            continue

        print(f"\n=== [{rid}] runner={runner} ===\n", flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"manifest stopped: {rid} exit {proc.returncode}", file=sys.stderr)
            return int(proc.returncode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
