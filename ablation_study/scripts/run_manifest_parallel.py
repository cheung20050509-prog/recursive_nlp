#!/usr/bin/env python3
"""Run manifest entries concurrently on one (or more) GPU(s).

Each job is a separate process with its own model copy; VRAM adds up. Use
--fixed-extra / --hkt-extra to lower --train_batch_size and raise
--gradient_accumulation_step so several jobs fit on one card.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

# Match Optuna MMSA + fixed_experiment repro: train.py defaults (see fixed_training/train.py).
_PROFILE_OPTUNA_FIXED = ["--train_batch_size", "8", "--gradient_accumulation_step", "1"]
# HKT UR-FUNNY batch lives in each JSON argv; do not override here.
_PROFILE_OPTUNA_HKT: list[str] = []

# 32GB single-GPU, two concurrent jobs: lower per-job VRAM (NOT comparable to Optuna/repro for MMSA).
_PROFILE_32G2_FIXED = ["--train_batch_size", "4", "--gradient_accumulation_step", "2"]
_PROFILE_32G2_HKT = ["--train_batch_size", "8", "--gradient_accumulation_step", "2"]


def _build_cmd(
    row: dict,
    i: int,
    ablation_root: Path,
    run_fixed: Path,
    launch_hkt: Path,
    py: str,
    fixed_extra: list[str],
    hkt_extra: list[str],
) -> tuple[str, list[str]] | None:
    rid = row.get("id", f"entry_{i}")
    runner = row.get("runner")
    cfg_rel = row.get("config")
    if not runner or not cfg_rel:
        print(f"skip {rid}: missing runner or config", file=sys.stderr)
        return None
    cfg_path = (ablation_root / cfg_rel).resolve()
    if not cfg_path.is_file():
        print(f"skip {rid}: missing file {cfg_path}", file=sys.stderr)
        return None

    if runner == "fixed":
        cmd = [py, "-u", str(run_fixed), "--config", str(cfg_path), *fixed_extra]
    elif runner == "hkt_urfunny":
        cmd = [py, "-u", str(launch_hkt), str(cfg_path), *hkt_extra]
    else:
        print(f"skip {rid}: unknown runner {runner}", file=sys.stderr)
        return None
    return rid, cmd


def _run_subprocess(rid: str, cmd: list[str]) -> tuple[str, int]:
    print(f"[parallel] START {rid}\n{subprocess.list2cmdline(cmd)}\n", flush=True)
    proc = subprocess.run(cmd)
    code = int(proc.returncode)
    print(f"[parallel] END {rid} exit={code}", flush=True)
    return rid, code


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run ablation manifest with bounded parallelism (same GPU: lower batch first)."
    )
    ap.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest.json (default: <ablation_study>/manifest.json)",
    )
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=2,
        help="Max concurrent training processes (default: 2)",
    )
    ap.add_argument("--start", type=int, default=0, help="Skip first N entries")
    ap.add_argument("--limit", type=int, default=0, help="Run at most N entries (0 = all)")
    ap.add_argument(
        "--fixed-extra",
        type=str,
        default="",
        help="Extra argv forwarded to run_fixed.py after --config (shlex-split), e.g. "
        "'--train_batch_size 4 --gradient_accumulation_step 2'",
    )
    ap.add_argument(
        "--hkt-extra",
        type=str,
        default="",
        help="Extra argv appended after JSON path for launch_hkt_json.py (shlex-split)",
    )
    ap.add_argument(
        "--profile",
        type=str,
        choices=("off", "optuna", "32g2"),
        default="optuna",
        help=(
            "optuna (default): MMSA fixed runs get train_batch_size=8, gradient_accumulation_step=1 "
            "(matches Optuna build_train_command + fixed_experiment repro). UR-FUNNY argv unchanged. "
            "off: no preset extras. 32g2: lower batch for multi-job VRAM; breaks strict MMSA repro match."
        ),
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Do not stop the pool when a job fails; exit non-zero if any failed",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands only")
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

    fixed_src = args.fixed_extra.strip()
    hkt_src = args.hkt_extra.strip()
    if args.profile == "32g2":
        if not fixed_src:
            fixed_extra = list(_PROFILE_32G2_FIXED)
        else:
            try:
                fixed_extra = shlex.split(args.fixed_extra)
            except ValueError as e:
                print(f"shlex split error (--fixed-extra): {e}", file=sys.stderr)
                return 2
        if not hkt_src:
            hkt_extra = list(_PROFILE_32G2_HKT)
        else:
            try:
                hkt_extra = shlex.split(args.hkt_extra)
            except ValueError as e:
                print(f"shlex split error (--hkt-extra): {e}", file=sys.stderr)
                return 2
    elif args.profile == "optuna":
        if not fixed_src:
            fixed_extra = list(_PROFILE_OPTUNA_FIXED)
        else:
            try:
                fixed_extra = shlex.split(args.fixed_extra)
            except ValueError as e:
                print(f"shlex split error (--fixed-extra): {e}", file=sys.stderr)
                return 2
        if not hkt_src:
            hkt_extra = list(_PROFILE_OPTUNA_HKT)
        else:
            try:
                hkt_extra = shlex.split(args.hkt_extra)
            except ValueError as e:
                print(f"shlex split error (--hkt-extra): {e}", file=sys.stderr)
                return 2
    else:
        try:
            fixed_extra = shlex.split(args.fixed_extra)
            hkt_extra = shlex.split(args.hkt_extra)
        except ValueError as e:
            print(f"shlex split error: {e}", file=sys.stderr)
            return 2

    planned: list[tuple[str, list[str]]] = []
    for i, row in enumerate(slice_entries):
        b = _build_cmd(row, i, ablation_root, run_fixed, launch_hkt, py, fixed_extra, hkt_extra)
        if b:
            planned.append(b)

    if args.dry_run:
        for rid, cmd in planned:
            print(subprocess.list2cmdline(cmd))
        return 0

    if args.jobs < 1:
        print("--jobs must be >= 1", file=sys.stderr)
        return 2

    lock = threading.Lock()
    failures: list[str] = []

    def _worker(rid: str, cmd: list[str]) -> tuple[str, int]:
        _, code = _run_subprocess(rid, cmd)
        if code != 0:
            with lock:
                failures.append(rid)
        return rid, code

    # Bounded concurrency: at most `jobs` training subprocesses at once.
    it = iter(planned)
    in_flight: dict[Future, str] = {}
    max_j = args.jobs
    stop_early = False

    with ThreadPoolExecutor(max_workers=max_j) as ex:

        def _fill() -> None:
            nonlocal stop_early
            while len(in_flight) < max_j and not stop_early:
                try:
                    rid, cmd = next(it)
                except StopIteration:
                    return
                fut = ex.submit(_worker, rid, cmd)
                in_flight[fut] = rid

        _fill()
        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                rid = in_flight.pop(fut)
                try:
                    _, code = fut.result()
                except Exception as e:
                    print(f"[parallel] {rid} crashed: {e}", file=sys.stderr)
                    code = 1
                    with lock:
                        failures.append(rid)
                if code != 0 and not args.continue_on_error:
                    stop_early = True
                    print(f"[parallel] no more jobs will start after failure: {rid}", file=sys.stderr)
                if not stop_early:
                    _fill()

    if failures:
        print(f"[parallel] failed runs: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
