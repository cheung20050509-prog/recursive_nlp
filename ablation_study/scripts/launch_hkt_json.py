#!/usr/bin/env python3
"""Run recursive_ITHP/train_hkt_binary.py from a UR-FUNNY ablation JSON (argv list)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _argv_tokens(raw: list) -> list[str]:
    out: list[str] = []
    for x in raw:
        if x is True or x is False:
            raise ValueError("Boolean argv entries are not supported; use flag strings like '--disable_hcf'.")
        out.append(str(x))
    return out


def _parse_launcher_argv(argv: list[str]) -> tuple[bool, str, list[str]]:
    """[--dry-run] CONFIG.json [extra tokens for train_hkt_binary.py ...]"""
    dry = False
    if argv and argv[0] == "--dry-run":
        dry = True
        argv = argv[1:]
    if len(argv) < 1:
        raise ValueError("missing CONFIG.json")
    return dry, argv[0], argv[1:]


def main() -> int:
    try:
        dry, cfg_str, train_extra = _parse_launcher_argv(sys.argv[1:])
    except ValueError as e:
        print(f"usage: launch_hkt_json.py [--dry-run] CONFIG.json [train_hkt_binary args...]: {e}", file=sys.stderr)
        return 2

    cfg_path = Path(cfg_str).resolve()

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw_argv = data.get("argv") or data.get("command_args")
    if not isinstance(raw_argv, list) or not raw_argv:
        print("config must contain a non-empty 'argv' list", file=sys.stderr)
        return 2

    ablation_root = Path(__file__).resolve().parent.parent
    hkt_dir = ablation_root.parent
    script = hkt_dir / "train_hkt_binary.py"
    if not script.is_file():
        print(f"missing trainer: {script}", file=sys.stderr)
        return 2

    py = os.environ.get("PY", sys.executable)
    # JSON argv first; trailing CLI tokens override (argparse last-wins in train_hkt_binary.py).
    cmd = [py, "-u", str(script), *_argv_tokens(raw_argv), *train_extra]
    if dry:
        print("cwd:", hkt_dir)
        print("config:", cfg_path)
        print("exec:", subprocess.list2cmdline(cmd))
        return 0
    proc = subprocess.run(cmd, cwd=str(hkt_dir))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
