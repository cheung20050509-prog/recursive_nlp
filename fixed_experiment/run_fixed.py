#!/usr/bin/env python3
"""Launch embedded ``fixed_training.train`` from a preset name or JSON config."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.registry import (  # noqa: E402
    PRESET_CONFIG_FILES,
    build_train_argv,
    flatten_experiment,
    load_experiment_file,
    parse_cli_overrides,
    resolve_config_path,
)


def _default_python() -> str:
    return os.environ.get("PY", sys.executable)


def _pop_leading_config_spec(argv: list[str]) -> tuple[str | None, list[str]]:
    """
    Support ``run_fixed.py mosi_refine2_t278`` / ``run_fixed.py config/foo.json`` without
    stealing numeric train.py overrides (e.g. ``--train_batch_size 2``), which argparse
    used to bind to an optional positional ``preset``.
    """
    if not argv or argv[0].startswith("-"):
        return None, argv
    head = argv[0]
    if head in PRESET_CONFIG_FILES:
        return head, argv[1:]
    if head.endswith(".json") or "/" in head or "\\" in head:
        return head, argv[1:]
    return None, argv


def _strip_config_option(argv: list[str]) -> tuple[str | None, list[str]]:
    """Remove ``--config PATH`` / ``-c PATH`` and return PATH."""
    out: list[str] = []
    spec: str | None = None
    i = 0
    while i < len(argv):
        if argv[i] in ("--config", "-c"):
            if i + 1 >= len(argv):
                raise ValueError("error: --config requires a path")
            if spec is not None:
                raise ValueError("error: multiple --config are not supported")
            spec = argv[i + 1]
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return spec, out


def _strip_first_preset_token(argv: list[str]) -> tuple[str | None, list[str]]:
    """Remove the first non-flag token that is a preset name or JSON/path-like config."""
    for i, tok in enumerate(argv):
        if tok.startswith("-"):
            continue
        if tok in PRESET_CONFIG_FILES or tok.endswith(".json") or "/" in tok or "\\" in tok:
            return tok, argv[:i] + argv[i + 1 :]
    return None, argv


def main() -> int:
    argv = sys.argv[1:]
    try:
        spec_from_flag, argv = _strip_config_option(argv)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    spec_from_pos, argv = _strip_first_preset_token(argv) if spec_from_flag is None else (None, argv)
    leading_spec = spec_from_flag or spec_from_pos

    ap = argparse.ArgumentParser(
        description="Run fixed / ablation training via fixed_training (embedded train stack)."
    )
    ap.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Explicit path to experiment JSON",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print command and exit")
    ap.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable (default: $PY or current interpreter)",
    )
    args, tail = ap.parse_known_args(argv)

    spec = args.config or leading_spec
    if not spec:
        ap.print_help()
        print("\nError: provide a preset name or --config PATH.", file=sys.stderr)
        return 2

    cfg_path = resolve_config_path(spec)
    exp = load_experiment_file(cfg_path)
    flat = flatten_experiment(exp)
    overrides = parse_cli_overrides(tail)
    train_argv = build_train_argv(flat, extras=overrides)

    py = args.python or _default_python()
    cmd = [py, "-u", "-m", "fixed_training.train", *train_argv]

    if args.dry_run:
        print("cwd:", _ROOT)
        print("config:", cfg_path)
        print("exec:", subprocess.list2cmdline(cmd))
        return 0

    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(_ROOT), env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
