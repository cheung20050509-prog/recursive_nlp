#!/usr/bin/env python
import argparse
import os
import shlex
import signal
import sqlite3
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", required=True, choices=["mosi", "mosei"])
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--parent-log", required=True)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--restart-log", required=True)
    parser.add_argument("--restart-command", required=True)
    parser.add_argument("--poll-seconds", default=0.5, type=float)
    parser.add_argument("--settle-seconds", default=0.2, type=float)
    return parser.parse_args()


def pid_exists(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_children(parent_pid):
    children_path = Path(f"/proc/{parent_pid}/task/{parent_pid}/children")
    if not children_path.exists():
        return []
    content = children_path.read_text().strip()
    if not content:
        return []
    return [int(token) for token in content.split()]


def read_cmdline(pid):
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return ""
    return cmdline_path.read_text().replace("\x00", " ").strip()


def matching_train_children(parent_pid, dataset):
    matches = []
    for child_pid in read_children(parent_pid):
        cmdline = read_cmdline(child_pid)
        if "train.py" in cmdline and f"--dataset {dataset}" in cmdline:
            matches.append((child_pid, cmdline))
    return matches


def current_running_trial(sqlite_path):
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cursor = connection.cursor()
        row = cursor.execute(
            "select number from trials where state = 'RUNNING' order by trial_id desc limit 1"
        ).fetchone()
    finally:
        connection.close()
    return None if row is None else int(row[0])


def restart_parent(args):
    restart_log = Path(args.restart_log)
    restart_log.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.split(args.restart_command)
    with open(restart_log, "a") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=args.work_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(
        f"[{args.label}] restarted patched parent pid={process.pid}, log={restart_log}",
        flush=True,
    )


def main():
    args = parse_args()
    parent_log = Path(args.parent_log)
    sqlite_path = Path(args.sqlite)

    print(
        f"[{args.label}] monitoring parent pid={args.parent_pid}, log={parent_log}, sqlite={sqlite_path}",
        flush=True,
    )

    log_position = parent_log.stat().st_size if parent_log.exists() else 0
    target_trial = current_running_trial(sqlite_path)
    print(f"[{args.label}] current running trial={target_trial}", flush=True)

    while True:
        if not pid_exists(args.parent_pid):
            print(f"[{args.label}] parent pid {args.parent_pid} already exited; no restart performed", flush=True)
            return

        latest_trial = current_running_trial(sqlite_path)
        if latest_trial != target_trial:
            target_trial = latest_trial
            print(f"[{args.label}] tracking running trial={target_trial}", flush=True)

        if not parent_log.exists():
            time.sleep(args.poll_seconds)
            continue

        with open(parent_log, "r") as handle:
            handle.seek(log_position)
            new_text = handle.read()
            log_position = handle.tell()

        if target_trial is None or f"Trial {target_trial} finished" not in new_text:
            time.sleep(args.poll_seconds)
            continue

        print(f"[{args.label}] detected completion of trial {target_trial}; attempting boundary stop", flush=True)
        os.kill(args.parent_pid, signal.SIGSTOP)
        time.sleep(args.settle_seconds)

        live_children = matching_train_children(args.parent_pid, args.dataset)
        if live_children:
            print(
                f"[{args.label}] missed the boundary; child already started {live_children}. Resuming old parent and waiting for the next trial boundary.",
                flush=True,
            )
            os.kill(args.parent_pid, signal.SIGCONT)
            time.sleep(args.poll_seconds)
            continue

        os.kill(args.parent_pid, signal.SIGKILL)
        deadline = time.time() + 10
        while pid_exists(args.parent_pid) and time.time() < deadline:
            time.sleep(0.1)

        if pid_exists(args.parent_pid):
            raise RuntimeError(f"[{args.label}] failed to terminate old parent pid {args.parent_pid}")

        print(f"[{args.label}] old parent stopped after trial {target_trial}; restarting patched process", flush=True)
        restart_parent(args)
        return


if __name__ == "__main__":
    main()