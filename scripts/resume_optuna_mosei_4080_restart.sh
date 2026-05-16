#!/usr/bin/env bash
# Resume MOSEI Optuna on the SAME study as log/4080_restart (sqlite + trial_logs).
#
# optuna_search.py counts COMPLETE trials per phase:
#   remaining_random = max(0, --random_trials - completed_random)
#   remaining_tpe    = max(0, --tpe_trials    - completed_tpe)
# So --random_trials / --tpe_trials are TARGET totals for that phase, not "add N more".
#
# Defaults: keep random cap at 50 (already full -> 0 new random runs).
#           set --tpe_trials to (current complete TPE) + MOSEI_EXTRA_TPE_TRIALS.
#
# Usage:
#   cd ITHP/recursive_ITHP
#   MOSEI_GPU=0 MOSEI_EXTRA_TPE_TRIALS=300 bash scripts/resume_optuna_mosei_4080_restart.sh
#   # optional: append log
#   MOSEI_EXTRA_TPE_TRIALS=400 bash scripts/resume_optuna_mosei_4080_restart.sh >>log/4080_restart/mosei_optuna_resume.log 2>&1 &
#
# Do NOT change --study_prefix or search space when resuming; Optuna merges categorical
# choices from existing trials (see optuna_search.py::_merge_frozen_categorical_distributions).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python3}"
OUT_DIR="${OUT_DIR:-$ROOT/log/4080_restart}"
if [[ "$OUT_DIR" == "$ROOT"/* ]]; then
  OUTPUT_REL="${OUT_DIR#$ROOT/}"
else
  OUTPUT_REL="$OUT_DIR"
fi
MOSEI_GPU="${MOSEI_GPU:-0}"
MOSEI_EXTRA_TPE_TRIALS="${MOSEI_EXTRA_TPE_TRIALS:-300}"
RANDOM_TARGET="${MOSEI_RANDOM_TRIALS_TARGET:-50}"

SQLITE="$OUT_DIR/mosei/optuna_study.sqlite3"
if [[ ! -f "$SQLITE" ]]; then
  echo "error: missing $SQLITE" >&2
  exit 2
fi

TPE_DONE="$("$PY" << PY
import optuna
from pathlib import Path
p = Path("$SQLITE").resolve()
study = optuna.load_study(study_name="ithp_mosei_mae", storage=f"sqlite:///{p}")
n = sum(
    1
    for t in study.trials
    if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("phase") == "tpe"
)
print(n)
PY
)"

TPE_TARGET=$((TPE_DONE + MOSEI_EXTRA_TPE_TRIALS))
echo "MOSEI resume: sqlite=$SQLITE"
echo "  completed TPE so far: $TPE_DONE"
echo "  MOSEI_EXTRA_TPE_TRIALS=$MOSEI_EXTRA_TPE_TRIALS -> --tpe_trials target=$TPE_TARGET"
echo "  random phase target=$RANDOM_TARGET (usually already satisfied)"
echo "  GPU=$MOSEI_GPU"

export CUDA_VISIBLE_DEVICES="$MOSEI_GPU"
exec "$PY" -u scripts/optuna_search.py \
  --dataset mosei \
  --gpu "$MOSEI_GPU" \
  --output_dir "$OUTPUT_REL" \
  --random_trials "$RANDOM_TARGET" \
  --tpe_trials "$TPE_TARGET"
