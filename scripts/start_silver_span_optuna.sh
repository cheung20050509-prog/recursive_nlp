#!/usr/bin/env bash
# Parallel silver-span Optuna fleet: MUStARD + UR-FUNNY (optuna_hkt_search.py) +
# CH-SIMSv2 (optuna_search.py).
#
# Unified default output root — all three under log/4080_restart/:
#   FLEET_OUT (default log/4080_restart) -> mustard/, urfunny/, simsv2/
#   (alongside existing mosi/ mosei/ from other scripts)
# Override with FLEET_OUT=..., or set HKT_OUT / SIMSV2_OUT separately if needed.
#
# Default: all three drivers on GPU 0 (OPTUNA_GPU).
# MOSI/MOSEI unchanged — use scripts/start_optuna_4080_restart.sh.
#
# Usage:
#   bash scripts/start_silver_span_optuna.sh
#
# Env (optional):
#   FLEET_OUT — default log/4080_restart (mustard + urfunny + simsv2 sqlite + trial_logs)
#   OPTUNA_GPU, MUSTARD_GPU, URFUNNY_GPU, SIMSV2_GPU
#   *_RANDOM_TRIALS, *_TPE_TRIALS, SIMSV2_BASE_MODEL
#   HKT_STUDY_PREFIX, SIMSV2_STUDY_PREFIX (default timestamped new studies)
#   STOP_PRIOR_FLEET — default 1: kill prior fleet driver PIDs (mustard/urfunny/simsv2
#     from log/silver_span_optuna/*.pid) + stray mustard/urfunny HKT processes.
#   STOP_PRIOR_SIMSV2 — default 0: if 1, also pkill every simsv2 optuna_search/train.py
#     (stops unrelated SIMSv2 runs). Fleet only SIGKILLs its own simsv2 driver via pid file by default.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/root/autodl-tmp/anaconda3/envs/ITHP5090/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/log/silver_span_optuna}"
mkdir -p "$OUT_DIR"

FLEET_OUT="${FLEET_OUT:-log/4080_restart}"
HKT_OUT="${HKT_OUT:-$FLEET_OUT}"
SIMSV2_OUT="${SIMSV2_OUT:-$FLEET_OUT}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
HKT_STUDY_PREFIX="${HKT_STUDY_PREFIX:-ithp_hkt_silver_${STAMP}}"
SIMSV2_STUDY_PREFIX="${SIMSV2_STUDY_PREFIX:-ithp_silver_${STAMP}}"

stop_all_urfunny_hkt() {
  pkill -TERM -f 'optuna_hkt_search.py.*--dataset urfunny' 2>/dev/null || true
  pkill -TERM -f 'optuna_hkt_search.py.*--dataset humor' 2>/dev/null || true
  pkill -TERM -f 'train_hkt_binary.py.*--dataset urfunny' 2>/dev/null || true
  pkill -TERM -f 'train_hkt_binary.py.*--dataset humor' 2>/dev/null || true
  pkill -TERM -f 'train_hkt_binary.py.*--run_name urfunny/' 2>/dev/null || true
  pkill -TERM -f 'build_hkt_silver_span_cache.py.*urfunny' 2>/dev/null || true
  sleep 2
  pkill -KILL -f 'optuna_hkt_search.py.*--dataset urfunny' 2>/dev/null || true
  pkill -KILL -f 'optuna_hkt_search.py.*--dataset humor' 2>/dev/null || true
  pkill -KILL -f 'train_hkt_binary.py.*--dataset urfunny' 2>/dev/null || true
  pkill -KILL -f 'train_hkt_binary.py.*--dataset humor' 2>/dev/null || true
  pkill -KILL -f 'train_hkt_binary.py.*--run_name urfunny/' 2>/dev/null || true
  pkill -KILL -f 'build_hkt_silver_span_cache.py.*urfunny' 2>/dev/null || true
  sleep 1
}

stop_mustard_hkt() {
  pkill -TERM -f 'optuna_hkt_search.py.*--dataset mustard' 2>/dev/null || true
  pkill -TERM -f 'optuna_hkt_search.py.*--dataset sarcasm' 2>/dev/null || true
  pkill -TERM -f 'train_hkt_binary.py.*--dataset mustard' 2>/dev/null || true
  pkill -TERM -f 'train_hkt_binary.py.*--dataset sarcasm' 2>/dev/null || true
  pkill -TERM -f 'train_hkt_binary.py.*--run_name mustard/' 2>/dev/null || true
  sleep 2
  pkill -KILL -f 'optuna_hkt_search.py.*--dataset mustard' 2>/dev/null || true
  pkill -KILL -f 'optuna_hkt_search.py.*--dataset sarcasm' 2>/dev/null || true
  pkill -KILL -f 'train_hkt_binary.py.*--dataset mustard' 2>/dev/null || true
  pkill -KILL -f 'train_hkt_binary.py.*--dataset sarcasm' 2>/dev/null || true
  pkill -KILL -f 'train_hkt_binary.py.*--run_name mustard/' 2>/dev/null || true
  sleep 1
}

stop_simsv2_train_optuna() {
  pkill -TERM -f 'optuna_search.py.*--dataset simsv2' 2>/dev/null || true
  pkill -TERM -f 'train.py.*--dataset simsv2' 2>/dev/null || true
  sleep 2
  pkill -KILL -f 'optuna_search.py.*--dataset simsv2' 2>/dev/null || true
  pkill -KILL -f 'train.py.*--dataset simsv2' 2>/dev/null || true
  sleep 1
}

# STOP_PRIOR_FLEET=0 to skip (e.g. you already killed processes manually)
if [[ "${STOP_PRIOR_FLEET:-1}" == "1" ]]; then
  for f in mustard urfunny simsv2; do
    pf="$OUT_DIR/${f}_optuna.pid"
    if [[ -f "$pf" ]]; then
      oldp="$(cat "$pf" 2>/dev/null || true)"
      if [[ -n "$oldp" ]] && kill -0 "$oldp" 2>/dev/null; then
        kill -TERM "$oldp" 2>/dev/null || true
        sleep 1
        kill -KILL "$oldp" 2>/dev/null || true
      fi
    fi
  done
  stop_all_urfunny_hkt
  stop_mustard_hkt
  if [[ "${STOP_PRIOR_SIMSV2:-0}" == "1" ]]; then
    stop_simsv2_train_optuna
  fi
fi

OPTUNA_GPU="${OPTUNA_GPU:-0}"
MUSTARD_GPU="${MUSTARD_GPU:-$OPTUNA_GPU}"
URFUNNY_GPU="${URFUNNY_GPU:-$OPTUNA_GPU}"
SIMSV2_GPU="${SIMSV2_GPU:-$OPTUNA_GPU}"

MUSTARD_RANDOM_TRIALS="${MUSTARD_RANDOM_TRIALS:-20}"
MUSTARD_TPE_TRIALS="${MUSTARD_TPE_TRIALS:-60}"
URFUNNY_RANDOM_TRIALS="${URFUNNY_RANDOM_TRIALS:-15}"
URFUNNY_TPE_TRIALS="${URFUNNY_TPE_TRIALS:-45}"
SIMSV2_RANDOM_TRIALS="${SIMSV2_RANDOM_TRIALS:-50}"
SIMSV2_TPE_TRIALS="${SIMSV2_TPE_TRIALS:-100}"

HKT_EXTRA_ARGS=()
if [[ -n "${HKT_N_EPOCHS:-}" ]]; then HKT_EXTRA_ARGS+=(--n_epochs "$HKT_N_EPOCHS"); fi
if [[ -n "${HKT_EARLY_STOP_PATIENCE:-}" ]]; then HKT_EXTRA_ARGS+=(--early_stopping_patience "$HKT_EARLY_STOP_PATIENCE"); fi

BASE_MODEL_ARGS=()
if [[ -n "${SIMSV2_BASE_MODEL:-}" ]]; then
  BASE_MODEL_ARGS=(--base_model "$SIMSV2_BASE_MODEL")
fi

cat >"$OUT_DIR/README.txt" <<EOF
Silver-span Optuna fleet — unified under FLEET_OUT=$FLEET_OUT (HKT_OUT=$HKT_OUT, SIMSV2_OUT=$SIMSV2_OUT).
Started: $(date -Is)
Study prefixes: HKT=$HKT_STUDY_PREFIX | simsv2=$SIMSV2_STUDY_PREFIX
Optional HKT overrides: HKT_N_EPOCHS, HKT_EARLY_STOP_PATIENCE (passed to optuna_hkt_search when set).
MUStARD  -> GPU $MUSTARD_GPU  | $HKT_OUT/mustard  | random=$MUSTARD_RANDOM_TRIALS  tpe=$MUSTARD_TPE_TRIALS
UR-FUNNY -> GPU $URFUNNY_GPU | $HKT_OUT/urfunny | random=$URFUNNY_RANDOM_TRIALS tpe=$URFUNNY_TPE_TRIALS
SIMSv2   -> GPU $SIMSV2_GPU  | $SIMSV2_OUT/simsv2 | random=$SIMSV2_RANDOM_TRIALS tpe=$SIMSV2_TPE_TRIALS
Logs + PIDs: $OUT_DIR/
EOF

(
  export CUDA_VISIBLE_DEVICES="$MUSTARD_GPU"
  cd "$ROOT"
  exec "$PY" -u scripts/optuna_hkt_search.py \
    --dataset mustard \
    --gpu "$MUSTARD_GPU" \
    --output_dir "$HKT_OUT" \
    --study_prefix "$HKT_STUDY_PREFIX" \
    --random_trials "$MUSTARD_RANDOM_TRIALS" \
    --tpe_trials "$MUSTARD_TPE_TRIALS" \
    "${HKT_EXTRA_ARGS[@]}"
) >"$OUT_DIR/mustard_optuna.log" 2>&1 &
echo $! >"$OUT_DIR/mustard_optuna.pid"

(
  export CUDA_VISIBLE_DEVICES="$URFUNNY_GPU"
  cd "$ROOT"
  exec "$PY" -u scripts/optuna_hkt_search.py \
    --dataset urfunny \
    --gpu "$URFUNNY_GPU" \
    --output_dir "$HKT_OUT" \
    --study_prefix "$HKT_STUDY_PREFIX" \
    --random_trials "$URFUNNY_RANDOM_TRIALS" \
    --tpe_trials "$URFUNNY_TPE_TRIALS" \
    "${HKT_EXTRA_ARGS[@]}"
) >"$OUT_DIR/urfunny_optuna.log" 2>&1 &
echo $! >"$OUT_DIR/urfunny_optuna.pid"

(
  export CUDA_VISIBLE_DEVICES="$SIMSV2_GPU"
  cd "$ROOT"
  exec "$PY" -u scripts/optuna_search.py \
    --dataset simsv2 \
    --gpu "$SIMSV2_GPU" \
    --output_dir "$SIMSV2_OUT" \
    --study_prefix "$SIMSV2_STUDY_PREFIX" \
    --random_trials "$SIMSV2_RANDOM_TRIALS" \
    --tpe_trials "$SIMSV2_TPE_TRIALS" \
    "${BASE_MODEL_ARGS[@]}"
) >"$OUT_DIR/simsv2_optuna.log" 2>&1 &
echo $! >"$OUT_DIR/simsv2_optuna.pid"

echo "OUT_DIR=$OUT_DIR"
echo "FLEET_OUT=$FLEET_OUT (mustard+urfunny+simsv2 sqlite under here)"
echo "HKT study_prefix=$HKT_STUDY_PREFIX | simsv2 study_prefix=$SIMSV2_STUDY_PREFIX"
echo "MUStARD  Optuna pid=$(cat "$OUT_DIR/mustard_optuna.pid")  GPU=$MUSTARD_GPU"
echo "UR-FUNNY Optuna pid=$(cat "$OUT_DIR/urfunny_optuna.pid") GPU=$URFUNNY_GPU"
echo "SIMSv2   Optuna pid=$(cat "$OUT_DIR/simsv2_optuna.pid")  GPU=$SIMSV2_GPU"
