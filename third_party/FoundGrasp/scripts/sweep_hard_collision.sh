#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"
CONFIG="${CONFIG:-configs/paper/dav3b.yaml}"
SPLIT="${SPLIT:-test_mini_mixed}"
ANN_IDS_PATH="${ANN_IDS_PATH:-mini_mixed_ann_ids.json}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
ROOT="${ROOT:-logs/paper/dav3b}"
EXTRA_ARGS=()
if [[ "${REUSE_EXISTING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--reuse-existing)
fi

python scripts/greedy_hard_collision_sweep.py \
  --gpu "$GPU" \
  --config "$CONFIG" \
  --split "$SPLIT" \
  --ann-ids-path "$ANN_IDS_PATH" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --root "$ROOT" \
  "${EXTRA_ARGS[@]}"
