#!/bin/bash
# Sequential test/eval queue.
#
# Usage:
#   bash scripts/test_queue.sh

set -e

GPU=0
CAMERA=realsense
SPLIT=test

# List of (config, checkpoint) pairs
JOBS=(
    "configs/paper/dav3s.yaml  logs/paper/dav3s/checkpoints/checkpoint_epoch_150.pth"
    "configs/paper/dav3b.yaml  logs/paper/dav3b/checkpoints/checkpoint_epoch_150.pth"
    "configs/paper/dav3l.yaml  logs/paper/dav3l/checkpoints/checkpoint_epoch_150.pth"
)

for JOB in "${JOBS[@]}"; do
    CONFIG=$(echo $JOB | awk '{print $1}')
    CKPT=$(echo $JOB | awk '{print $2}')

    echo ""
    echo "=================================================="
    echo "  CONFIG:     $CONFIG"
    echo "  CHECKPOINT: $CKPT"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
    CUDA_VISIBLE_DEVICES=$GPU python test.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --camera $CAMERA \
        --split $SPLIT \
        --infer --eval
    echo "  DONE: $CONFIG  ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo "All done."
