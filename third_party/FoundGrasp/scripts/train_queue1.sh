#!/bin/bash
# Sequential training queue.
#
# Usage:
#   tmux new -s train
#   conda activate vit-grasp
#   bash scripts/train_queue.sh
#   Ctrl+B, D   (detach)
#
# To monitor:
#   tmux attach -t train

set -e  # stop on first error

GPU=1
CONFIGS=(
configs/tmech/t1_dav2s.yaml                 
configs/tmech/t2_dav2b_kinect.yaml          
configs/tmech/t2_dav2s_kinect.yaml          
configs/tmech/t4_frozen_geoformer.yaml      
)

for CONFIG in "${CONFIGS[@]}"; do
    echo ""
    echo "=================================================="
    echo "  START: $CONFIG"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================="
    CUDA_VISIBLE_DEVICES=$GPU python train.py --config "$CONFIG"
    echo "  DONE:  $CONFIG  ($(date '+%Y-%m-%d %H:%M:%S'))"
done

echo ""
echo "All done."
