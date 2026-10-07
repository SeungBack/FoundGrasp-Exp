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

GPU=3
CONFIGS=(
configs/tmech/t5_ft_dinov2b.yaml
configs/tmech/t5_ft_siglipb.yaml
configs/tmech/t5_ft_vitb.yaml
configs/tmech/t6_repr_only.yaml
configs/tmech/t6_visual_only.yaml
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
