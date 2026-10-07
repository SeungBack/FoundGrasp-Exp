#!/bin/bash
# Smoke-test every yaml in configs/tmech/: runs exactly 1 training iteration each.
# Validation and final-test are disabled so it finishes in seconds per config.
#
# Usage:
#   conda activate vit-grasp
#   bash scripts/smoke_tmech.sh
#
# Results are printed at the end: PASS / FAIL per config.

set -euo pipefail

GPU=${1:-0}
TMECH_DIR="configs/tmech"
TMP_DIR="configs/debug/tmp_smoke"
mkdir -p "$TMP_DIR"

PASS=()
FAIL=()

for NAME in siglipb t5_frozen_samb t5_frozen_siglipb t5_frozen_vitb t5_ft_siglipb t5_ft_vitb vitb; do
    YAML="$TMECH_DIR/${NAME}.yaml"
    TMP_YAML="$TMP_DIR/${NAME}.yaml"

    # Create a temporary config that inherits from this tmech yaml
    # and overrides only what's needed for a 1-step smoke test.
    cat > "$TMP_YAML" << EOF
__base__: '../../tmech/${NAME}.yaml'
training:
  num_epochs: 1
  batch_size: 4
  num_workers: 2
  evaluation_freq: 999999   # skip in-training validation
  checkpoint_freq: 999      # skip checkpoint saves
  final_test:
    enabled: false          # skip final test
    graspclutter6d:
      enabled: false
dataset:
  train_max_samples: 4      # 1 batch only
EOF

    echo ""
    echo "=========================================="
    echo "  TESTING: $NAME"
    echo "=========================================="

    if CUDA_VISIBLE_DEVICES=$GPU python train.py --config "$TMP_YAML" 2>&1; then
        PASS+=("$NAME")
        echo "  >>> PASS: $NAME"
    else
        FAIL+=("$NAME")
        echo "  >>> FAIL: $NAME"
    fi

    rm -f "$TMP_YAML"
done

rmdir "$TMP_DIR" 2>/dev/null || true

echo ""
echo "=========================================="
echo "  RESULTS"
echo "=========================================="
echo "PASS (${#PASS[@]}):"
for name in "${PASS[@]}"; do echo "  ✓ $name"; done
echo ""
echo "FAIL (${#FAIL[@]}):"
for name in "${FAIL[@]}"; do echo "  ✗ $name"; done
echo ""
