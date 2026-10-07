# FoundGrasp

FoundGrasp is an RGB-D grasp pose prediction project built around frozen vision
foundation backbones, geometric adapters, and pixel-wise grasp supervision.

This README documents the active workflow in this repository. Current configs
live under `configs/final/`.

## Installation

Create an environment:

```bash
conda create -n foundgrasp python=3.10
conda activate foundgrasp
```

Install one PyTorch build that matches your CUDA version:

```bash
# CUDA 12.8
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121
```

Install Python dependencies:

```bash
pip install "numpy<2" scipy opencv-python pillow tqdm matplotlib albumentations h5py
pip install transformers omegaconf huggingface_hub open3d rerun-sdk wandb comet_ml
pip install git+https://github.com/seungback/graspnetAPI.git
pip install git+https://github.com/seungback/graspclutter6dAPI.git
pip install git+https://github.com/ByteDance-Seed/depth-anything-3
```

For RealSense live inference, also install `pyrealsense2`.

## Inference

`demo.py` runs on an arbitrary RGB image or a live RealSense RGB-D stream. It
does not require GraspNet preprocessing, but it does require a trained
checkpoint.

Run a single image with rerun visualization:
```bash
python demo.py sample.png \
  --config configs/final/foundgrasp_base.yaml \
  --checkpoint logs/final/foundgrasp_base/checkpoints/final_model.pth \
  --output demo_output \
  --rerun
```

Run live RealSense inference with rerun visualization:

```bash
python demo.py --realsense \
  --config configs/final/foundgrasp_base.yaml \
  --checkpoint logs/final/foundgrasp_base/checkpoints/final_model.pth \
  --rerun
```

Run RealSense inference only when triggered from the terminal:

```bash
python demo.py --realsense \
  --config configs/final/foundgrasp_base.yaml \
  --checkpoint logs/final/foundgrasp_base/checkpoints/final_model.pth \
  --rs-mode manual \
  --rerun
```

Useful demo options:

```bash
--rerun                     # launch rerun visualization
--skip-cd                   # skip collision detection; enabled by default
--no-nms                    # disable final GraspGroup NMS; voxel candidate reduction remains in to_grasp_group
--no-rs-postprocess         # disable RealSense depth postprocessing; disparity/spatial/temporal are enabled by default
--rs-mode live|manual|once  # continuous, Enter-triggered, or single-frame RealSense inference
--status-interval 0.5       # live status-line refresh interval in seconds
```

By default, rerun uses fixed local ports and a stable viewer URL:
`http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy`.

## Training

The dataset setup and preprocessing below are only required for training or
dataset benchmark evaluation. They are not required for single-image inference
with `demo.py`.

### Data

Set dataset roots in `configs/_base.yaml`:

```yaml
datasets:
  GraspNet1B:
    root: /data/Grasp/GraspNet-1Billion
  GraspClutter6D:
    root: /data/Grasp/GraspClutter6D
```

GraspNet training expects pixel-wise grasp labels under:

```text
<GraspNet root>/pixelwise_grasp_gt_top300/<camera>/scene_XXXX_ann_YYYY.npz
```

If you do not have GraspClutter6D locally, disable the GraspClutter6D final-test
block in the config before training.

### Preprocessing

Generate GraspNet pixel-wise labels. The multiprocessing script dispatches
scenes across the GPUs listed in `--gpu_ids`.

```bash
python scripts/gen_pixelwise_gt_mp.py \
  --root /data/Grasp/GraspNet-1Billion \
  --camera realsense \
  --gpu_ids 0,1,2,3
```

For Kinect labels, change `--camera kinect`.

### Run Training

Use `configs/final/foundgrasp_base.yaml` as the canonical current config:

```bash
python train.py \
  --config configs/final/foundgrasp_base.yaml
```

Checkpoints and logs are written to:

```text
logs/final/foundgrasp_base/
logs/final/foundgrasp_base/checkpoints/
```

Resume from the latest epoch checkpoint:

```bash
python train.py \
  --config configs/final/foundgrasp_base.yaml \
  --resume
```

For multi-GPU training, set `training.ddp.enabled: true` in the config and then
launch with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --config configs/final/foundgrasp_base.yaml
```

## Dataset Evaluation

Dataset evaluation is for benchmark runs and requires the dataset setup above.
If `--checkpoint` is omitted, `test.py` uses the latest
`checkpoint_epoch_*.pth` under the matching log directory.

```bash
python test.py \
  --config configs/final/foundgrasp_base.yaml \
  --infer --eval \
  --camera realsense \
  --split test_seen \
  --batch_size 16 \
  --num_workers 8
```

Evaluate a specific checkpoint:

```bash
python test.py \
  --config configs/final/foundgrasp_base.yaml \
  --checkpoint logs/final/foundgrasp_base/checkpoints/final_model.pth \
  --infer --eval \
  --camera realsense \
  --split test_seen \
  --batch_size 16 \
  --num_workers 8
```

Evaluate on GraspClutter6D:

```bash
python test.py \
  --config configs/final/foundgrasp_base.yaml \
  --checkpoint logs/final/foundgrasp_base/checkpoints/final_model.pth \
  --infer --eval \
  --dataset GraspClutter6D \
  --camera realsense-d435 \
  --split test \
  --batch_size 16 \
  --num_workers 8
```

Useful evaluation options:

```bash
--save-vis    # save 2D visualization images
--rerun       # launch rerun visualization server
--skip-cd     # skip collision detection
```

## Utilities

Run an inference-parameter sweep on `test_mini_mixed`:

```bash
python tools/sweep.py \
  --config configs/final/foundgrasp_base.yaml \
  --ann-ids-path mini_mixed_ann_ids.json \
  --gpu 0
```

Analyze sweep outputs:

```bash
python tools/analyze_sweep.py \
  logs/final/foundgrasp_base/test_mini_mixed_sweep_results.json \
  --top-k 10 \
  --out-dir logs/final/foundgrasp_base/sweep_analysis
```

Visualize depth corruption settings:

```bash
python tools/visualize_depth_corruption.py \
  --config configs/final/foundgrasp_base.yaml \
  --split train \
  --force-prob 1.0 \
  --save-dir figures/depth_corruption
```
