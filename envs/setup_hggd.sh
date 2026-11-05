#!/usr/bin/env bash
# efg-hggd: RNGNet·HGGD 워커 공용 환경 (두 방법 모두 THU-VCLab 코드베이스, 같은 의존성).
#   pytorch3d 는 torch 2.9/cu128/sm_120 용 휠이 없어 소스 빌드.
#   cupoch 점군 voxel 연산은 workers/shims/cupoch.py의 PyTorch GPU 구현을 사용한다.
#   Open3D와 같은 voxel origin/centroid 정의를 유지한다.
export MAX_JOBS="${MAX_JOBS:-2}"
source "$(dirname "$0")/common.sh"
ENV=efg-hggd
PY="$(env_py "$ENV")"
if [ ! -x "$PY" ] && [ -x "$(env_py foundgrasp)" ]; then
  conda create -y -n "$ENV" --clone foundgrasp
elif ! "$PY" -c "import torch; assert torch.__version__ == '2.9.0+cu128'" 2>/dev/null; then
  create_env "$ENV"
fi
"$PY" -m pip install 'setuptools<70' 'numpy==1.23.4' 'numba==0.58.1' scipy scikit-image matplotlib \
    'opencv-python-headless<4.12' 'open3d==0.19.0' trimesh transforms3d tqdm pandas h5py \
    fvcore iopath autolab_core grasp_nms dill cvxopt ipython tensorboardX torchsummary thop
RNG_SOURCE="$WS/RegionNormalizedGrasp"
RNG_REV=25a60d0a4232e9e6061193200837dc4f8a18a508
if [ ! -d "$RNG_SOURCE" ]; then
  git init "$RNG_SOURCE"
  git -C "$RNG_SOURCE" remote add origin https://github.com/THU-VCLab/RegionNormalizedGrasp.git
  git -C "$RNG_SOURCE" fetch --depth 1 origin "$RNG_REV"
  git -C "$RNG_SOURCE" checkout --detach FETCH_HEAD
fi
if [ "$(git -C "$RNG_SOURCE" rev-parse HEAD)" != "$RNG_REV" ]; then
  echo "RNGNet source must be at $RNG_REV: $RNG_SOURCE (existing checkout left unchanged)" >&2
  exit 1
fi
# The official checkout includes checkpoints/realsense and checkpoints/kinect.
test -f "$RNG_SOURCE/checkpoints/realsense"
bash "$(dirname "$0")/build_pytorch3d.sh" "$ENV"
"$PY" -c "import torch, pytorch3d.ops, numba, skimage, open3d; print('RNGNet dependencies ready:', torch.__version__, torch.cuda.is_available())"
