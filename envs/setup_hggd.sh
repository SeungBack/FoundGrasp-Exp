#!/usr/bin/env bash
# efg-hggd: RNGNet·HGGD 워커 공용 환경 (두 방법 모두 THU-VCLab 코드베이스, 같은 의존성).
#   pytorch3d 는 torch 2.9/cu128/sm_120 용 휠이 없어 소스 빌드.
#   cupoch(점군 voxel 다운샘플에만 사용)은 휠 설치를 시도하고, GPU에서 안 돌면 워커가
#   efg/workers/shims/cupoch.py 대체 모듈을 쓴다.
source "$(dirname "$0")/common.sh"
ENV=efg-hggd
create_env "$ENV"
PY="$(env_py "$ENV")"
"$PY" -m pip install "numpy==1.23.5" "numba==0.58.1" scipy scikit-image matplotlib \
    "opencv-python-headless<4.12" "open3d==0.19.0" trimesh transforms3d tqdm pandas h5py fvcore iopath
"$PY" -m pip install autolab_core grasp_nms dill cvxopt || true
"$PY" -m pip install cupoch || true
if ! "$PY" -c "import pytorch3d" 2>/dev/null; then
  SRC="$BUILD_DIR/pytorch3d"
  [ -d "$SRC" ] || git clone --depth 1 --branch stable https://github.com/facebookresearch/pytorch3d.git "$SRC"
  (cd "$SRC" && FORCE_CUDA=1 CPATH="$(gcc_header_override)" "$PY" -m pip install --no-build-isolation .)
fi
"$PY" -c "import torch, pytorch3d.ops, numba, skimage; print('OK', torch.__version__, torch.cuda.is_available())"
