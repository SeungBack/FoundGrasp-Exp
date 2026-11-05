#!/usr/bin/env bash
# Isolated GFLA runtime for RTX 5090. Preserve upstream sources and checkpoints.
export MAX_JOBS="${MAX_JOBS:-2}"
source "$(dirname "$0")/common.sh"
ENV=efg-gfla
PY="$(env_py "$ENV")"
if [ ! -x "$PY" ] && [ -x "$(env_py foundgrasp)" ]; then
  conda create -y -n "$ENV" --clone foundgrasp
elif ! "$PY" -c "import torch; assert torch.__version__ == '2.9.0+cu128'" 2>/dev/null; then
  create_env "$ENV"
fi

# numpy 1.23 keeps graspnetAPI's legacy numpy aliases available. FastSAM uses
# the upstream Ultralytics 8.0.120 API, not the current YOLO API.
"$PY" -m pip install 'setuptools<70' wheel ninja 'pybind11==3.1.0' \
  'numpy==1.23.4' scipy matplotlib pandas Pillow tqdm 'opencv-contrib-python<4.12' \
  'open3d==0.19.0' trimesh networkx 'ruamel.yaml==0.17.21' typing_extensions \
  'graspnetAPI==1.2.11' ipython tensorboard ftfy regex 'ultralytics==8.0.120' \
  'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'

checkout_source() {
  local name="$1" url="$2" revision="$3" destination="$4"
  if [ ! -d "$destination" ]; then
    git clone "$url" "$destination"
    git -C "$destination" checkout --detach "$revision"
  fi
  if [ "$(git -C "$destination" rev-parse HEAD)" != "$revision" ]; then
    echo "$name must be at $revision: $destination (existing checkout left unchanged)" >&2
    exit 1
  fi
}

checkout_source GFLA https://github.com/Nx1021/GFLA-release.git \
  314dbe069d835957461fbd1078aafb246abecc83 "$WS/GFLA-release"
# These pinned helpers declare Python >=3.11. The inference import path uses
# typing.Self and typing.TypeVarTuple; gfla_worker supplies those annotation
# backports from typing_extensions within its isolated process.
"$PY" -m pip install --no-deps --ignore-requires-python \
  'git+https://github.com/Nx1021/LauncherTemplate.git@eb8d0df9a0fa1d0b66934e013df6fe64030d8e33' \
  'git+https://github.com/Nx1021/DatasetTemplate.git@9543930258cf4ba98e543ad6b654d2088049a329'

# Package the unchanged upstream NMS source outside the external checkout.
# Eigen is a header-only dependency, installed privately rather than via sudo.
EIGEN_DIR="$BUILD_DIR/eigen-3.3.7"
if [ ! -f "$EIGEN_DIR/Eigen/Dense" ]; then
  curl --fail --location --retry 3 \
    https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz \
    --output "$BUILD_DIR/eigen-3.3.7.tar.gz"
  tar -xzf "$BUILD_DIR/eigen-3.3.7.tar.gz" -C "$BUILD_DIR"
fi
NMS_BUILD="$BUILD_DIR/gfla_native_nms"
mkdir -p "$NMS_BUILD"
cp "$WS/GFLA-release/models/my_grasp_nms/grasp_nms.cpp" "$NMS_BUILD/"
cat > "$NMS_BUILD/setup.py" <<'PY'
import os
import pybind11
from setuptools import Extension, setup

setup(
    name="gfla-native-nms", version="0.1.0",
    ext_modules=[Extension(
        "grasp_nms_cpp", ["grasp_nms.cpp"],
        include_dirs=[pybind11.get_include(), os.environ["GFLA_EIGEN_INCLUDE"]],
        language="c++", extra_compile_args=["-O3", "-std=c++17"],
    )],
)
PY
GFLA_EIGEN_INCLUDE="$EIGEN_DIR" "$PY" -m pip install --no-build-isolation --no-deps "$NMS_BUILD"
bash "$(dirname "$0")/build_pytorch3d.sh" "$ENV"
"$PY" - <<'PY'
import numpy as np
import torch
import grasp_nms_cpp
assert grasp_nms_cpp.grasp_nms(np.empty((0, 17), np.float32)).shape == (0, 17)
print("GFLA dependencies ready:", torch.__version__, torch.cuda.is_available())
PY
for asset in realsense_best.pth FastSAM-s.pt; do
  if [ ! -f "$WS/GFLA-release/base_weights/$asset" ]; then
    echo "Required upstream checkpoint missing: $WS/GFLA-release/base_weights/$asset" >&2
    exit 1
  fi
done
