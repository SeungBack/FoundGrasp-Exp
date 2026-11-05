#!/usr/bin/env bash
# Build one CUDA wheel for the pinned Python 3.10 / torch 2.9 / cu128 worker ABI.
# Reuse it between RNGNet and GFLA without changing either source checkout.
export MAX_JOBS="${MAX_JOBS:-2}"
source "$(dirname "$0")/common.sh"
ENV="${1:?usage: build_pytorch3d.sh <conda-env>}"
PY="$(env_py "$ENV")"
# Absolute Python paths do not activate a conda environment: expose its ninja
# executable so torch does not silently fall back to a slow serial build.
export PATH="$(dirname "$PY"):$PATH"
P3D_REV=75ebeeaea0908c5527e7b1e305fbc7681382db47
P3D_SOURCE="$BUILD_DIR/pytorch3d"
P3D_WHEELS="$BUILD_DIR/pytorch3d-py310-torch2.9-cu128-sm120"
"$PY" - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 10), 'This wheel build uses Python 3.10'
assert torch.__version__ == '2.9.0+cu128', 'Expected torch 2.9.0+cu128'
PY
if "$PY" -c 'import torch; import pytorch3d._C; import pytorch3d.ops' 2>/dev/null; then
  exit 0
fi
"$PY" -m pip install 'setuptools<70' wheel ninja fvcore iopath
mkdir -p "$P3D_WHEELS"
# Both setup scripts share this wheel directory; serialize a first-time build.
exec 9>"$P3D_WHEELS/.build.lock"
flock 9
shopt -s nullglob
wheels=("$P3D_WHEELS"/pytorch3d-*-cp310-*.whl)
if [ "${#wheels[@]}" -eq 0 ]; then
  if [ ! -d "$P3D_SOURCE" ]; then
    git init "$P3D_SOURCE"
    git -C "$P3D_SOURCE" remote add origin https://github.com/facebookresearch/pytorch3d.git
    git -C "$P3D_SOURCE" fetch --depth 1 origin "$P3D_REV"
    git -C "$P3D_SOURCE" checkout --detach FETCH_HEAD
  fi
  if [ "$(git -C "$P3D_SOURCE" rev-parse HEAD)" != "$P3D_REV" ]; then
    echo "PyTorch3D source must be at $P3D_REV: $P3D_SOURCE" >&2
    exit 1
  fi
  (cd "$P3D_SOURCE" && FORCE_CUDA=1 CPATH="$(gcc_header_override)" \
    "$PY" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$P3D_WHEELS" .)
  wheels=("$P3D_WHEELS"/pytorch3d-*-cp310-*.whl)
fi
[ "${#wheels[@]}" -eq 1 ] || { echo "Expected one PyTorch3D wheel in $P3D_WHEELS" >&2; exit 1; }
flock -u 9
"$PY" -m pip install --no-deps "${wheels[0]}"
"$PY" - <<'PY'
import torch
from pytorch3d.ops import sample_farthest_points, ball_query
points = torch.rand(1, 16, 3, device='cuda')
sample_farthest_points(points, K=4)
ball_query(points[:, :2], points, K=4, radius=1.)
torch.cuda.synchronize()
print('PyTorch3D CUDA kernels ready:', torch.cuda.get_device_name())
PY
