#!/usr/bin/env bash
# Isolated AnyGrasp environment; no packages installed into foundgrasp.
export MAX_JOBS="${MAX_JOBS:-2}"
source "$(dirname "$0")/common.sh"
ENV=efg-anygrasp
PY="$(env_py "$ENV")"
if [ ! -x "$PY" ]; then
  if [ -x "$(env_py foundgrasp)" ]; then
    conda create -y -n "$ENV" --clone foundgrasp
  else
    create_env "$ENV"
    "$PY" -m pip install 'numpy<2' scipy open3d Pillow tqdm graspnetAPI
  fi
fi
if ! "$PY" -c 'import MinkowskiEngine' >/dev/null 2>&1; then
  MAX_JOBS="${MAX_JOBS:-2}" bash "$(dirname "$0")/build_minkowski.sh" "$ENV"
fi
if ! "$PY" -c 'import torch; import pointnet2._ext' >/dev/null 2>&1; then
  # Build a private copy, leaving the original SDK checkout untouched.
  SRC="$(mktemp -d "$BUILD_DIR/anygrasp_pointnet2.XXXXXX")"
  cp -a "$WS/anygrasp_sdk/pointnet2/." "$SRC/"
  (cd "$SRC" && CPATH="$(gcc_header_override)" "$PY" -m pip install --no-build-isolation --no-deps .)
fi
"$PY" -c 'import torch, MinkowskiEngine, pointnet2._ext, graspnetAPI; print("AnyGrasp dependencies ready:", torch.__version__, torch.cuda.is_available())'
