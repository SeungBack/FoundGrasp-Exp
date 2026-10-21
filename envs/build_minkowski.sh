#!/usr/bin/env bash
# MinkowskiEngine 0.5.4 (chenxi-wang 포크, cuda-12-1 브랜치 — AnyGrasp 지정) 휠을 한 번 빌드해
# 같은 Python/torch 환경들에 설치한다.  사용: build_minkowski.sh <env_name>
source "$(dirname "$0")/common.sh"
ENV="$1"; PY="$(env_py "$ENV")"
WHEEL_DIR="$BUILD_DIR/wheels"; mkdir -p "$WHEEL_DIR"
# Wheel names are normalized to lowercase by newer setuptools.
shopt -s nocaseglob
conda install -y -n "$ENV" -c anaconda openblas-devel
conda install -y -n "$ENV" -c conda-forge ninja
if ! ls "$WHEEL_DIR"/MinkowskiEngine-*.whl >/dev/null 2>&1; then
  SRC="$BUILD_DIR/MinkowskiEngine"
  [ -d "$SRC" ] || git clone https://github.com/chenxi-wang/MinkowskiEngine.git "$SRC"
  git -C "$SRC" checkout cuda-12-1
  PREFIX="$CONDA_BASE/envs/$ENV"
  (cd "$SRC" && CPATH="$(gcc_header_override)" "$PY" setup.py bdist_wheel \
      --blas_include_dirs="$PREFIX/include" --blas_library_dirs="$PREFIX/lib" --blas=openblas \
   && cp dist/MinkowskiEngine-*.whl "$WHEEL_DIR/")
fi
"$PY" -m pip install --no-deps --force-reinstall "$WHEEL_DIR"/MinkowskiEngine-*.whl
