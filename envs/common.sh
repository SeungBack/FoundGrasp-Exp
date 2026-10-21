# 비교군 conda 환경 공통 설정 — setup_*.sh 가 source 한다.
# 전 환경 공통: Python 3.10, torch 2.9.0+cu128 (RTX 5090 = sm_120 은 CUDA 12.8 이상 필요,
# 메인 foundgrasp 환경과 같은 torch). 빌드 산출물은 레포 밖 BUILD_DIR 에 둔다.
set -euo pipefail
CONDA_BASE="$(conda info --base)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"          # RAM 30GB — 컴파일 병렬도 제한
BUILD_DIR="${BUILD_DIR:-$HOME/Workspaces/efg_build_deps}"
WS="${WS:-$HOME/Workspaces}"
TORCH_PKGS="torch==2.9.0 torchvision==0.24.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
mkdir -p "$BUILD_DIR"

env_py() { echo "$CONDA_BASE/envs/$1/bin/python"; }

create_env() {  # create_env <name> — Python 3.10 + torch(cu128), 이미 있으면 재사용
  local name="$1"
  if [ ! -x "$(env_py "$name")" ]; then
    conda create -y -n "$name" python=3.10
  fi
  "$(env_py "$name")" -m pip install --upgrade pip wheel "setuptools<70"
  "$(env_py "$name")" -m pip install $TORCH_PKGS --index-url "$TORCH_INDEX"
}

# gcc 13 libstdc++ + nvcc 12.8 의 __to_address 모호성 회피용 헤더 사본.
# AnyGrasp 안내는 /usr/include 를 직접 sed 하지만, 시스템 파일은 건드리지 않고
# 패치 사본을 CPATH 로 앞세운다 (빌드 프로세스에만 적용).
gcc_header_override() {
  local ver; ver="$(gcc -dumpversion | cut -d. -f1)"
  local src="/usr/include/c++/$ver/bits/shared_ptr_base.h"
  local dst="$BUILD_DIR/include_override/bits/shared_ptr_base.h"
  mkdir -p "$(dirname "$dst")"
  sed 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' "$src" > "$dst"
  echo "$BUILD_DIR/include_override"
}
