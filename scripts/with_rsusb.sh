#!/usr/bin/env bash
# 별도 설치한 RSUSB Python 바인딩을 이 명령에만 적용한다.
set -euo pipefail
rsusb_prefix="${FOUNDGRASP_RSUSB_PREFIX:-${HOME}/.cache/foundgrasp-rsusb/install}"
if ! compgen -G "${rsusb_prefix}/python/pyrealsense2*.so" > /dev/null; then
    echo "RSUSB 설치를 찾을 수 없습니다: ${rsusb_prefix}" >&2
    exit 1
fi
if [ "$#" -eq 0 ]; then
    echo "사용법: bash scripts/with_rsusb.sh python scripts/check_camera.py --seconds 60" >&2
    exit 2
fi
export PYTHONPATH="${rsusb_prefix}/python${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${rsusb_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$@"
