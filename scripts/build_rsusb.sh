#!/usr/bin/env bash
# 현재 Python용 SDK 2.58.2를 사용자 캐시에 설치한다. sudo/전역 설치 없음.
set -euo pipefail
rsusb_root="${HOME}/.cache/foundgrasp-rsusb"
rsusb_python="${RSUSB_PYTHON:-$(command -v python)}"
rsusb_prefix="${FOUNDGRASP_RSUSB_PREFIX:-${rsusb_root}/install}"
rsusb_cmake="${rsusb_root}/build-tools/cmake/data/bin/cmake"
mkdir -p "${rsusb_root}"
if [ ! -x "${rsusb_cmake}" ]; then
    "${rsusb_python}" -m pip install --target "${rsusb_root}/build-tools" 'cmake==4.4.4'
fi
if [ ! -d "${rsusb_root}/librealsense/.git" ]; then
    git clone --branch v2.58.2 --depth 1 \
        https://github.com/realsenseai/librealsense.git "${rsusb_root}/librealsense"
fi
if [ "$(git -C "${rsusb_root}/librealsense" rev-parse HEAD)" != "5330e760a266d19c7e1e7ec7ada8caa4ff8b3196" ]; then
    echo "SDK 소스 버전이 v2.58.2와 다릅니다: ${rsusb_root}/librealsense" >&2
    exit 1
fi
# Ubuntu의 기존 libusb 런타임을 사용한다. 개발용 헤더만 별도로 가져온다.
rsusb_usb_library="$(c++ -print-file-name=libusb-1.0.so.0)"
if [ ! -f "${rsusb_usb_library}" ]; then
    echo "libusb-1.0 런타임 라이브러리가 필요합니다." >&2
    exit 1
fi
mkdir -p "${rsusb_root}/include"
if [ ! -f "${rsusb_root}/include/libusb.h" ]; then
    curl --fail --location --retry 3 \
        https://raw.githubusercontent.com/libusb/libusb/v1.0.27/libusb/libusb.h \
        -o "${rsusb_root}/include/libusb.h.tmp"
    mv "${rsusb_root}/include/libusb.h.tmp" "${rsusb_root}/include/libusb.h"
fi
"${rsusb_cmake}" -S "${rsusb_root}/librealsense" -B "${rsusb_root}/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="${rsusb_python}" -DPython_EXECUTABLE="${rsusb_python}" \
    -DPYTHON_INSTALL_DIR="${rsusb_prefix}/python" \
    -DCMAKE_INSTALL_PREFIX="${rsusb_prefix}" -DCMAKE_INSTALL_LIBDIR=lib \
    -DLIBUSB_LIB="${rsusb_usb_library}" -DLIBUSB_INC="${rsusb_root}/include" \
    -DUSE_EXTERNAL_USB=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_GLSL_EXTENSIONS=OFF -DBUILD_TOOLS=OFF \
    -DBUILD_WITH_DDS=OFF -DCHECK_FOR_UPDATES=OFF
"${rsusb_cmake}" --build "${rsusb_root}/build" --parallel "${RSUSB_BUILD_JOBS:-4}"
"${rsusb_cmake}" --install "${rsusb_root}/build"
echo "RSUSB 설치 완료: ${rsusb_prefix}"
