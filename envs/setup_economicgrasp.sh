#!/usr/bin/env bash
# efg-economicgrasp: EconomicGrasp 워커 환경.
#   MinkowskiEngine(chenxi-wang 포크, AnyGrasp 와 공유하는 휠) + EconomicGrasp libs 의 pointnet2/knn.
#   코드: $WS/FoundGrasp/ext/EconomicGraspPrivate (논문 벤치마크와 같은 사본)
source "$(dirname "$0")/common.sh"
ENV=efg-economicgrasp
create_env "$ENV"
PY="$(env_py "$ENV")"
bash "$(dirname "$0")/build_minkowski.sh" "$ENV"
"$PY" -m pip install "numpy==1.23.5" scipy "open3d==0.19.0" Pillow tqdm pyyaml tensorboard
"$PY" -m pip install --no-deps "git+https://github.com/seungback/graspnetAPI.git"
"$PY" -m pip install transforms3d trimesh scikit-image "opencv-python-headless<4.12" h5py autolab_core cvxopt dill grasp_nms || true
LIBS="$WS/FoundGrasp/ext/EconomicGraspPrivate/libs"
for op in pointnet2 knn; do
  rm -rf "$BUILD_DIR/eco_$op" && cp -r "$LIBS/$op" "$BUILD_DIR/eco_$op"
  (cd "$BUILD_DIR/eco_$op" && CPATH="$(gcc_header_override)" "$PY" setup.py install)
done
"$PY" -c "import torch, MinkowskiEngine as ME, pointnet2._ext, knn_pytorch; print('OK', torch.__version__, ME.__version__, torch.cuda.is_available())"
