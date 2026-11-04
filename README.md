# exp_foundgrasp

FoundGrasp 파지 검출을 **실제 로봇(xArm7 + RealSense D435 + KORAS 그리퍼)** 에서
실행하는 파지-이적재 실험 플랫폼. 물체 박스를 스캔해 그립을 검출·선정하고,
집어서 릴리즈 박스로 옮기는 루프를 반복한다. 검출 백엔드가 교체 가능한 구조라
다른 파지 모델(비교군)을 같은 하드웨어·절차로 비교 실험할 수 있다.

<p align="center"><img src="docs/sample_grasps.jpg" width="640" alt="검출 후보(주황)와 최고 점수 그립(빨강)"></p>

```
스캔 자세 → 캡처 → 검출(FoundGrasp) → ROI 필터 → 선정(IK·수직성·카메라 상향)
→ pre-grasp(그리퍼 개방 병행) → 저속 접근 → 파지 → 상승 → 상방 경유 → 릴리즈 → 반복
```

## 구성

```
exp_foundgrasp/
├── run.py                  실행 진입점 (python run.py [--auto|--dry-run])
├── configs/
│   ├── default.yaml        모든 설정 (로봇·카메라·그리퍼·선정·장면·옵션)
│   └── calib/              hand_eye.yaml, intrinsics.yaml, box_floor.npz, payload_baseline.npz
├── efg/                    핵심 패키지
│   ├── detectors/          파지 검출 백엔드 — 비교군 확장 지점 (base.py 계약)
│   ├── camera / robot / gripper / motion   하드웨어
│   ├── transforms / selection / roi / floor / payload / viz   파이프라인
│   └── demo.py             파지-이적재 루프
├── scripts/                사전 준비·점검 도구 (아래 "빠른 시작")
├── tests/                  단위 테스트 (하드웨어 불필요) + tests/data/sample_frame.npz
├── third_party/
│   ├── FoundGrasp/         원본 사본 — 수정하지 않음
│   └── koras-gripper-wrapper/  그리퍼 드라이버 원본 사본 — 수정하지 않음
└── outputs/                실행 산출물 (git 미추적)
```

## 요구 사항

| 항목 | 내용 |
|---|---|
| 로봇 | UFACTORY xArm7 (SDK 1.18.4, 이더넷). 설정의 **TCP offset = 그리퍼 끝**을 연결 시 적용·확인. **TCP payload = 실제 그리퍼 질량·무게중심** |
| 카메라 | Intel RealSense D435, 손목 장착(eye-in-hand). hand-eye 캘리브레이션 결과 필요 |
| 그리퍼 | KORAS 2FPA-I-03-100 (Modbus RTU, USB 시리얼). 계정이 `dialout` 그룹이어야 함 |
| GPU | CUDA GPU (FoundGrasp 추론) |
| 환경 | conda `foundgrasp` (Python 3.10, torch 2.9 cu128, numpy<2) |

## 설치

```bash
conda create -n foundgrasp python=3.10 -y && conda activate foundgrasp
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy<2" scipy opencv-contrib-python pillow tqdm matplotlib albumentations h5py \
            transformers omegaconf huggingface_hub open3d pyyaml pytest
pip install git+https://github.com/seungback/graspnetAPI.git
pip install pyrealsense2 xarm-python-sdk pymodbus pyserial rerun-sdk==0.33.1 trimesh
```

- FoundGrasp 체크포인트는 git에 없다. `third_party/FoundGrasp/logs/final/foundgrasp_base/checkpoints/final_model.pth`
  에 배치 (경로는 `detector.foundgrasp.checkpoint`). 백본 가중치는 첫 실행 시 HuggingFace 캐시로
  받는다 (오프라인 강제: `HF_HUB_OFFLINE=1`).
- `opencv-python`과 `opencv-contrib-python`을 함께 설치하지 말 것.

설치 확인 (카메라·로봇 불필요):

```bash
python -m pytest tests/ -q
python scripts/run_inference.py --npz tests/data/sample_frame.npz --vis outputs/grasps.jpg
```

## 빠른 시작

하드웨어 배치가 바뀌었을 때 순서대로 한 번씩. 각 스크립트는 결과를 `configs/default.yaml`에 기록한다.

```bash
conda activate foundgrasp && cd exp_foundgrasp

python scripts/set_roi.py            # 1) 스캔 영역: 카메라 이미지에서 물체 박스 내부 드래그
python scripts/align_scan_pose.py    # 2) 스캔 자세: 카메라를 박스 중심 수직 상방으로 (자동)
python scripts/teach_scan_pose.py    #    또는 수동 — 라이브 화면 보며 조정 후 s
python scripts/set_roi.py            # 3) 시점이 바뀌었으므로 스캔 영역 재지정
python scripts/teach_drop.py         # 4) 릴리즈 자세: 릴리즈 박스 상방으로 옮기고 Enter

python run.py --dry-run              # 5) 로봇 없이 스캔·검출·선정 확인
python run.py                        # 6) 실행 (pre-grasp 이동 중 폭 조절 후 자동 접근)
python run.py --auto                 #    무인 연속 실행
```

실행 중: 라이브 창에서 `q` = 정지 요청(홈 복귀 후 종료). ROI에 후보가 없거나 연속 실패
시 터미널에서 대기(Enter 재개 / q 종료). 이동 실패로 중단되면 로봇은 그 자리에 정지한다
(자동 복귀가 2차 충돌을 만들 수 있음 — xArm Studio로 확인 후 복귀).

## Rerun 뷰어

기본 설정(`rerun.viewer: native`)에서는 실행 시 별도의 데스크톱 창이 자동으로 열린다.
`.rrd` 기록 저장은 유지하며, 데이터 연결은 웹 서버와 겹치지 않는 9877 포트를 사용한다.

브라우저 방식이 필요하면 `python run.py --set rerun.viewer=web`으로 실행한다.
웹 모드에서는 아래 **동일한 주소**로 접속한다. 전체 링크를 북마크하면 된다.

[Rerun 열기](http://127.0.0.1:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy)
웹 포트는 `rerun.web_port: 9090`, 데이터 포트는 `rerun.grpc_port: 9876`이다.
`python run.py`가 서버를 시작하고, 다음 실행에서는 재사용한다. 실행이 끝나도 서버는
남아 있으며 기록은 실행별로 분리된다. `/` 주소만 열면 데이터에 자동 연결되지 않을 수
있으므로 위 전체 URL을 사용한다. 서버 시작 실패 로그는 `outputs/rerun_server.log`에 남는다.

로봇 외형은 제조사 xArm7 모델을 GLB로 변환해 부품별 흰 외장·회색 금속 재질을 보존한다.
원본에는 이미지 텍스처가 없다. 출처·라이선스는 `third_party/xarm7_visual/`에 있고,
재생성 도구는 `scripts/build_xarm_visuals.py`이다. 기구학은 기존 MJCF를 사용한다.

로봇 자세와 스캔·파지 데이터는 공통 `elapsed` 시간축으로 기록하며, 기본 뷰는
최신 시점을 따라간다. 관절각은 SDK 캐시에서 10Hz로 읽으므로 이동 중에도 갱신된다.
기존 뷰어 설정이 남아 있으면 시간축을 `elapsed`로 선택하고 최신 시점 따라가기를 켠다.
`Scan snapshot (not live)`는 홈 스캔 당시의 이미지·점군이며 로봇 이동 중에도 고정된다.
`Robot / scene`에는 고정 스캔과 함께 이동 중 **최근 3프레임**의 라이브 점군도 표시한다.
`rerun.live_cloud_history`로 N을 바꾸고, 0이면 끈다. 기본 갱신은 5Hz,
픽셀 간격은 4 (`live_cloud_hz`, `live_cloud_stride`)이며 오래된 프레임부터 교체한다.
기존 카메라 공유 메모리와 로봇 보고 콜백을 사용하므로 카메라를 추가로 열거나
로봇 조회 명령을 보내지 않는다. 프레임 수신 시각을 앞뒤로 감싸는 로봇 보고 자세를
보간한 뒤 TCP 오프셋을 제거하고 hand-eye를 적용한다. 이는 **호스트 수신 시각 기준
근사 정렬**이므로 빠른 이동·depth temporal filter 잔상에 따라 점군이 어긋날 수 있다.
시간이 맞지 않거나 오래된 프레임은 버린다. 라이브 점군은 시각화 전용이며,
검출·후보 선정·모션은 기존 파지용 캡처를 사용한다. `Grasp detail`도 파지용 스캔만 표시한다.
화면에 보이는 개수는 N으로 제한되지만 `.rrd`에는 갱신 이력이 쌓여 기록 용량이 증가한다.
`--dry-run`에서는 실측 로봇 자세가 없으므로 라이브 점군을 시작하지 않는다.
후보는 점수 순으로 12mm 이내 중복 지점을 덜어내 최대 25개만 표시한다
(`rerun.max_candidates`, 검출·제어 후보 수와 무관). `Grasp detail` 탭은 로봇을
제외한 파지 장면을 확대해 볼 수 있다.
Matplotlib `RdYlGn`으로 표시 후보의 점수 최저→최고를 빨강→노랑→초록으로 매핑하고 범례에 범위를 적는다.
색은 현재 표시 집합 내 상대 점수이며 성공 확률이 아니다. 같은 규칙이 2D/3D와 저장 이미지에 적용된다.
1위는 화면 크기 기준 굵은 선과 별도 라벨로 표시하고, 2D에서는 흰색 외곽선과 최상위
표시 순서를 적용한다. 실행 대상으로 선정된 그립은 청록색이다. 성공·실패 색/지표는 표시하지 않는다.
상태 패널에는 홈 복귀·스캔·접근·닫기·상승·이송·놓기 등 동작 단계가 표시된다.
그리퍼 메쉬는 모서리를 다듬은 본체, 금속 핑거, 고무 팁의 근사 형상이며 제조사 CAD가 아니다.
개방 폭은 기존 제어 루프의 마지막 상태 판독값과 경과 시간을 표시한다(추가 Modbus 읽기 없음).
dry-run에서는 측정값이 없으므로 목표 폭 미리보기로 명시한다.

## 카메라 프로세스와 추론 초기화

기본 `calib.hand_eye`는 캘리브레이션 프로젝트의 `outputs/hand_eye_latest.yaml`을
직접 가리킨다. `calib.intrinsics: hand_eye`이면 같은 파일의 `camera_intrinsics`에서
K/D를 읽는다. 새 캘리브레이션 저장 후 프로그램을 재시작하면 적용되며, 별도 복사는
필요 없다. 다른 장비에서는 `calib.hand_eye` 경로를 수정한다. 파일이 없으면 오류로
종료하며 과거 값으로 자동 대체하지 않는다. 기존 분리형 intrinsics YAML 경로도 지원한다.

`calib.undistort: true`가 기본이다. RealSense color 정렬과 depth 필터 뒤에,
캡처 프로세스에서 보정 K/D로 RGB·depth를 동일하게 remap한다. RGB는 선형 보간,
depth는 최근접 보간(깊이 경계의 가짜 표면 방지), 영상 밖은 0이다. 출력 크기와 K는
유지하고 광학 좌표축을 회전하지 않으므로 hand-eye 변환은 그대로 사용한다.
추론·점군·Rerun·라이브 프리뷰·ROI 지정은 모두 이 보정 영상 좌표를 사용한다.
기존 `scan_roi_px` 숫자는 유지되지만 영상 내용이 보정되므로 박스 경계를 확인한다.
보정은 캐시된 맵을 재사용하며 크기 불일치·보정 파일 누락 시 조용히 우회하지 않는다.

현재 K/D는 `calibrate_intrinsics.py`의 OpenCV 추정값이다. 이 스크립트가 복사한
이전 RealSense `distortion.inverse_brown_conrady` 이름보다 생성 출처를 우선하여
OpenCV 계수로 해석한다. 출처가 확인되지 않은 inverse-Brown 계수는 거부한다.
새 NPZ에는 `rectified` 표시를 저장한다. `run_inference.py`는 예전 무표시 파일만
보정하고, 이미 보정된 파일은 저장된 영상/K를 유지해 이중 보정을 막는다.
`--set calib.undistort=false`는 기존 미보정 캡처 동작으로 돌아가는 옵션이며,
이미 보정된 NPZ를 원본 영상으로 되돌리지는 않는다.

`run.py`는 카메라 SDK의 생성·캡처·정렬·depth 필터·종료를 별도 프로세스에서 수행한다.
메인 프로세스는 공유 메모리의 최신 RGBD 한 장만 복사한다. 캡처 시각 기준 2초 이상
오래된 프레임이나 카메라 프로세스 오류가 있으면 다음 접근·자동 홈 복귀를 차단한다.
SDK의 암묵적인 재연결 대신 프레임 수신 기한을 적용하며, 종료가 멈추면 카메라
프로세스만 종료한다. USB 장치 자체의 단절을 자동 복구하거나 동작을 재개하지는 않는다.

최초 CUDA 초기화 비용은 `detector.warmup_runs: 1`로 로봇 연결 전에 처리한다.
워밍업 결과는 파지 선정에 사용하지 않는다. 로그의 `검출` 시간에는 refiner가 포함되며,
공유 충돌검사·선정·로봇 이동 시간은 포함되지 않는다.
실제 실행은 로봇 홈 복귀를 완료한 뒤 그리퍼 연결·방향 점검을 진행한다.
홈 복귀 실패 시 그리퍼 점검을 시작하지 않는다. 첫 사이클은 이 홈 자세에서 바로
스캔하며 초기 홈 복귀·그리퍼 점검은 택타임에서 제외한다. 이후 사이클은 홈 복귀부터 측정한다.
`--dry-run`은 로봇 홈 이동과 그리퍼 점검을 모두 생략한다.

모델과 로봇 없이 카메라 연결만 확인하려면:

```bash
python scripts/check_camera.py --seconds 60
```

Linux UVC 경로의 연결 오류를 우회해 확인하려면 별도 RSUSB 설치를 사용할 수 있다.
SDK 2.58.2를 `~/.cache/foundgrasp-rsusb/install`에 빌드하며, 기존 Conda 패키지는
덮어쓰지 않는다. 빌드에는 C/C++ 컴파일러, make, git, curl, libusb 런타임이 필요하다.

```bash
conda activate foundgrasp
bash scripts/build_rsusb.sh
bash scripts/with_rsusb.sh python scripts/check_camera.py --seconds 60
# 카메라 확인 후 추론까지 확인 (로봇 연결/이동 없음)
bash scripts/with_rsusb.sh python run.py --dry-run
```

실제 로봇 실행에도 같은 접두사(`bash scripts/with_rsusb.sh`)를 붙인다.
접두사 없이 실행하면 기존 SDK를 사용한다. RSUSB는 커널 `uvcvideo` 대신 libusb를
사용하지만, 케이블·전원·USB 컨트롤러 등에서 발생하는 단절까지 해결을 보장하지는 않는다.
RealSense Viewer와 캡처 프로그램은 동시에 실행하지 않는다.

## 실행 옵션

실행 옵션 외의 동작은 설정 파일에서 정한다.

| | |
|---|---|
| `--auto` | 기본 동작과 동일 (호환 옵션) |
| `--confirm` | pre-grasp에서 정지하고 접근 확인 (블렌딩 비활성) |
| `--dry-run` | 스캔·검출·선정만, 로봇/그리퍼 미사용 |
| `--config PATH` | 다른 설정 파일 사용 |
| `--set KEY=VALUE` | 설정 항목 일시 변경 (반복 가능) |
| `--grasp-detector NAME` | 검출 모델 선택 (`detector.name` 별칭) |

```bash
python run.py --set live.enabled=false                   # 라이브 창 끄기
python run.py --set floor_limit.enabled=true             # 하강 제한 켜기
python run.py --set output.save_cycles=true              # 사이클 이미지 기록
python run.py --set selection.approach_max_tilt_deg=20 --auto
```

실험별 설정은 `base:`로 기본값을 상속해 바꿀 항목만 적는다:

```yaml
# configs/exp_tilt20.yaml
base: default.yaml
selection: {approach_max_tilt_deg: 20}
output: {dir: outputs/tilt20, save_cycles: true}
```

## 설정 요약 (`configs/default.yaml`)

| 그룹 | 주요 항목 |
|---|---|
| `detector` | `name`(백엔드), `device`, `foundgrasp.{config, checkpoint, top_k, collision_detection}` |
| `calib` | `hand_eye`(T_flange_cam), `intrinsics`(캘리브레이션 K/D), `undistort`(RGB-D 왜곡 보정) |
| `robot` | `ip`, `tcp_offset_mm_deg`(플랜지→팁 끝 중앙, XYZ mm/RPY deg), `home_joints_deg`(스캔 자세), 이동·접근 속도/가속도 |
| `camera` | 해상도·fps, `postprocess`(RealSense depth 필터) |
| `gripper` | `port`(by-id 고정 경로), 힘/속도, `pre_open_margin_mm`, `grip_success_min_pct`, `tip_length_mm`(시각화) |
| `selection` | 작업공간, `approach_max_tilt_deg`, 삽입 깊이(`approach_extra_m`), `j7_limit_deg`, `camera_up` |
| `scan_roi_px`, `drop` | 스캔 영역, 릴리즈 자세·이송 속도·경유 높이 |
| `grip_check`, `payload_check`, `floor_limit` | 아래 "선택 기능" |
| `live`, `output` | 라이브 창, 산출물 폴더·사이클 이미지 저장 |

## 선택 기능 (기본 꺼짐)

**하강 제한 `floor_limit`** — 빈 박스를 한 번 스캔해 높이맵을 만들고, 선정 그립의 TCP·핑거
끝이 `표면 + clearance` 아래로 가지 않게 삽입량을 줄인다 (부족하면 후보 탈락·재선정).
박스 CAD 없이 바닥·벽(경사 포함) 충돌을 막는다.
```bash
python scripts/scan_box_floor.py        # 빈 박스 스캔 → configs/calib/box_floor.npz
python run.py --set floor_limit.enabled=true
```

**페이로드 파지 판정 `payload_check`** — 릴리즈 자세에서 관절 토크를 빈 손 기준값과 비교해
"손에 질량이 있는가"로 성공을 판정한다. 플렉스 팁처럼 finger 간격으로 판정할 수 없는
그리퍼용. 릴리즈 자세를 다시 티칭하면 재측정.
```bash
python scripts/calib_payload.py         # 빈 손 + 샘플 물체 → configs/calib/payload_baseline.npz
python run.py --set payload_check.enabled=true
```

**파지 성공 판정 `grip_check`** — **기본 꺼짐**: 판정값(finger %, 페이로드 Δ)을 출력만 하고 항상 이송을
진행한다 (플렉스 팁은 finger 간격 판정이 부정확하므로). `enabled: true`로 켜면 finger 간격
(`gripper.grip_success_min_pct`) + 켜져 있는 경우 페이로드 판정으로 실패를 걸러 재시도한다.
꺼진 상태에선 실패 재시도가 동작하지 않으므로 `--auto`와 함께 쓸 때 주의.

## 비교군(다른 파지 모델) 추가

AnyGrasp는 기존 레지스트리·상주 워커를 사용합니다. SDK 코드는
`~/Workspaces/anygrasp_sdk`, 가중치는 `ckpts/checkpoint_detection.tar`, 라이선스는
`license_registration/license`에 둡니다. 원본 SDK를 수정하지 않고 워커 임시 폴더에
Python 버전에 맞는 바이너리와 라이선스 디렉터리 링크를 만들어 로드합니다.

```bash
# 최초 설치: foundgrasp 환경과 분리된 efg-anygrasp 환경
bash envs/setup_anygrasp.sh

# 저장 프레임으로 검사 — 로봇/카메라 연결 없음
python scripts/run_inference.py --config configs/benchmark.yaml \
  --grasp-detector anygrasp --npz outputs/live_frame.npz \
  --warmup 3 --repeat 5 --record-dir outputs/anygrasp_check \
  --out outputs/anygrasp_check/grasps.npy --vis outputs/anygrasp_check/overlay.jpg

# 동일 설정에서 AI만 교체: 위 명령의 --grasp-detector foundgrasp
# 실제 실행 (기존 캘리브레이션·모션·그리퍼·재시도 설정 상속)
python run.py --config configs/benchmark.yaml --grasp-detector anygrasp
```

`--grasp-detector`는 `detector.name`을 덮어쓰는 별칭이며 `--set detector.name=...`도
계속 지원합니다. 기본 `configs/default.yaml`의 FoundGrasp 독립 추론 동작은 유지합니다.
`configs/benchmark.yaml`은 현재 **FoundGrasp / AnyGrasp**용 공통 실행 프로토콜입니다.
EconomicGrasp/RNGNet은 이 프로토콜의 검증 대상에 아직 포함하지 않습니다.

공통 설정은 같은 입력 ROI·실측 K·폭 한계·공유 충돌 검사(voxel 1cm, 접근 5cm,
thresh 0.01)·ROI 필터·IK·대칭 선정·모션을 사용합니다. FoundGrasp의 최종 NMS/top-50과
AnyGrasp 데모의 외부 NMS를 켜고, 각 모델의 내부 후보 생성·refiner/내장 CD를 유지합니다.
SDK 내부 처리는 비공개이므로 동일한 내부 후보 수를 보장하지 않습니다.
실장비 설정은 refiner 측정면 검사에 `VERIFIER_BODY: center`를 사용합니다.
5mm 격자 셀이 몸체에 조금만 걸쳐도 포함하는 원본 `overlap` 대신, 셀 중심이 몸체 안에
있는 경우 검사합니다. 얇은 물체에서 격자 근사로 인한 과도한 기각을 줄이기 위한 변경이며,
원본 판정을 재현하려면 `detector.foundgrasp.refiner.env.VERIFIER_BODY=overlap`으로 설정합니다.
측정면 거리 2.5mm와 refiner 신경망·공유 충돌 검사는 유지합니다.
공통 설정의 AnyGrasp 검사 높이는 기존 FoundGrasp 후보와 같은 20mm로 지정합니다.
FoundGrasp 최종 폭은 그대로 실행하며, AnyGrasp는 SDK 출력 폭에
`postprocess.open_multiplier`(기본 1.2)를 곱한 뒤 개방 한계로 제한합니다.
예: AnyGrasp 60mm → 72mm, 90mm → 100mm(한계). benchmark 설정도 같은 값을 상속합니다.
AnyGrasp의 숨은 x1.2 마진을 추정해 나누던 코드는 제거했습니다.
이는 실제 장비의 공통 실행 비교이며, 시뮬레이터의 raw-width × 1.2 규약을 그대로
재현했다는 의미는 아닙니다. 기존 단독 실행 결과와 공통 프로토콜 결과를 구분해 기록하세요.

`efg/perception.py`는 실제 실행과 저장 프레임 검사에 같은 인식·후처리 경로를 제공합니다.
`efg/recording.py`는 실제 설정, 코드/가중치/캘리브레이션 해시, 입력 RGB-D/K,
필터 전후 후보와 단계별 개수·시간을 기록합니다. `output.save_cycles=true`이면
실행에서도 사이클/재추론별로 같은 산출물을 저장합니다.
AnyGrasp의 **워커 내부 시간**은 전처리+SDK+디코딩/고유 필터를 CUDA 동기화 후 측정하고,
**검출 시간**은 파일 통신·폴링까지 포함합니다. 공통 후처리 시간은 별도입니다.
모델 로딩과 워밍업은 측정에서 제외됩니다. 이송 완료를 실제 파지 성공으로 간주하지 않습니다.

검출 백엔드 하나만 구현하면 ROI·좌표 변환·선정·모션·판정은 그대로 쓰인다.

1. 원본 코드를 `third_party/<Name>/`에 두고(무수정), `efg/detectors/<name>.py`에 구현:
   ```python
   from ..paths import THIRD_PARTY, add_vendor_paths
   from .base import GraspDetector

   class MyDetector(GraspDetector):
       name = "mymethod"
       def __init__(self, cfg, device="cuda"):
           super().__init__(cfg, device)
           add_vendor_paths(THIRD_PARTY / "MyMethod")
           ...  # cfg = 설정의 detector.mymethod 블록
       def detect(self, rgb, depth_m, K, top_k=None):
           ...  # → graspnetAPI GraspGroup (카메라 좌표계, score 내림차순)
   ```
   계약: 입력 `rgb (H,W,3) uint8`, `depth_m (H,W) float32 m (무효=0)`, `K (3,3)`; 출력은
   graspnetAPI 규약(회전행렬 1열=접근, 2열=개폐, width·depth 미터).
2. `efg/detectors/__init__.py`의 `REGISTRY`에 `"mymethod": ("efg.detectors.mymethod", "MyDetector")` 추가.
3. 설정에 `detector.mymethod: {...}` 블록을 두고 `detector.name: mymethod` (또는 `--set detector.name=mymethod`).

오프라인 검증: `python scripts/run_inference.py --npz tests/data/sample_frame.npz --set detector.name=mymethod --vis outputs/my.jpg`

## 운용 규칙 · 안전

- **그리퍼**: 전원 투입 후 본체 버튼(I)으로 초기화. 소프트웨어 Modbus 초기화(101)는 close 방향
  학습을 손상시키므로 금지 (`initialize_on_connect: false` 유지). 연결 시 개방·close 방향을
  실측 점검해 이상 상태로 로봇이 움직이는 것을 차단한다.
- **로봇**: 확인 프롬프트가 필요하면 `--confirm`으로 실행하고 비상정지에 손을 올린 채 진행.
  TCP offset·payload가 실제 그리퍼와 다르면 삽입 깊이·충돌 감지(C31)가 어긋난다.
- **캘리브레이션**: 카메라 마운트를 다시 조이거나 위치가 바뀌면 hand-eye 재캘리브레이션
  (별도 도구 `r_foundgrasp/02.dev/hand_eye_calib`) 후 `configs/calib/hand_eye.yaml`·`intrinsics.yaml` 교체.
- **팁·툴 교체 시**: TCP offset/payload, `gripper.stroke_mm`, `selection.max_grasp_width_m`,
  파지 판정 방식(`grip_success_min_pct` 또는 `payload_check`) 재검토.

## 점검 도구

```bash
python scripts/camera_preview.py                 # D435 라이브 (s: 프레임 npz 저장)
python scripts/gripper_test.py [--status]        # 그리퍼 연결·개폐 / 상태만
python scripts/gripper_direction_check.py        # 펌웨어 방향 반전 여부 실측 판정
python scripts/run_inference.py --live --vis outputs/live.jpg   # 검출만
```

## 테스트

```bash
python -m pytest tests/ -q     # 38건: 변환·선정 기하, 그리퍼 시퀀스(모의), 페이로드 판정,
                               #       하강 제한, 설정 로드/상속/--set, 검출기 레지스트리
```

## 트러블슈팅

| 증상 | 조치 |
|---|---|
| `그리퍼 포트 열기 실패` | `ls -l /dev/serial/by-id/`로 경로 확인 → `gripper.port`. 권한: `sudo usermod -aG dialout $USER` 후 재로그인 |
| `grip_close 방향 점검 실패 (반전/무동작)` | 펌웨어 방향 손상 — 24V 재투입 + 핑거 자유 상태에서 버튼(I) 초기화. 임시 우회 `gripper.grip_mode: "position"` |
| `후보 N개 → ROI 내 0개` 반복 | ROI가 박스 안쪽인지, 스캔 자세 변경 후 `set_roi.py`를 다시 했는지 확인 |
| 선정 후보 전부 `작업공간 밖` | `--dry-run`은 로봇 자세가 없어 정상. 실행 시엔 `selection.workspace_*_m`·hand-eye 확인 |
| `릴리즈 자세가 기준값 측정 자세와 다름` | `teach_drop.py` 후 `calib_payload.py` 재실행 |
| C31 충돌 감지 | 주변 물체 걸림 여부 확인. 실제 접촉이 없으면 TCP payload 질량·무게중심과 가속도 점검. 닫힘 시간 초과와는 다른 오류 |
| 백본 다운로드 시도 | 오프라인이면 `HF_HUB_OFFLINE=1` |

## 설계 근거 (요약)

- **ROI = depth 마스킹 + 투영 필터**: 전체 프레임 추론은 top-k가 ROI 밖 물체 더미에 몰려 ROI 내
  0개가 되는 문제가 실측됨. ROI 밖 depth를 0으로 두면 인공 표면 없이 후보가 ROI 안에서만 생긴다.
- **선정**: score 순으로 첫 실행 가능 후보 채택. 같은 후보의 두 파지 대칭 중
  관절 이동 비용이 작은 경로를 선택. tilt 상한 25°(29°에서 self-collision 실측).
- **모션**: 그리퍼 폭 조절은 pre-grasp 이동과 병행. 파지 후 pre-grasp까지 복귀해
  파지 방향을 유지하며 상승한 뒤 릴리즈 자세로 조인트 이송하며 자세 전환.
  제어된 종료만 홈 복귀.
- **재시도**: 실패 시 홈 복귀 후 전체 재스캔(물체가 밀렸을 수 있음), 직전 실패 지점 반경 제외.

접근은 pre-grasp 이동을 먼저 비차단 전송하고, 이동 중 그리퍼 폭을 조절합니다.
폭 도착을 확인한 뒤에만 grasp 명령을 전송합니다. 폭 조절이 늦으면 pre-grasp에서
대기할 수 있으며, 조절 중에도 카메라·중단 요청을 확인합니다.
`robot.pregrasp_blend_radius_mm`(기본 설정 20 mm)로 pre-grasp 부근을 둥글게 통과하고,
두 구간 모두 `robot.approach_acc_mm_s2`를 적용합니다. 최종 파지점은 정지 후
위치 2 mm·자세 1° 이내 도착을 확인해야 그리퍼를 닫습니다. 카메라 오류나
로봇 오류, 접근 시간 초과 시 정지하고 대기 명령을 취소합니다. 파지·릴리즈·스캔
정지는 유지됩니다. 상승→조인트 이송의 혼합 블렌딩은
`drop.transfer_blend_radius_mm: 0`으로 기본 해제합니다. FW 2.7.1 실험에서 20 mm
연결부 부근에 C24가 재발했으며, 감속 사전 검사 통과 후에도 실제 오류가 발생했습니다.
높은 경유점에서 정지·빈 큐·TCP 위치 2 mm/자세 1° 이내 도착을
`drop.transfer_confirm_s`(0.05초) 동안 확인한 뒤, 실제 관절 상태에서 아직 전송하지 않은
조인트 이송 단계를 준비합니다. 목표 관절과 정지 반경은 유지하고, 이 단계의 검사에서만
속도 초과가 나오면 준비된 이송 명령의 속도·가속도를 제한적으로 낮춰 재검사합니다.
검사 중 시작 관절이 변하거나 J7 제한을 벗어나면 이송을 보내지 않습니다.
0보다 큰 값은 검증되지 않은 혼합 연결 경로를 사용하므로
현재 장비의 오류 대응으로 다시 켜지 않습니다.
파지 후에는 파지 자세로 pre-grasp까지 역방향 복귀한 뒤, XY를 유지하며 상승합니다.
기본 `drop.lift_orientation: grasp`는 각 높이에서 먼저 검사할 자세입니다. 이 경로는
Cartesian 복귀·상승 동안 파지 방향을 유지하고 조인트 이송에서 릴리즈 자세로 전환합니다.
`drop.route_search: true`이면 높이와 상승 중 자세 전환 대안도 탐색합니다.
상승 목표를 180° 뒤집는 대안은 `grasp` 모드에서 허용하지 않습니다.
2026-11-04 기록에서 회전이 섞인 pre-grasp 연결부를 지나며 실측 J7이
−86°에서 −120°까지 변하고 C31이 발생했습니다. 자세 유지는 그 구간의 툴 회전을
없애기 위한 조치이며, 7축 로봇의 개별 관절각이 고정된다는 보장은 아닙니다.
상승 중에도 실측 J7을 계속 확인하며 범위를 벗어나면 큐를 정지합니다.
`upright`는 홈 툴 z축 방향으로 세우고, `home`은 홈의 전체 방향에 맞추는 기존 모드입니다.
이 두 모드는 상승 방향의 180° 대칭 후보도 검사합니다. pre-grasp 전환은
`robot.retreat_blend_radius_mm` 반경으로 연결합니다.
접근·복귀 반경은 인접 직선 구간 길이의 1/4 이하로, 상방 반경은 기준 높이까지의
상승 거리의 1/4 이하로 제한합니다. 반경을 키우면 경유점
주변에서 경로를 더 둥글게 통과합니다. `drop.via_height_m: 0.35`는 기존 기준 높이,
`drop.min_transfer_height_m: 0.25`는 사용자가 지정한 수평 이송 시작 높이의 하한입니다.
빈 높이 10cm와 적재물 높이를 고려해 25cm로 설정했습니다. 하한을 `null`로 설정하면
더 낮은 경로를 만들지 않습니다. 각 경유점은 `max(grasp 높이 + selection.lift_height_m,
해당 후보 높이, pre-grasp 높이 + 5 cm)`에 높이 여유와 유효 블렌딩 반경 중 큰 값을
더합니다. `drop.transfer_clearance_margin_mm`의 기본값은 10 mm이므로,
다른 하한이 더 높지 않으면 실제 이송 시작점은 260mm입니다.
이는 연결 코너의 높이 여유이며, 내려가는 릴리즈 조인트 경로 전체의 최저 높이나
박스·물체 충돌을 보장하지는 않습니다. 높은 경유점의 IK·J7도 파지 후보 선정 시
검사하며, 파지점 도착 후 **그리퍼를 닫기 전** 실측 자세에서 IK·J7을 다시 확인합니다.
경로 생성기는 허용된 낮은 높이부터 `drop.height_search_step_m` 간격으로 검사하고,
기존 기준 높이를 마지막에 포함합니다. 기본 간격은 20mm로, 다른 높이 하한이 없으면
실제 TCP 목표는 260·280·300·320·340·360mm입니다.
각 높이에서 설정 자세 → 기울임을 절반 줄인 자세 → 가장 가까운 수직 자세를 시도합니다.
기본 설정에서는 최대 18개 경로이며, 실측 재검사는 이전에 통과한 경로를 우선합니다.
경로가 같은 대안은 중복 검사하지 않습니다. 높이는 최대 64개까지 허용하며,
간격이 너무 작아 이를 넘으면 오류로 알려 줍니다. 높은 경로를 임의로 잘라내지 않습니다.
자세를 바꾸는 대안에는 pre-grasp 위의 짧은 수직 경유점을 둡니다. 경유점 블렌딩을
인접 구간 길이의 1/4 이하로 제한해 pre-grasp 아래에서 회전이 시작되지 않도록 합니다.
모든 Cartesian 상승 경로는 XY를 유지하며 위쪽으로만 움직입니다. 선택한 경로와
실제 이송 시작 Z를 로그에 출력합니다.
`upright/home`의 명시적 설정에서는 기존 상승 대칭 검사도 유지합니다.
카메라 하향 제한과 IK·J7 조건은 동일합니다. 상승 경로의 중간 자세도 병진 20mm·회전 5°
중 더 촘촘한 간격으로 검사합니다 (`robot.rise_ik_step_mm`, `rise_ik_step_deg`).
중간 IK/J7 검사 실패 시 선정 단계에서는 다른 대칭 조합·후보를 검토하고,
파지점에서의 계획 실패 시에는 닫힘·상승 명령을 보내지 않습니다.
자동 실행은 `pick_plan.py`에서 **접근 전에** 후보의 접근→복귀→상승→이송 전체를 계획합니다.
`transfer_plan.py`는 복귀 경로 대안을 생성·검사하고 `motion.py`는 확정된 명령을 실행합니다.
바닥 제한으로 파지점을 수정할 경우 수정된 좌표를 기준으로 계획합니다.
좌표·속도·가속도·반경은 불변 `MotionCommand` 목록으로 구성하고, 전체 명령열을
컨트롤러의 비구동 검사 `only_check_type=1 → 3 → … → 2`로 확인합니다
(FW 1.11.100 이상, Mode 0). 통과한 `PickPlan`의 접근 명령을 그대로 실행합니다.
같은 파지의 복귀 대안이 모두 불가능하면 반대 파지 대칭·다음 score 후보를 검사합니다.
각 단계에서 통과한 동일 명령을 `only_check_type=0`으로 실행하며, 전역 검사 모드를 변경하지 않습니다.
사전 검사만 속도 초과(`code=1, only_check_result=24`)로 기각되면 전체 명령열의 속도
상한을 기본 계획값의 80% → 60% → 50%, 가속도 상한을 각각 64% → 36% → 25%로 낮춰
처음부터 다시 검사합니다. 매 검사 전 실제 로봇의 정지·오류 없음·시작 자세 유지를 확인하고,
목표 좌표와 블렌딩 반경은 유지합니다. 통과한 배율은 로그에 표시하며 다음 사이클은
다시 원래 설정값으로 시작합니다. 이는 명령 상한 조정이며, 컨트롤러 내부 jerk와
실제 궤적의 정확한 시간 스케일링을 보장하지 않습니다.
속도 외 오류에는 감속 반복을 하지 않습니다. C21(기구학 경로 불가)은 다른 경로를
찾아야 하는 결과입니다. 사전 검사에서만 발생한 C21/22/23/24와 정상 실제 로봇 상태가
확인되면 다른 대안을 검사합니다. 알 수 없는 검사 결과·통신 오류·실제 컨트롤러 오류는
계획 탈락으로 숨기지 않습니다. 접근과 이송 중에도 실측 J7 범위를 확인합니다.
초기 전체 검사의 상승 관절값은 가상 계획 상태이므로, 실제 상승 도착 후의 검사 결과와
다를 수 있습니다. 상승 도착 후에는 현재 관절을 고정된 시작점으로 다시 기록하고,
미전송 조인트 명령에 같은 100% → 80% → 60% → 50% 속도 배율(가속도는 배율의 제곱)을
적용합니다. 최대 네 번 검사하며 최저값은 **초기 검사에서 준비된 이송 명령의 50%**입니다.
예를 들어 초기 검사를 통과한 이송 속도가 96°/s라면 이 단계의 한도는 48°/s입니다.
`이송 계획` 로그에는 실제 적용 속도·가속도를 출력해 중첩된 배율을 혼동하지 않게 합니다.
최저 한도에서도 실패하거나 속도 이외의 오류이면 이송하지 않습니다.
실행 전 시작 자세가 유지되는지 확인하며, 실제 조인트 명령 전송 후에는 재계획·재시도하지 않습니다.
추가 IK 질의 시간은 선정·상승 계획에 포함됩니다. 거부 시 복귀점/상승점/중간 경로와
IK 실패 또는 J7 제한 초과를 구분해 출력합니다.
IK 실패에는 SDK 원래 반환 코드와 검사한 TCP 목표(mm/rad)도 남깁니다.
`get_inverse_kinematics()`의 API 반환값 10(`RET_IS_INVALID`)은 유효한 IK 결과를
얻지 못한 후보로 처리합니다. 실제 로봇의 정지·오류 없음·빈 큐를 재확인한 뒤
다른 높이·자세를 검사하며, 모든 경로가 불가하면 다른 파지 후보를 선정합니다.
API 반환값 1(`HAS_ERROR`)이나 통신·파라미터·관절 데이터 오류는 중단합니다.
이는 비구동 모션 검사의 `code=1, only_check_result=21`과 별개의 반환 규약입니다.
FW 2.7.1의 독립 IK 질의에는 이전 표본 관절해를 시드로 전달할 수 없어, 홈에서의
후보 검사와 실제 파지점에서의 검사 결과가 달라질 수 있습니다. 독립 IK 실패가
반드시 물리적 도달 불가를 뜻하지는 않지만, 이를 무시하고 실행하지는 않습니다.
실제 파지점에서는 선택된 복귀 경로를 먼저 재검사하고 필요하면 나머지 대안도 찾습니다.
순수 계획 기각(`PlanRejected`)으로 대안이 모두 소진되었고 아직 그리퍼를 닫지 않았다면,
실측 파지점→pre-grasp→캡처 당시 스캔 TCP의 역방향 Cartesian 복귀를 새로 검사합니다.
정상·정지·빈 큐·시작 위치·J7 조건과 복귀 경로 검사를 통과한 경우에만 복귀합니다.
이 결과는 파지 실패가 아닌 `replan`으로 기록하고 해당 지점을 제외한 새 프레임에서 계속합니다.
실제 충돌/C31·통신/카메라 오류·사용자 정지에는 이 자동 복귀를 적용하지 않으며,
복귀 경로 자체가 불가하면 정지합니다. 보호 오류를 자동 해제하거나 검사 실패를 무시하지 않습니다.
자동 모드에서 후보가 소진되면 스캔 위치에서 `retry.no_candidate_wait_s`(2초) 대기 후
지점 제외 목록을 비우고 새 프레임으로 재스캔합니다. 즉시 재선정 시에는 제외를 유지하되,
한 차례 재추론까지 모두 소진한 뒤에는 장면 변경을 다시 평가합니다.
검사된 역방향 복귀나 추론만 반복한 경우에는 중복 홈 이동을 보내지 않습니다.
수동 확인 모드와 드라이런은 기존 사용자 재개 입력을 유지합니다.
이를 줄이기 위해 `selection.j7_selection_margin_deg: 10`으로 후보 선정에는
실행 한계 `[-89,90]°`보다 안쪽인 `[-79,80]°`를 사용합니다. pre-grasp·grasp·상승
경유점과 상승 내부 IK 표본에 모두 적용하며, 통과하지 못하면 이동 전에 다른 파지
대칭 또는 다음 score 후보를 검토합니다. 이 10°는 관절해 오차를 보장하는 값이 아닌
선정 여유입니다. 실제 파지점의 재검사와 실행 중 J7 한계는 `[-89,90]°`로 유지합니다.
`손목 계획`에 선정 당시 예상 J7 범위와 한계까지 최소 여유를 출력하며, 상승 계획 실패
시에도 이 예상 범위를 함께 남깁니다. 실제 로봇이 따라갈 연속 관절 경로를 뜻하지는 않습니다.
이는 직선 병진·최단 회전 경로의 표본 검사이며, 컨트롤러 블렌딩·관절해 연속성·특이점·충돌을
보장하는 경로 계획은 아닙니다. 상승 중 오류에는 실제 TCP·관절각·상승 목표와 error/warn을
분리해 출력합니다. C24에는 `get_c24_error_info()`의 컨트롤러 원본 servo ID와
관절 속도(°/s)를 정지 명령 전에 조회합니다. 진단 조회 실패가 원래 오류를 가리지는 않습니다.
검사 전용 C24 기각에는 이 API가 이전 실제 오류를 반환할 수 있으므로 사용하지 않으며,
대신 실제 시작 관절·목표 관절·검사 기준 속도·가속도를 남깁니다.
릴리즈에서는
정지·큐 비움·관절 목표 오차 0.5° 이하를 확인한 뒤 개방합니다. 명령 거부 시
다른 경로로 재시도하지 않고 큐를 정지합니다.
컨트롤러 검사는 자체 충돌·관절/Cartesian 제한·속도 초과 검사를 제공하지만,
외부 박스·파지 물체·블렌딩 경로 전체를 인증하는 검사는 아닙니다.
구조는 [MoveIt Task Constructor 논문](https://noah.nrw/ubbihs/download/pdf/5137431)의
단계별 조작 계획과 [Fallbacks](https://moveit.picknik.ai/main/doc/concepts/moveit_task_constructor/parallel_containers.html)
원리를 따릅니다. 여기서는 이미 쓰는 xArm 명령 생성기·검사기를 이용해 유한한 경로 집합을
탐색합니다. MoveIt/OMPL을 통합한 전역 장애물 회피 계획이나 완전성 보장은 아닙니다.
[사전 검사 API](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/xarm/wrapper/xarm_api.py)는
실제 혼합 블렌딩 전체의 동역학을 보증하지 않습니다.
[공식 SDK 이슈 답변(2023-03-22)](https://github.com/xArm-Developer/xArm-Python-SDK/issues/63#issuecomment-1478833694)은
IK의 비결정성 때문에 선형→조인트 블렌딩을 지원하지 않는다고 설명합니다.
오래된 혼합 블렌딩 예제만으로 현재 장비에서의 안정성을 판단하지 않습니다.
[UFACTORY C24 안내](https://help.ufactory.cc/en/articles/4712781-guide-user-to-solve-the-error-c24-speed-exceeds-limit)는
속도 감소를 권장하지만 특이 경로도 가능한 원인이므로, 감속만으로 해결된다고 가정하지 않습니다.

IK 검사를 사용하는 실제 실행에서는 파지 두 방향을 비교하며,
`upright/home`에서는 상승 두 방향까지 최대 네 조합을 비교합니다.
현재 실측 관절 → pre-grasp → grasp → pre-grasp → 상승 → 티칭된
릴리즈 관절 경로에서, 각 구간의 최대 관절 변화량을 합산하고 동률이면 전체 관절
이동량 합계를 비교합니다. 각도 차를 ±180°로 감아 케이블 회전을 축소 계산하지 않습니다.
카메라 하향 한도와 J7 범위를 통과한 조합만 비교하고, 비용이 같으면 카메라 상향을
우선합니다. 제약 때문에 두 대칭 중 하나만 가능하면 회전량이 줄지 않을 수 있습니다.
이는 종점 IK 기반 비용 추정이며 전체 경로 충돌 검사나 실제 최단시간 보장은 아닙니다.
드라이런·후보 IK 검사 비활성 시에는 관절 비용 최적화를 하지 않고 설정된 상승 기본 방향을 우선합니다.
실제 파지점에서의 상승 계획 검사는 후보 IK 설정과 관계없이 수행합니다.
사이클 상세에 선택한 상승 방향과 TCP 회전각을 표시하며, `--verbose`에서는
파지·상승 sign, 가능한 조합 수와 관절 이동 비용도 볼 수 있습니다.

파지점과 릴리즈 관절 목표의 연속 도착 확인 시간은 각각
`robot.approach_confirm_s`, `drop.arrival_confirm_s`(기본 설정 모두 0.05초)입니다.
세 확인 시간은 0.04~1초 범위이며 조건이 깨지면 연속 확인을 처음부터 다시 합니다.
설정이 없는 기존 파일은 0.2초를 유지합니다. 이 시간은 실제 이동이 끝난 뒤의
추가 확인 시간이며, 정지·빈 큐·목표 도착 조건을 생략하지 않습니다.
연속 모드 택타임에는 `복귀·상승·이송` 합산 구간과 릴리즈 도착 확인 시간을 표시합니다.
중간 정지 시점이 없으므로 상승·이송 시간을 명령 전송 시각으로 임의 분리하지 않습니다.
기본 속도는 이동 900, 접근 500, 복귀 750, 상승 최대 900 mm/s, 홈·이송 180°/s입니다.
`upright/home`에서 회전량/이동거리 기반 속도 산정 기준은 300°/s이며,
접근 가속도는 1100, 복귀 가속도는 600, 상승 가속도는 1200 mm/s²입니다.
홈·이송 조인트 가속도는 각각 450·350°/s²입니다. 이 값은 명령 상한이며,
컨트롤러 사전 검사 결과에 따라 앞서 설명한 감속이 적용됩니다.
홈 속도는 SDK 1.18.4의 조인트 명령 상한 180°/s를 이미 사용하므로 가속도만 올렸습니다.
이 프로파일은 설정 상한을 높인 것이며, 실제 도달 속도와 택타임은 경로·부하에 따라 달라집니다.

그리퍼 닫힘은 이동 시작을 확인한 뒤 최대 10초 동안 정착을 기다립니다.
천천히 닫히는 것 자체는 실패가 아니며, 물체 접촉으로 멈췄을 때의 유지 전류도 허용합니다.
이 대기 중에도 카메라·사용자 정지 요청과 로봇 정지 상태·오류를 검사합니다.
닫힘 명령 실패·방향 이상·정착 실패는 빈 파지 판정과 구분하여 `grip_check.enabled: false`여도
상승하지 않습니다. 로봇 오류는 발생한 단계에서 출력하고 자동 해제하거나 다시 움직이지 않습니다.
특히 [C31](https://docs.supportarticle.ufactory.cc/support_articles/software/how-to-solve-c31-error.html)은
관절 전류 기반 충돌 감지이며 그리퍼 대기 시간 초과가 아닙니다. 닫힘 중 또는 최초 상승 때
다른 물체에 걸렸다면 그 원인을 제거해야 하며, 대기 시간을 늘리는 것으로 해결되지 않습니다.
현재 실측 치수는 플랜지→팁 끝 중앙 203mm, 교체형 팁 전체 길이 80mm입니다.
`robot.tcp_offset_mm_deg: [0, 0, 203, 0, 0, 0]`를 연결 시 정지·빈 큐 상태에서 적용하고,
컨트롤러 판독값을 확인한 뒤 내부 좌표 변환을 갱신합니다. 적용·확인 실패 시 이동하지 않습니다.
이 키가 없는 기존 설정은 Studio의 TCP offset을 그대로 읽습니다.
프로그램은 `save_conf()`로 컨트롤러의 전체 설정을 저장하지 않고, 실행마다 이 TCP 값을 확인합니다.
Rerun도 같은 TCP 설정을 사용하며, `gripper.tip_length_mm: 80`은 그림 치수만 조절합니다.
팁은 URDF가 아닌 치수 기반 근사 메시입니다. 핑거와 패드를 합친 길이는 80mm이며,
장착 캐리지까지 포함하면 96mm로 보입니다. 패드 길이를 80mm에 덧붙이지 않습니다.
팁 길이 보정을 삽입량에서 분리할 때 `approach_extra_m`는 70→50mm로 조정했습니다.
접근 축과 플랜지 +Z축이 일치하는 현재 구성에서 `(extra − TCP Z)`가 같으므로,
기존 TCP 223mm/extra 70mm와 새 TCP 203mm/extra 50mm는 같은 파지점·pre-grasp의
플랜지 목표를 만듭니다. 절대 TCP 높이로 지정한 상승 경유점의 플랜지 위치까지 같다는 뜻은 아닙니다.
카메라와 플랜지의 상대 위치가 그대로라면 hand-eye 변환은 유지됩니다.
파지 자세에서 pre-grasp로 돌아오는 구간의 가속도는 자유 공간 상승보다 낮은 600mm/s²입니다.
가속도 조정은 이미 걸린 물체를 자동으로 빼내는 기능은 아닙니다.
현재 `selection.approach_extra_m`는 이후 조정한 47mm입니다.
`selection.use_grasp_depth: true`이므로 네트워크가 예측한 depth는 이 값에 더해집니다.
콘솔의 `검출점`은 검출된 파지 위치, `TCP 목표`는 이 삽입을 반영한 실제 명령 위치입니다.
`검출점에서 삽입`은 최종 목표까지의 거리로, 하강 제한을 사용하면 제한 후 거리입니다.

C24(속도 한계 초과) 대응으로 파지 후 복귀와 손목 자세 전환의 속도를 분리했습니다.
`robot.retreat_speed_mm_s`는 pre-grasp 복귀 상한,
`robot.reorient_speed_mm_s`는 pre-grasp 이후 상승 상한입니다. 두 값 모두
`robot.lift_speed_mm_s` 이하로 제한됩니다.
`upright/home`의 자세 전환은 추가로 `이동거리(mm) × reorient_angular_speed_deg_s / 회전각(deg)`로
속도를 제한합니다. 이는 TCP 경로 평균 회전 시간 기반의 보수적 산정이며 관절별
최대 속도나 특이점 회피를 보장하지 않습니다. 전환 가속도는
`robot.reorient_acc_mm_s2`, 조인트 이송 속도는 `drop.joint_speed_deg_s`입니다.
오류 발생 시 실제 TCP·관절각·대기 명령 수를 출력하고 큐를 정지하며 자동 재시도하지 않습니다.

정상 반복 운전에서는 최초 활성화 이후 모드·모터·오류·정지 상태·빈 큐를 확인하고,
매번 활성화 명령과 0.5초 대기를 반복하지 않습니다. 실행 중 정지/오류/모드 변경이
발견되면 자동 재활성화 없이 중단합니다. 조인트 가속도는 홈 복귀의
`robot.home_joint_acc_deg_s2`, 이송의 `drop.joint_acc_deg_s2`로 지정합니다.
기본 설정은 홈 330·이송 220°/s²입니다. 키를 생략하면 기존 50°/s²를
사용합니다. 짧은 이동에서는 최고 속도보다 가속도가
이동 시간을 제한할 수 있습니다.

매 사이클 홈 복귀 시작부터 릴리즈 조건 충족까지의 전체 시간과 구간별 시간을 출력합니다.
접근 시간은 병행하는 폭 조절을 포함하고, 추론·선정·시각화 시간은 결과 기록까지
포함합니다. 재추론은 같은 사이클에 누적되며, `--confirm`에서는 사용자 입력 대기도
포함됩니다. 이 값은 호스트에서 측정한 구간 경과 시간으로, 컨트롤러 내부 궤적 시간과
동일하지 않습니다.
접근 상세에는 최종 grasp 명령 전송 전·후의 시간을 나눠 표시합니다. 명령 전 시간에는
pre-grasp 이동과 병행하는 폭 조절이 포함됩니다. 명령 후 시간에는 남은 pre-grasp 이동,
최종 접근과 도착 확인이 포함될 수 있습니다. 폭 조절 시간은 별도 참고값으로 표시하며
전체 시간에 다시 더하지 않습니다.

기본 터미널 출력은 사이클별 검출·후보·선정·실행 결과와 **택타임**으로 묶습니다.
평균·최단·최장 택타임은 릴리즈까지 완료한 사이클만 집계하며, 파지 성공률을 뜻하지
않습니다. 후보 없음·중단·dry-run은 `소요시간`으로 표시하고 평균에서 제외합니다.
장면 보충 입력 대기는 해당 사이클 측정을 끝낸 뒤 시작합니다.
refiner 통계·폭 분포·자세 선택·모션 속도 등 상세 진단은 `python run.py --verbose`
또는 `--set output.verbose=true`로 표시합니다. 경고·오류는 기본 모드에서도 표시합니다.
모델 초기화의 긴 stdout은 `outputs/<실행이름>/model_init.log`에 저장합니다
(`output.save_cycles`와 무관). stderr 경고와 로딩 진행률은 그대로 표시합니다.

출력 예시 (시간은 설명용):

```text
  CYCLE 003  |  홈 → 파지 → 릴리즈
  검출      114 ms  |  추론 113 ms (refiner 64 ms 포함) · 후처리 1 ms
  후보      50 → score 50 → 폭 5 → ROI 5  |  폭 한계 95 mm
  선정      #1/5 · score 0.265133 · 폭 60 → 개방 60 mm · 기울기 18°
  결과      이송 완료
  택타임    6.42s  |  평균 6.51s · 최단 6.42s · 최장 6.60s · 완료 3회
  구간      홈 복귀 1.20s · 캡처·안정화 0.31s · 추론·선정·시각화 0.18s
            모션 준비 0.02s · 접근(폭 조절 병행) 1.30s · 파지 닫힘 0.40s
            복귀·상승 1.50s · 조인트 이송 1.20s · 개방 0.31s
```

pre-grasp 폭 조절 최소 대기는 `gripper.pre_open_min_wait_s`(기본 0.15초, 기존 1초)이며,
목표 폭 도착과 연속 3회 정지 확인을 통과해야 최종 접근 명령을 보냅니다.
그리퍼 닫힘 최소 대기는 `gripper.close_min_wait_s`(기본 설정 0.15초)입니다.
`gripper.speed_pct`는 이미 API 상한인 100%이며, 최소 대기를 줄이는 것은
모터 회전 속도 증가와 다릅니다. 손가락 이동·정착이 더 오래 걸리면 계속 기다립니다.
명령 후 이동 시작과 연속 3회 위치 안정 확인은 유지하며, 안정화 실패 시
파지 성공 판정 사용 여부와 관계없이 상승을 중단합니다.

place는 기본 `gripper.release_mode: width`입니다. 완전 개방 명령을 보내되,
`min(stroke, max(선정 개방 폭, 실측 파지 폭) + release_margin_mm)`까지 벌어졌으면
`release_confirm_s` 동안 폭을 확인하고 복귀합니다. 기본 여유는 5 mm, 확인 시간은
현재 기본 설정은 0.1초이며 나머지 개방은 펌웨어가 홈 복귀 중 계속 수행합니다.
일반 목표의 판독 허용 오차는 여유의 10%, 최대 0.5 mm입니다. 목표가 최대 스트로크에
걸리면 `release_full_open_tolerance_mm`(100mm 도구의 기본값 2.0mm)를 사용합니다.
허용 오차는 최대 2mm 또는 스트로크의 2% 중 작은 값이며, 100mm 도구에서
98.2~98.9mm까지 열린 실측 사례를 반영합니다. 이 경우 개방 방향 진행과 낮은 전류도
확인합니다. 처음부터 기존 엄격한 끝단 범위(기본 99.5 mm 이상)에 있으면 이미 열린
상태로 처리할 수 있습니다. 단순한 명령 ACK만으로 작은 폭을 끝단으로 간주하지 않습니다.
힘 유지에서 개방으로 전환할 때 작은 역방향 변화는 최대 폭 대비 0.5 mm를 넘는 상태가
0.2초 지속되는지 확인합니다. 2 mm 초과 역방향은 즉시 중단하며, 단일 표본의 작은
변화 뒤 정상 개방하는 경우에는 계속합니다. 개방 진행이 없으면 한 번만 재전송합니다.
이는 폭에 따른 릴리즈 가정이며 실제 물체 이탈을 감지한 결과는 아닙니다.
폭 미도달·통신 오류·fault·잼·카메라 오류 때는 복귀하지 않고 중단합니다.
`release_timeout_s`(기본 5초)는 최초 명령부터 재전송·폭 확인을 포함한 전체 실패 한도이며,
시간이 지났다고 성공 처리하지 않습니다. 시간 초과에는 통과 기준·측정 폭 범위·마지막
전류와 상태를 출력합니다.
기존 완전 개방·정착 대기는 `--set gripper.release_mode=full`로 사용할 수 있고,
이 모드와 연결 점검에서는 `gripper.open_min_wait_s`(현재 기본 설정 0.2초)가 적용됩니다.
`payload_check.enabled: false`이면 토크 측정용 `payload_check.settle_s` 대기는 적용되지 않습니다.

기본 설정의 FoundGrasp는 `third_party/FoundGrasp/grasp_detector.py`의 독립 `detect()`를
직접 호출합니다. 모델 내부 4 mm voxel NMS(최대 4096개), 원본 폭 보정,
refiner/GPU collision 분기, 최종 grasp NMS(5 cm / 30°), top-50을 그대로 사용합니다.
원본과 동일하게 forward에 별도 autocast를 추가하지 않습니다. Refiner는 기존
가중치·환경 레시피를 사용하며, 원본 함수의 max_w=0.1 m를 따릅니다.
측정면 검사가 켜진 refiner 사용 시 내부 점군 collision은 생략되고,
refiner를 끄면 원본 GPU hard collision 검사가 실행됩니다.
FoundGrasp에는 추가 공유 충돌 검사와 개방 폭 x1.2를 적용하지 않습니다.
로봇 실행 시 score<=0 후보와 실제 그리퍼 한계 95 mm 초과 후보를 제외하고,
ROI·기하·IK 검사를 유지합니다. 따라서 실행 가능한 후보 수는 50보다 적을 수 있습니다.
AnyGrasp는 공유 충돌 검사를 적용하고 SDK의 개방 폭을 그대로 사용합니다.
기존 EconomicGrasp/RNGNet/HGGD 워커는 raw 폭에 추가 마진을 적용합니다.
공유 충돌 검사는
`postprocess.collision.backend=auto`에서 CUDA가 있으면 GPU `detect_soft()`를 사용합니다.

후보가 없거나 ROI·IK·하강 제한 후 실행 자세가 없으면 같은 홈에서 새 프레임으로
`retry.max_inference_retries`회(기본 3회) 추가 추론합니다. 첫 시도 포함 총 4회
실패한 뒤 사용자 입력을 기다리며, 재추론 중에는 홈 이동과 파지 동작을 하지 않습니다.
refiner 로그는 최종 NMS 이전 후보의 측정면 기각·예산 제외·네트워크 평가 후
0점 수를 구분합니다. 점수와 원본 후보 처리 순서는 변경하지 않습니다.
