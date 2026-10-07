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
| 로봇 | UFACTORY xArm7 (SDK 1.18.4, 이더넷). Studio에 **TCP offset = 그리퍼 끝**, **TCP payload = 그리퍼 질량** 설정 |
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
pip install pyrealsense2 xarm-python-sdk pymodbus pyserial
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
python run.py                        # 6) 실행 (접근 직전 [y/N] 확인)
python run.py --auto                 #    무인 연속 실행
```

실행 중: 라이브 창에서 `q` = 정지 요청(홈 복귀 후 종료). ROI에 후보가 없거나 연속 실패
시 터미널에서 대기(Enter 재개 / q 종료). 이동 실패로 중단되면 로봇은 그 자리에 정지한다
(자동 복귀가 2차 충돌을 만들 수 있음 — xArm Studio로 확인 후 복귀).

## 실행 옵션

플래그는 셋뿐이다. 나머지 동작은 전부 설정 파일에서 정한다.

| | |
|---|---|
| `--auto` | 접근 확인 프롬프트 생략 (무인) |
| `--dry-run` | 스캔·검출·선정만, 로봇/그리퍼 미사용 |
| `--config PATH` | 다른 설정 파일 사용 |
| `--set KEY=VALUE` | 설정 항목 일시 변경 (반복 가능) |

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
| `calib` | `hand_eye`(T_flange_cam), `intrinsics`(캘리브레이션 K — hand-eye와 동일해야 함) |
| `robot` | `ip`, `home_joints_deg`(스캔 자세), 이동·접근 속도/가속도 |
| `camera` | 해상도·fps, `postprocess`(RealSense depth 필터) |
| `gripper` | `port`(by-id 고정 경로), 힘/속도, `pre_open_margin_mm`, `grip_success_min_pct` |
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
- **로봇**: 첫 실행은 `--auto` 없이 확인 프롬프트를 유지하고 비상정지에 손을 올린 채 진행.
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
| C31 충돌 오감지 정지 | Studio의 TCP payload 확인. 이송 가속도는 50 deg/s² 고정 |
| 백본 다운로드 시도 | 오프라인이면 `HF_HUB_OFFLINE=1` |

## 설계 근거 (요약)

- **ROI = depth 마스킹 + 투영 필터**: 전체 프레임 추론은 top-k가 ROI 밖 물체 더미에 몰려 ROI 내
  0개가 되는 문제가 실측됨. ROI 밖 depth를 0으로 두면 인공 표면 없이 후보가 ROI 안에서만 생긴다.
- **선정**: score band(`score_tie_tolerance`) 안에서 가장 수직인 접근 우선. closing_sign ± 두 해 중
  카메라 상향·J7 회전 최소 쪽 선택. tilt 상한 25°(29°에서 self-collision 실측).
- **모션**: 그리퍼 개방은 pre-grasp 이동과 병행, 접근은 저속·저가속, 이송은 pre-grasp 상방
  경유점을 블렌딩(정지 없음)으로 지나 릴리즈 직행. 제어된 종료만 홈 복귀.
- **재시도**: 실패 시 홈 복귀 후 전체 재스캔(물체가 밀렸을 수 있음), 직전 실패 지점 반경 제외.
