# exp_foundgrasp — 세션 인수인계 (2026-09-16)

새 세션에서 개발을 이어가기 위한 요약. 프로젝트 자체의 사용법·설계 근거는 `README.md`,
설정 전체는 `configs/default.yaml`(주석 포함)에 있으므로 여기서는 **현재 상태, 검증 범위,
반드시 알아야 할 하드웨어 사실, 미해결 과제**만 적는다.

## 1. 프로젝트

FoundGrasp 파지 검출을 실제 로봇(xArm7 + 손목 D435 + KORAS 2FPA 그리퍼, FESTO 플렉스 팁)에서
실행하는 파지-이적재 실험 플랫폼. 물체 박스 스캔 → 검출 → 선정 → 파지 → 릴리즈 박스로
이송을 반복한다. 목적은 (1) FoundGrasp 실환경 구현, (2) 이후 추가될 비교군(다른 파지 모델)을
**같은 하드웨어·절차**로 비교 실험하는 것. 검출 백엔드만 교체 가능하게 `efg/detectors/`로
분리되어 있다 (계약: `efg/detectors/base.py`, 추가 절차: README "비교군 추가").

실행: `conda activate foundgrasp && python run.py [--auto | --dry-run | --set key=value]`

## 2. 현재 상태 — 무엇이 검증되었고 무엇이 아닌가

이 프로젝트는 2026-09-16에 기존 코드(grasp_pick + exp_autograsp + FoundGrasp/koras 사본 +
hand-eye 로봇 모듈)를 **재구성**한 것이다. 로직은 실장비 검증 완료본(2026-08-07)을 그대로
이식했고, 재구성 후 확인한 범위는 다음과 같다.

| 검증됨 | 미검증 |
|---|---|
| 단위 테스트 38/38 (`python -m pytest tests/ -q`) | **실장비 파지 사이클 전체** (스캔→파지→이송→릴리즈) |
| 새 위치 FoundGrasp 추론 (샘플 프레임, 후보 50개) | 플렉스 팁으로의 실제 파지 성공률 |
| 카메라 `--dry-run` 3사이클 (검출·ROI·선정) | 페이로드 판정 실측 (기준값 미측정) |
| 로봇·그리퍼 연결 (읽기 전용) | |

→ **첫 작업은 `python run.py`(프롬프트 유지)로 전체 사이클을 실장비에서 확인하는 것.**

## 3. 하드웨어 사실 — 반드시 지킬 것

- **그리퍼**: 전원 투입 후 본체 버튼(I)으로 초기화. 소프트웨어 Modbus 초기화(101)는 grip_close
  방향 학습을 손상시킴(실측 재현 100%, 버튼 초기화로만 복구). `initialize_on_connect: false` 유지.
  연결 시 코드가 개방·close 방향을 실측 점검해 이상이면 중단한다.
- **그리퍼 포트**: `/dev/ttyACM*` 번호는 재연결마다 바뀜(ACM0→ACM1 실측) → 설정은 by-id 고정 경로
  (`/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A101355-if00`). 보드 교체 시 S/N 갱신.
- **로봇**: TCP offset(현재 215mm)·TCP payload는 사용자가 플렉스 팁 기준으로 재설정 완료.
  이송 가속도는 50 deg/s² 고정(페이로드 토크 스파이크로 충돌 감지 C31 오작동 방지).
- **OS**: 커널 7.0.0-28로 부팅 금지 — D435 USB 스택 실패 → xHCI 사망 → 입력 전체 프리즈 실측.
  GRUB가 6.17.0-35로 고정되어 있고 커널·NVIDIA 자동 업데이트가 차단되어 있음.
- **카메라 K**: hand-eye 캘리브레이션 때의 refined K(fx 943.6)를 `configs/calib/intrinsics.yaml`로
  사용. 스트림 K(factory, fx 907)를 쓰면 역투영 좌표가 hand-eye와 어긋난다.
- **캘리브레이션 재실행**: 카메라 마운트가 바뀌면 별도 도구(`r_foundgrasp/02.dev/hand_eye_calib`,
  conda `env_foundgrasp`)로 재캘리브레이션 후 결과를 `configs/calib/`에 교체.

## 4. 현재 설정 상태 (`configs/default.yaml`)

판정·안전 기능은 **모두 꺼져 있다** (사용자 지시).

| 항목 | 값 | 의미 |
|---|---|---|
| `grip_check.enabled` | false | 파지 성공 판정 미적용 — finger % 출력만, 항상 이송. 실패 재시도 없음 |
| `payload_check.enabled` | false | 기준값 파일도 아직 없음 (`scripts/calib_payload.py`로 생성) |
| `floor_limit.enabled` | false | 높이맵(`configs/calib/box_floor.npz`)은 있음 — 박스 배치 바뀌면 재스캔 |
| `selection.approach_max_tilt_deg` | 25 | 29°에서 self-collision 정지 실측 |
| `selection.approach_extra_m` | 0.04 | 삽입 여유. `use_grasp_depth: true`로 예측 depth 가산 |
| `gripper.pre_open_margin_mm` | 3 | |
| `gripper.stroke_mm` / `max_grasp_width_m` | 100 / 0.095 | **플렉스 팁 두께 미반영 — 실측 필요** |
| `scan_roi_px`, `robot.home_joints_deg`, `drop.joints_deg` | 08-06 실측값 | 박스·자세 바뀌면 `scripts/` 사전 준비 재실행 |

## 5. 알아둘 기술적 사실 (근거가 있는 것들)

- **ROI 처리 = depth 마스킹 + 투영 필터**. 전체 프레임 추론은 top-k가 ROI 밖(릴리즈 박스 더미)에
  몰려 "후보 45개 → ROI 내 0개"가 실측됨. ROI 밖 depth를 0으로 두면 인공 표면 없이 해결.
- **camera_up 선정**: 평행 그리퍼 closing_sign ±는 등가 → 두 해 중 손목 카메라가 위로 가는 쪽 선택,
  하향 20° 초과는 탈락. 카메라가 바닥·박스에 닿는 문제 해결.
- **모션**: 그리퍼 pre-open은 pre-grasp 이동과 스레드 병행 — `robot.enable_motion()`에 sleep(0.5)가
  있어 반드시 스레드 시작 전에 호출. 접근은 100 mm/s·500 mm/s². 이송은 상승점 위 `via_height_m`를
  MoveArcJoint 블렌딩(`radius=`)으로 정지 없이 지나 릴리즈 직행(홈 경유 안 함).
- **플렉스 팁 파지 판정**: finger 간격은 팁이 물체 형상대로 휘어 무의미. 대안으로 구현한 것이
  **릴리즈 자세 관절 토크 페이로드 판정**(`efg/payload.py`). 실측: xArm7 `get_joints_torque()`
  정지 재현성 std 0.000 Nm(분해능 0.001), J1/J7은 항상 0(축 수직), 단발 스파이크(J4 +5.7 Nm) 관측
  → 중앙값 사용. F/T 센서 없음. 판정량은 샘플 물체 차이 벡터 방향으로의 부호 있는 투영.
  기준값은 자세 고정이라 릴리즈 자세 재티칭 시 재측정(자세 1° 이상 다르면 자동 생략+경고).
- **그리퍼 전류 기반 조기 판정**(미구현): Modbus 폴링 100 Hz 실측으로 close 중 전류-위치 프로파일
  기록은 가능. "전류 상승 시작 위치"로 접촉을 볼 수 있으나 이송 중 낙하는 못 잡음 → 페이로드
  판정의 보조로만 의미. 착수하려면 먼저 빈 손/물체 close 곡선 수집 스크립트부터.
- **카메라 필터**: `camera.postprocess: true` = disparity→spatial→temporal→depth, 전부 pyrealsense2
  기본값. temporal Holes Fill=3이라 정지 직후 캡처에 잔상 가능 → 스캔 전 0.3s 대기 있음.
- **FoundGrasp `gamma_max_abs_deg`**(`configs/_base.yaml`)는 코드가 읽지 않는 죽은 설정. tilt 제한은
  `selection.approach_max_tilt_deg`가 유일.
- **import 주의**: FoundGrasp 루트가 `utils/`, `model/`을 최상위 이름으로 노출 → 이 프로젝트 루트에
  같은 이름 디렉터리 금지 (테스트로 고정). 그리퍼 드라이버는 `third_party/` 사본이 sys.path 우선.

## 6. 알려진 이슈

- 그립 삽입 깊이가 더 들어가지 않는 경우가 있음 — 원인 미조사 (floor_limit 축소 로그, 예측
  depth·`approach_extra_m` 대조 필요)
- 얇은 물체(포크·납작 브래킷)는 FoundGrasp 후보가 잘 안 생김 → "ROI 내 후보 없음" 대기 발생
- tilt 제한은 self-collision의 근사. 재발하면 방향별 차등 제한 검토

## 7. 다음 할 일 (우선순위)

1. **실장비 전체 사이클 검증** (`python run.py`, 비상정지 준비)
2. **플렉스 팁 파지 판정 확정**: `scripts/calib_payload.py` → `payload_check.enabled: true` 실측.
   임계값이 애매하면 `grip_check` 끈 채로 값을 모아 조정. `stroke_mm`·`max_grasp_width_m` 실측 갱신
3. pre-grasp 거리 단축 검토(`pre_grasp_offset_m` 0.10), 삽입 깊이 이슈 조사
4. 비교군 백엔드 추가 (후보: AnyGrasp·GSNet·HGGD) — GraspGroup 변환 규약 정리 필요
5. 비교 실험용 사이클 결과 로그(CSV: 방법·사이클·score·성공·소요시간) — 아직 없음, 논문 통계에 필요

## 8. 이전 기록 위치

- 실운용 개선 7건의 파라미터 근거: `r_foundgrasp/01.docs/grasp_pick_튜닝노트.md`
- 그리퍼 펌웨어 진단 이력·프리즈 원인 분석 등: `r_foundgrasp/04.archive/handoff-log_dev.md`,
  `r_foundgrasp/00.연구노트/연구노트_개발.md`
