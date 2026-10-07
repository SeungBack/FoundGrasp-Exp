"""exp_foundgrasp 핵심 패키지.

구성 (확장 지점은 detectors/ 하나):
  paths / config          프로젝트 경로·설정 로드
  se3 / robot / motion    xArm7 제어 (플랜지↔TCP 변환, IK, 직선/조인트 이동)
  camera                  RealSense D435 캡처 + 라이브 스트림 스레드
  gripper                 KORAS 그리퍼 제어·파지 시퀀스
  detectors/              파지 검출 백엔드 (FoundGrasp; 비교군은 여기에 추가)
  transforms / selection  카메라→base 변환, 실행 가능 후보 선정
  roi / floor / payload   스캔 영역, 하강 제한, 페이로드 파지 판정
  viz                     그리퍼 와이어프레임 2D 시각화
  demo                    파지-이적재 루프 (run.py 진입점)
"""
