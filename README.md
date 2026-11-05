# FoundGrasp 로봇 실험

FoundGrasp · AnyGrasp · RNGNet · GFLA를 같은 로봇 실행 파이프라인에서 비교합니다.
현재 장비의 환경·체크포인트·캘리브레이션이 준비되어 있다는 기준입니다.

```bash
conda activate foundgrasp
cd ~/exp_foundgrasp

# foundgrasp | anygrasp | rngnet | gfla 중 하나만 변경
grasp_detector=foundgrasp

# 실제 로봇 실행: 홈 복귀 후 자동 파지·이송 반복
python run.py --config configs/benchmark.yaml --grasp-detector "$grasp_detector"
```

카메라 검출만 확인하려면 같은 명령에 `--dry-run`을 추가합니다.
로봇은 움직이지 않으므로 카메라가 이미 스캔 자세에 있어야 합니다.

```bash
python run.py --config configs/benchmark.yaml --grasp-detector "$grasp_detector" --dry-run
```

- 설정: [configs/default.yaml](configs/default.yaml). 일시 변경은 `--set 항목=값`.
- 비교 실험은 네 방법 모두 **`configs/benchmark.yaml`** 사용. `python run.py`만 실행하면 FoundGrasp 단독 설정입니다.
- 비교군의 별도 Conda 환경은 자동으로 실행됩니다. Rerun도 별도 창으로 열립니다.
- 기록: `outputs/`의 실행 폴더에 설정·입력 프레임·후보·시간, 별도 `.rrd`에 시각화 저장.
- 모델 고유 필터·폭·추가 입력의 차이: [비교 조건](docs/benchmark_protocol.md).
- 설치·저장 프레임 추론·상세 설정: [상세 가이드](docs/guide.md).
