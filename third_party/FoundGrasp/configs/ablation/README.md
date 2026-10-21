# Ablation configs (최종 레시피 = ../ours_base.yaml 상속)

모든 파일은 `ours_base.yaml`(RC-GT + propagation/conf-weighting ON + 150ep)을
상속하고 자기 ablation 축만 오버라이드한다.

## 런 없이 채워지는 표 행 (앵커/기존 런 재사용 — 별도 config 없음)

| 표 행 | 재사용하는 런 |
|---|---|
| Table IV — Frozen + GeoFormer (Ours) | ours_base |
| Table V — DAv2-B | ours_base |
| Table VI — Ours (repr+pred) | ours_base |
| Table VI — Prediction-only | ablation/adapter_frozen_noadapter (동일 실험) |
| Table VII — Full supervision | ours_base |
| Geometry source — xyz (canonical) | ours_base (_base의 input_type='xyz') |

## FT 세팅

FT 계열(adapter_ft_*)은 **LLRD 0.65를 기본 내장** (head/adapter = base LR,
backbone layer-wise decay 0.65). llrd 0.75/0.90 스윕 변형은 구 레시피 기록으로
`configs/final/`에만 남아 있다.
