# Experiment Configs — Phase 1

Naming scheme: `configs/tmech/t{table}_{condition}.yaml`
Kinect variants use `_kinect` suffix on the condition.

---

## Ambiguous items — resolved

### 1. "FT no adapter" config 존재 여부 (Table IV Row 1)

**없음.** `foundgrasp_vitb_ft.yaml`은 FT+GeoFormer (Table IV Row 2)이며 Row 1에 해당하는 FT+no-adapter config는 `configs/tmech/`에 없었다. `t4_ft_noadapter.yaml`을 신규 생성.

### 2. t4 = t6 여부

**부분 동일, 전체 동일 아님.**

| 결론 | 근거 |
|---|---|
| `t4_frozen_geoformer` ≡ Table VI Ours row | 동일 실험 — config 재사용 가능 |
| Table VI Row 1 (visual-only) ≠ Table IV Row 3 (frozen no-adapter) | Row 1은 head depth도 비활성화 필요 (`rgbd_fusion_type: none`); Row 3은 head depth 유지 |
| Table VI Row 2–3, 5 | Table IV에 없는 조건 — 구현 필요 (Phase 2) |

→ Phase 1에서 t6 Ours row는 `t4_frozen_geoformer.yaml` 재사용. 나머지 t6 rows는 Phase 2.

---

## Phase 1 config 목록

### Table I — GraspNet-1Billion RealSense SOTA comparison

| Config | Table row | 비고 |
|---|---|---|
| `foundgrasp_vitb.yaml` (기존) | FoundGrasp DAv2-B | 78.21/72.74/33.88 완료 |
| `t1_dav2s.yaml` | FoundGrasp DAv2-S | 신규 |
| `t1_dav2l.yaml` | FoundGrasp DAv2-L | 신규 |

### Table II — GraspNet-1Billion Kinect SOTA comparison

| Config | Table row | 비고 |
|---|---|---|
| `t2_dav2s_kinect.yaml` | FoundGrasp DAv2-S Kinect | 신규 |
| `t2_dav2b_kinect.yaml` | FoundGrasp DAv2-B Kinect | 신규 |
| `t2_dav2l_kinect.yaml` | FoundGrasp DAv2-L Kinect | 신규 |

Training 및 evaluation 모두 `camera: kinect`. 모델은 Kinect 데이터로 별도 훈련.

### Table IV — Adapter ablation (DAv2-B, RealSense)

| Config | Row | Frozen | Adapter | 비고 |
|---|---|:---:|---|---|
| `t4_ft_noadapter.yaml` | 1 | ✗ | none | 신규 (Fatal — Row 3 isolation의 FT reference) |
| `t4_ft_geoformer.yaml` | 2 | ✗ | GeoFormer | = `foundgrasp_vitb_ft.yaml` 동일 |
| `t4_frozen_noadapter.yaml` | 3 | ✓ | none | 신규 (**Fatal 1** — freezing vs adapter gain 분리) |
| `t4_frozen_lora.yaml` | 4 | ✓ | LoRA | = `lora.yaml` 동일 |
| `t4_frozen_adaptformer.yaml` | 5 | ✓ | AdaptFormer | = `adaptformer.yaml` 동일 |
| `t4_frozen_vitadapter.yaml` | 6 | ✓ | ViT-Adapter | = `vit_adapter.yaml` 동일 |
| `t4_frozen_geovitadapter.yaml` | 7 | ✓ | Geo-ViT-Adapter | = `geo_vit_adapter.yaml` 동일 |
| *(skip)* | 8 | ✓ | GRIP-Adapter | 실험 제외 |
| `t4_frozen_geoformer.yaml` | 9 | ✓ | GeoFormer (Ours) | = `foundgrasp_vitb.yaml` 동일 |

### Table V — Cross-backbone validation (GeoFormer fixed, matched recipe)

| Config | Row | Backbone | Frozen | 비고 |
|---|---|---|:---:|---|
| `t4_ft_noadapter.yaml` (재사용) | 1 | DAv2-B | ✗ | Table IV Row 1과 동일 실험 |
| `t4_ft_geoformer.yaml` (재사용) | 2 | DAv2-B | ✗ | Table IV Row 2와 동일 실험 |
| `t4_frozen_geoformer.yaml` (재사용) | 3 | DAv2-B | ✓ | Table IV Row 9 (Ours) |
| `t5_ft_dinov2b.yaml` | 4 | DINOv2-B | ✗ | 신규 |
| `t5_frozen_dinov2b.yaml` | 5 | DINOv2-B | ✓ | 신규 |

---

## Phase 2 (별도 구현 필요 — 이번 Phase에서 제외)

| Table | 내용 | 이유 |
|---|---|---|
| Table VI Row 1 | visual-only (frozen ViT, no geometry) | `rgbd_fusion_type: none` 또는 depth encoder 비활성화 필요 |
| Table VI Row 2 | prediction-level only (no adapter, head depth) | head-only depth without adapter: 별도 config 분기 필요 |
| Table VI Row 3 | representation-level only (GeoFormer, no head depth) | `rgbd_fusion_type: none` at head 필요 |
| Table VI Row 5 | + decoder injection | DPT decoder에 depth path 추가 구현 필요 |
| Table VII | supervision design ablation (4 rows) | 별도 loss/supervision 변형 구현 필요 |
| Table VIII | robot experiment | 하드웨어 필요 |

---

## Base config 신규 추가

- `configs/tmech/dinov2b.yaml` — DINOv2-B base (backbone + bottleneck 파라미터)
  - `backbone.name: facebook/dinov2-base`
  - `peft.bottleneck: 96`, `peft.drop_path: 0.2` (DAv2-B와 동일)
