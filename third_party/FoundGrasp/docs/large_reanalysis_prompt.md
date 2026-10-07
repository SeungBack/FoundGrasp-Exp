# 과제: Frozen DAv2 grasp 모델에서 DAv2-Large(ViT-L)가 DAv2-Base(ViT-B)를 못 넘는 원인 규명 및 해결

## 0. 이 문서의 성격
핸드오프용 **팩트 모음**이다. 아래 수치·코드사실은 logs·checkpoints·소스에서 추출했으나,
그대로 신뢰하지 말고 직접 재검증한 뒤 진행하라(검증법 3절). 해석·가설·다음에 할 일은 적지 않았다.
가설 수립과 실험 설계는 이 세션이 직접 하라.
직전 세션이 4·5절의 run들을 실행했고 그중 L을 B까지 올린 것은 없다. 직전 세션의 해석은 이 문서에
포함하지 않았으니 출발점으로 삼지 말 것.

## 1. 목적
Frozen DAv2 백본 + GeoFormer 어댑터 grasp 모델에서, 동일 method 하에
DAv2-Large(1024d/24L)가 DAv2-Base(768d/12L)와 같거나 더 나은 GraspNet-1B RealSense grasp AP를 내게 한다.
현재 L은 검증된 모든 설정에서 B보다 ~1 AP 낮다.

## 2. 하드 제약
1. **단일 method**: S/B/L에 동일 아키텍처/규칙. Large 전용 모듈/loss/분기 추가 금지.
   허용되는 변경 형태는 (a) S/B/L 공통 규칙의 재정의(같은 공식을 셋 다에 적용, 예: 폭 = f(dim),
   tap = 상대깊이 함수), 또는 (b) per-size 표준 학습 HP(lr/warmup 등). 어떤 변경이든 S/B/L 모두
   재실험 가능해야 한다.
2. **백본 frozen 유지**: LoRA/부분 unfreeze 금지.
   (관련 데이터: `foundgrasp_vitb_ft`는 Base 백본을 파인튜닝한 run으로 train_loss −15.31,
   test all 54.44 — 4절 표.)
3. **데이터 동일**: GraspNet-1B RealSense. 데이터/카메라 추가 금지.

## 3. ⚠️ run 이름 ≠ 실제 백본 (먼저 확인)
`logs/tmech/foundgrasp_vitl*` 중 **다수가 실제로는 Base 백본으로 학습됨**(이름만 vitl).
어떤 분석이든 **체크포인트에서 백본 차원과 adapter 블록 수를 직접 확인**한 뒤 진행하라.

검증 한 줄 (qkv=1024→Large, 768→Base, 384→Small; peft 블록 수 = 백본 층수):
```python
import torch, re
ck = torch.load(PATH, map_location='cpu', weights_only=False, mmap=True)
sd = ck.get('model_state_dict') or ck['state_dict']
dim = sd['backbone.encoder.layer.0.attention.attention.query.weight'].shape[0]
nlayers = len({int(m.group(1)) for k in sd if (m:=re.search(r'encoder\.layer\.(\d+)\.',k))})
cfg = ck['config']  # 실제 bottleneck/drop_path/lr/wd/epochs (이름과 다를 수 있음)
print(dim, nlayers, ck['train_loss'], cfg['model']['peft'], cfg['training']['learning_rate'])
```
오염 사례: `foundgrasp_vitl_0.3_256`는 이름과 달리 bn192, `foundgrasp_vitl_0.3_96_wd`는 실제
bn192/dp0.2/wd0.15. 이름의 bottleneck/dp/epoch를 믿지 말 것.
gc6d 비교 주의: 과거에 가짜-Large(실제 Base) run을 Large로 착각해 비교한 사례가 있었음.

## 4. 검증된 결과 (실제 백본 기준, GraspNet-1B RealSense, final test + CD)
모든 행은 체크포인트 config로 백본/HP 재확인됨. all=all_ap, tl=train_loss(최종),
gc6d=GraspClutter6D zero-shot AP(일부 run만 JSON에 저장됨; `foundgrasp_vitb`의 gc6d는 JSON에 없음).

```
sz bn   dp   run                          tl      all    seen   sim    nov    gc6d
S  48   0.1  t1_dav2s                      -11.32  60.30  77.92  69.69  33.28  33.16
B  96   0.2  foundgrasp_vitb (HEADLINE)    -11.71  61.61  78.21  72.74  33.88  (JSON없음)
B  64   0.3  foundgrasp_vitl_64            -11.39  61.39  78.32  72.10  33.75  35.40
B  96   0.3  foundgrasp_vitl_96            -11.51  61.04  78.33  71.77  33.01  33.76
B  128  0.3  foundgrasp_vitl               -11.58  61.29  78.31  71.49  34.08  37.67
B  192  0.3  foundgrasp_vitl_100 (100ep)   -11.38  61.06  77.61  71.96  33.62  36.21
B  192  0.3  foundgrasp_vitl_192 (75ep)    -11.36  61.45  78.72  71.72  33.91  35.41
B  192  0.3  foundgrasp_vitl_200 (200ep)   -12.01  61.37  78.37  72.02  33.73  31.43
B  192  0.3  foundgrasp_vitl_250 (250ep)   -12.15  61.14  78.27  71.70  33.46  31.19
B  192  0.3  foundgrasp_vitl_256           -12.19  60.94  78.58  71.45  32.80  36.26
B  192  0.3  foundgrasp_vitl_75 (75ep)     -11.12  60.90  77.02  71.15  34.52  35.75
B  96   0.2  foundgrasp_vitb_ft (FT)       -15.31  54.44  73.46  61.75  28.11  -
L  192  0.2  foundgrasp_vitl_0.2           -12.53  60.99  78.05  71.47  33.47  37.24
L  192  0.3  foundgrasp_vitl_0.3           -12.27  59.96  77.95  70.23  31.71  37.54
L  192  0.3  foundgrasp_vitl_0.3_256       -12.66  60.36  78.16  70.00  32.92  34.53
L  192  0.3  foundgrasp_vitl_0.3_100epoch  -11.95  59.71  77.91  69.31  31.90  35.32
L  192  0.4  foundgrasp_vitl_0.4           -12.01  60.00  78.31  69.56  32.14  37.49
L  192  0.5  foundgrasp_vitl_0.5           -11.79  60.26  77.47  70.44  32.86  38.19
L  192  0.2  foundgrasp_vitl_0.3_96_wd(wd.15)-12.55 59.95  78.07  70.24  31.54  33.76
L  192  0.2  foundgrasp_vitl_k12 (주입12점) -12.15  60.30  78.08  70.86  31.96  35.62
L  192  0.2  foundgrasp_vitl_lr1e4 (lr1e-4) -12.36  60.26  77.90  70.48  32.40  36.89
L  128  0.2  foundgrasp_vitl_128_dp0.2     -12.39  59.59  77.61  69.50  31.67  37.75
L  96   0.2  foundgrasp_vitl_96_dp0.2      -12.22  59.96  77.89  69.46  32.53  36.88
L  96   0.3  foundgrasp_vitl_0.3_96        -11.97  60.40  77.43  70.63  33.14  37.89
```

### 산수 (해석 아님)
- all_ap: S≈60.3 / B 60.9–61.61(평균 ~61.2) / L 59.59–60.99(평균 ~60.1). S < B > L.
- B−L 격차의 split 분포: seen ~0(동률), similar/novel에서 큼(~1–2).
- 어댑터/옵티마이저 설정 변동(bn 64→256, dp 0.2→0.5, lr, wd, 주입점, epoch 75→250) 폭:
  B는 all 60.9–61.45, L은 all 59.59–60.99. 두 백본 모두 이 범위 안에서만 움직임.
- train_loss: L은 대체로 B보다 낮음(−12.x vs −11.x).

## 5. 이미 실행된 단일변수 실험 (기준 = `foundgrasp_vitl_0.2`, all 60.99 / tl −12.53; 단일 변수만 변경)
| 변경 | all | tl |
|---|---|---|
| 주입점 24→12 (k12, layers=[1,3..23]) | 60.30 | −12.15 |
| lr 2e-4→1e-4 (lr1e4) | 60.26 | −12.36 |
| bottleneck 192→128 (dp0.2) | 59.59 | −12.39 |
| bottleneck 192→96 (dp0.2) | 59.96 | −12.22 |
| dp 0.2→0.3 | 59.96 | −12.27 |
| dp 0.2→0.4 | 60.00 | −12.01 |
| dp 0.2→0.5 | 60.26 | −11.79 |
| wd 0.05→0.15 | 59.95 | −12.55 |
| epoch 100 | 59.71 | −11.95 |

## 6. 검증된 아키텍처 사실 (코드 기반, model/)
- `found_grasp.py` `FoundGrasp`: 백본 frozen→patch tokens. 백본 입력은 RGB(pixel_values)만.
  depth(xyz)는 별도 `depth_encoder`(resnet34)로 인코딩.
- **GeoFormer** = `peft.py:RGBDAdapter`, 병렬-to-MLP. config `layers:"all"` → 모든 블록의 FFN
  residual에 depth-조건 bottleneck delta를 주입. 주입점 수 = 백본 층수 (S/B=12, L=24).
  층당 `scale`은 상수(config `scale`, 기본 1.0). delta는 `drop_path` 후 더해짐.
- **DPT taps**: `utils.py:get_intermediate_layer_config` — 백본 이름으로 하드코딩.
  S/B=[2,5,8,11], L=[5,11,17,23]. (참고: 공식 DAv2-L 디코더 tap은 [4,11,17,23].)
- **다운스트림 폭은 전 사이즈 동일**: DPT readout out_channels=[48,96,192,384],
  decoder/depth-encoder/head `feature_dim=64`, head_dim=128 — S/B/L 전부 같음(= 공식 DAv2-Small
  디코더 스펙). 백본만 768/1024로 다르고 1024d는 곧장 [48..384]로 압축됨.
  (참고: 공식 DAv2 디코더는 사이즈별로 커짐 — B [96,192,384,768]/128, L [256,512,1024,1024]/256.)
- effective batch: B=1GPU×64, L=2GPU×32(DDP grad평균), 둘 다 global_step 60000. lr 둘 다 2e-4.
  (참고: `_base.yaml` 주석은 DDP에서 lr=2e-4*N(linear scaling)을 적으나, 실제 L run은 2e-4로 학습됨.)
- tap feature에 backbone.norm 적용.
- config override: `decoder.feature_dim`, `decoder.out_channels`는 config로 덮을 수 있음
  (found_grasp.py:318-321). `out_indices`(tap)는 utils.py 하드코딩 → 코드 수정 필요.

## 7. 코드/데이터/자산 위치
- 모델: `model/found_grasp.py`, `model/peft.py`(GeoFormer=RGBDAdapter), `model/dpt.py`,
  `model/blocks.py`(head/depth stem), `model/utils.py`(tap 인덱스), `utils/optimizer.py`.
- config: `configs/_base.yaml`, `configs/tmech/`(t1_dav2{s,l}.yaml, dav2b.yaml, foundgrasp_vit*.yaml).
- 로그: `logs/tmech/<run>/checkpoints/checkpoint_epoch_*.pth`(config+train_loss 포함),
  `final_test_test_cd_results.json`(GraspNet), `final_test_gc6d_cd_results.json`(GC6D, 일부만).
- 미실행 config(존재만): `configs/tmech/foundgrasp_vitl_dec128.yaml`,
  `configs/tmech/foundgrasp_vitb_dec128.yaml` (feature_dim 64→128, out_channels [96,192,384,768]).
- 진단 스크립트(존재): `scripts/probe_bl_features.py`.
- 데이터: `/data/Grasp/GraspNet-1Billion`(RealSense), `/data/Grasp/GraspClutter6D`.
- HF 캐시에 DAv2 S/B/L 전부 존재. GPU 4×96GB 가용.

## 8. 작업 방식 (워크플로 사실)
- **사용자가 학습/데이터 스크립트를 직접 실행한다.** config/스크립트만 준비하고 실행은 사용자에게
  맡겨라. 읽기전용 진단 forward는 허용.
- 4절 표를 직접 재검증한 뒤 출발하라(3절 검증법).
