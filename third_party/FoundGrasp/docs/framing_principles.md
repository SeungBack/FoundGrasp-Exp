# FoundGrasp Paper Framing Principles

Current Best Ours model: configs\tmech\foundgrasp_vitb.yaml

Core principles for writing, framing, and reviewing contributions.
Derived from: gisbi-kim.github.io (what-paper-to-write, motivation-is-not-novelty, research-agenda-design-talk) and The InComplete Guide to Academic Writing in Robotics Research.

---

## Meta-Principles (apply before writing anything)

### 1. Motivation ≠ Novelty

"Existing methods fail at X → we added module Y" = motivation only.

Genuine novelty requires:
1. **Principled diagnosis**: WHY existing methods fail, not just THAT they fail
2. **Derived solution**: the proposed method's design naturally follows from the root cause
3. **Rigorous validation**: ablations proving each design choice addresses the identified problem

Test: can you answer "why must the solution take this exact form, and not a simpler/different one?"

### 2. One-Sentence Test

Can you explain the contribution in a single sentence **without listing components**?

- Pass: "Transformer enables sequence modeling without recurrence" (states a principle)
- Fail: "combining MobileNetV3 + BiFPN + scaling" (lists components)
- Fail: "we propose a pipeline with depth encoder, adapter, DPT decoder, and dense head"

### 3. Absence Test

What does the community lose without this work? Vague answers = insufficient novelty.

### 4. Generation Test

Can future researchers ask "why?" based on your findings? Papers that generate follow-up questions have structural depth.

---

## Pre-Writing Question Gate

Use this before drafting the abstract, introduction, method, or experiment section. Every checked item must have a concrete answer in one or two sentences. If the answer is still a component name, a vague benefit, or "because it improves AP", the framing is not ready.

### 0. Stop Conditions

Do not start writing if any of these are unanswered:

- [ ] **Absence**: If this paper did not exist, what exactly would the community lose?
- [ ] **One sentence**: Can the paper be stated without listing modules?
- [ ] **Root cause**: Why do existing approaches fail, beyond "they perform worse"?
- [ ] **Derived solution**: Why does the method's form follow from that root cause?
- [ ] **Evidence contract**: Which table, figure, ablation, or failure analysis supports each main claim?
- [ ] **Reviewer home**: Which community is this paper speaking to, and what debate is it entering?

### 1. Paper Identity

- [ ] Is this primarily a **problem-defining paper**, a **solution-advancing system paper**, a **benchmark/metric paper**, or an **empirical finding paper**?
- [ ] If it is a system paper, what principle or empirical finding survives after the implementation details become obsolete?
- [ ] If it is a benchmark/metric paper, what previously invisible failure does it make measurable?
- [ ] If it is a problem-defining paper, what new name, axis, or abstraction will future papers reuse?
- [ ] What is the paper's durable object: a concept, metric, diagnosis, protocol, empirical law, or reusable system?

### 2. Blind Spot and Research Agenda

- [ ] Across 20-30 related papers, what condition, failure mode, assumption, or metric is repeatedly ignored?
- [ ] Is this a weakness of one paper, or a structural blind spot of the field?
- [ ] Can the blind spot be named in 2-4 words?
- [ ] Can it be measured, benchmarked, or turned into a taxonomy?
- [ ] Why is the blind spot becoming important now: new model scale, new data, new deployment pressure, new hardware, or new evaluation capability?
- [ ] If the proposed method is not the best, does the paper still leave behind a useful map, metric, diagnosis, or negative result?

### 3. Motivation to Novelty

- [ ] What is the precise failure mode of prior work?
- [ ] What hidden assumption makes prior work fail under this setting?
- [ ] Why is the proposed design the simplest sufficient response to that failure?
- [ ] Why would a simpler baseline fail?
- [ ] Why would a more complex variant be unnecessary or harmful?
- [ ] Which ablation proves each load-bearing design choice?
- [ ] What result would falsify the central explanation?

### 4. Claim-Evidence Contract

Before writing, fill this mapping. A claim without evidence should be removed or weakened. Evidence without a claim should be moved to appendix or reframed.

| Claim type | Question to answer first | Evidence required |
|---|---|---|
| Problem claim | What real limitation is not captured by current metrics? | Failure cases, split-wise gaps, qualitative examples, or metric critique |
| Diagnosis claim | Why does the limitation happen? | Controlled ablation, representation analysis, error taxonomy, or counterexample |
| Method claim | Why must the method take this form? | Minimal baseline, stronger variant, and design-choice ablation |
| Generalization claim | Where should the gain appear if the story is true? | Novel/similar split, OOD condition, cross-backbone check, or deployment-like stress test |
| Practical claim | What can future researchers do differently because of this paper? | Recipe, protocol, limitation boundary, or actionable negative result |

### 5. Writing Discipline Checklist

- [ ] The abstract has a problem, diagnosis, method, evidence, and implication. It is not a module inventory.
- [ ] The introduction starts from a tension or failure the field must care about, not from "recent advances in X".
- [ ] Related work is organized by unresolved assumptions or failure modes, not by a chronological list of papers.
- [ ] Contributions distinguish **findings** from **implementation infrastructure**.
- [ ] Each section answers the questions raised by the introduction in the same order.
- [ ] Every figure/table caption states the takeaway, not just the contents.
- [ ] Every loaded term has a stable name and is used consistently.
- [ ] No vague adjective is used unless a metric or protocol makes it concrete: robust, general, efficient, scalable, safe, trustworthy.
- [ ] The method includes enough detail for reproduction, but only claims novelty for load-bearing choices.
- [ ] Limitations and failure cases are owned explicitly; they are boundaries of the claim, not embarrassments.

### 6. FoundGrasp-Specific Fill-In

Fill these before writing the FoundGrasp abstract or introduction:

- [x] Current axis **A**: What has 6-DoF grasping optimized so far?  
  Answer: Learning-based 6-DoF grasp detection has advanced through increasingly capable task-specific networks trained on dedicated grasp datasets; however, available grasp benchmarks cover a limited diversity of objects and scenes, constraining the transferability of learned representations.

- [x] Real bottleneck **B**: What matters for novel-object deployment but is not handled by that axis?  
  Answer: Performance on unseen object categories is bounded by the representational diversity of the training distribution — a structural constraint shared across existing approaches regardless of architectural design.

- [x] New research axis **C**: What does FoundGrasp make visible, measurable, or actionable?  
  Answer: FoundGrasp achieves higher grasp detection performance across seen, similar, and novel object splits by adapting VFM visual priors with geometry-aware adaptation. Cross-dataset validation (Grasp Clutter6D) will further support this claim when available.

- [x] Named blind spot:  
  Answer: No formal name needed for a T-Mech system paper. State directly: existing methods do not leverage pretrained visual representations, instead learning task-specific representations from limited grasp supervision from scratch.

- [x] One-sentence claim:  
  Answer: Task-specific visual representations trained on grasp data are bounded by the diversity of that data, regardless of architectural improvements; a frozen vision foundation model, adapted with minimal geometry injection, breaks this ceiling.

- [x] Absence answer:  
  Answer: Without this work, it would remain unrecognized that visual representations bounded by grasp-specific training data are a key limiting factor for novel-object generalization, regardless of architectural design. FoundGrasp demonstrates this with frozen VFM adaptation: 33.88% vs. 28.75% Novel AP (DSF-Grasp).

---

## FoundGrasp-Specific Framing

### The Two Root-Cause Diagnoses

**F1 — visual representation is bounded by grasp dataset diversity**
Task-specific representations are bounded by the grasp training distribution — a structural constraint shared across all existing methods regardless of architecture. GraspNet-1Billion training set: 100 scenes, 30 objects, 25.6K images. DSF-Grasp illustrates this ceiling: dual-stream RGB+geometry fusion raises Novel AP from 24.31% (GSNet) to 28.75% — a 4.44-point gain that still leaves generalization bounded by the 30-object training distribution.

**F2 — Adapting VFMs to 6-DoF grasping is non-trivial**
VFMs operate on RGB input and have no interface for the metric 3D quantities grasping requires — approach direction, gripper orientation, and grasp depth in camera coordinates. Fine-tuning the backbone is counterproductive: full fine-tuning degrades novel-object AP by 6.99 points relative to freezing, suggesting that visual generalization priors are sensitive to task-specific gradient updates on limited grasp data.

These two diagnoses naturally derive the solution:

> Keep the VFM frozen to preserve visual priors. Inject depth-derived geometry at both the representation and prediction levels. Use dense image-space 6-DoF attribute maps as the supervision interface for VFM tokens (approach direction, gripper direction, grasp width, insertion depth, graspness, and score), while explicitly handling orientation ambiguity.

FoundGrasp is a backbone-agnostic framework: the geometry injection design is validated across DA-V2, DINOv2, SAM, and SigLIP, confirming that the gain comes from the adaptation principle rather than a specific backbone.

### Pixel Grasp Field (PGF) — Method Design, Not Standalone Contribution

PGF is the output formulation that makes frozen VFM adaptation compatible with 6-DoF grasping. It is part of the method, not the primary contribution.

The primary contribution is the finding: task-specific backbones trained on grasp data cannot exceed the diversity of that data, regardless of architectural design. Frozen VFM + geometry injection breaks this limit. PGF is the formulation required to make VFM token-level supervision work — VFM tokens correspond to image patches, so pixel-level prediction keeps supervision aligned with the token grid without 2D-to-3D feature conversion.

**Contribution hierarchy:**

| Layer | Content |
|---|---|
| **Finding** | The generalization bottleneck in 6-DoF grasping is representational, not architectural. Frozen VFM breaks the task-specific training ceiling. Evidence: Novel AP 33.88% vs. 28.75% (DSF-Grasp); frozen vs. fine-tuned +6.99 AP. |
| **Method** | Freeze VFM + bi-level geometry injection (RGBD-AdaptFormer at feature level, geometric features at prediction level) + pixel-level grasp supervision (PGF). |
| **PGF specifically** | Dense per-pixel continuous regression of approach direction, gripper direction, width, and depth — the formulation that keeps 6-DoF grasp supervision aligned with the frozen VFM token grid. Described in method section, not claimed as representational novelty. |

**Method description (for method section, not contribution bullet):**
> We formulate 6-DoF grasp detection as dense per-pixel regression of approach direction, gripper direction, width, and depth. Each valid pixel is lifted to a full grasp pose via aligned XYZ, with orientation assembled from the predicted continuous vectors. This formulation keeps grasp supervision aligned with the frozen VFM token grid, avoiding 2D-to-3D feature conversion that would disrupt pretrained visual representations.

Do not claim "first pixel-wise 6-DoF grasp method" — GG-CNN predates for planar, RGB Matters predates for per-pixel orientation. Do not lead contribution bullets with "We introduce PGF."

### The Paper-Level Claim (One Sentence)

**Finding-first (abstract/intro):**
> Task-specific visual representations trained on grasp data are bounded by the diversity of that data, regardless of architectural improvements; a frozen vision foundation model, adapted with minimal geometry injection, breaks this ceiling — achieving the largest novel-object AP gain over existing 6-DoF methods.

**Internal framing compass:**
> The generalization bottleneck in 6-DoF grasping is representational, not architectural. Frozen VFM + geometry injection exposes and breaks this limit.

---

### Contribution Structure (Final — for Introduction)

**Paper type**: System paper (T-Mech convention). Finding is the claim; system is the evidence.

**Rule**: C1 is the finding (what we showed). C2 is the system (what we built to show it). C3 is the validation (how we know). Do not reverse this order. Do not lead with the system.

**Our contributions can be summarized as follows:**

**C1 (Finding):**
> We demonstrate that the generalization bottleneck in 6-DoF grasping is representational, not architectural: task-specific networks trained on grasp data cannot exceed the diversity of that data regardless of design improvements. Within GraspNet-1B, all methods plateau at 28–29% Novel AP despite years of architectural progress; FoundGrasp achieves 33.88% (+5.13 AP over DSF-Grasp, +17.8% relative). Independently, on GraspClutter6D direct transfer, all task-specific methods collapse to 10–23 AP while FoundGrasp achieves 40.46 AP (+17.77 AP, +78.3% relative over the best prior).

**C2 (System):**
> We propose FoundGrasp, a geometry-aware adaptation framework that freezes a VFM backbone and injects metric geometry at two levels — inside frozen ViT blocks via GeoFormer, and at the prediction stage via geometric feature fusion — with dense pixel-level 6-DoF grasp supervision aligned to the VFM token grid.

**C3 (Validation):**
> Ablation confirms that freezing outperforms full fine-tuning by 6.99 Novel AP, that GeoFormer (feature-level geometry injection) contributes +2.48 AP over AdaptFormer (head-only depth), and that the adaptation principle holds across four VFM architectures (DA-V2, DINOv2, SAM, SigLIP).

---

### T-Mech Acceptability Conditions

Assuming real robot experiments and full ablation are completed, the paper is acceptable for T-Mech if:

| Condition | Requirement | Issue ref |
|---|---|---|
| Real robot experiment | Required for mechatronics relevance; results on previously unseen objects | — |
| Frozen vs. Full FT fairness | Full FT must use matched training recipe (same LR, epochs, batch); report "under matched recipe" | Issue 9 |
| Benchmark table normalization | Split RealSense / Kinect; flag collision detection; use consistent AP metric | Issue 15 |
| External pretraining disclosure | State VFM uses large-scale pretraining; no additional grasp labels; category overlap cannot be excluded | Issue 11 |
| Backbone-agnostic validation | Show finding holds across ≥2 VFMs to rule out backbone-specific artifact | C3 |
| Candidate budget fairness | Report raw candidates before NMS and final after; add max_grasps sensitivity | Issue 17 |

**Core risk**: DSF-Grasp RGB backbone is task-specific (2D U-Net trained from scratch on GraspNet). This must be explicitly stated to make the representational diversity argument hold. If a reviewer claims "DSF-Grasp also uses pretrained RGB backbone," the finding weakens.

---

### Finalized Terminology

| Concept | Term | Notes |
|---|---|---|
| FoundGrasp 전체 접근 | **geometry-aware bi-level adaptation** | umbrella term |
| output formulation | **Pixel Grasp Field (PGF)** | dense pixel-level continuous regression of 6-DoF attributes, lifted to SE(3) via aligned XYZ |
| depth encoder output | **geometric feature(s)** | "depth feature/depth" 대체 |
| adapter 이름 | **GeoFormer** | the adapter module; replaces "RGBD-AdaptFormer" |
| adapter가 하는 것 | **feature-level geometry injection** | additive injection inside each frozen ViT FFN block |
| head가 하는 것 | **prediction-level geometry injection** | geometric features fused at the grasp prediction stage |
| backbone | **frozen VFM backbone** | 그대로 유지 |


---


## Key Numbers to Anchor Claims (To be Added)

| Comparison | AP gap | Interpretation |
|---|---|---|
| FoundGrasp vs Scale-Balanced-Grasp (GraspClutter6D direct transfer) | +17.77 AP abs (+78.3% rel) | Cross-dataset; strongest C1 evidence — all task-specific 10–23 AP |
| FoundGrasp vs DSF-Grasp (GraspNet-1B Novel) | +5.13 | VFM prior beats dual-stream RGB+3D on unseen objects |
| FoundGrasp vs MTGrasp (GraspNet-1B Similar) | +9.35 | Large generalization gap |
| GeoFormer (frozen) vs Full FT + GeoFormer | +5.77 Novel AP (+7.17 mean AP) | Freezing effect cleanly isolated — adapter held constant |
| GeoFormer vs AdaptFormer | +2.48 mean AP | Geometry injection at representation level helps |

---

## Reviewer Attack Responses

**"The adapter is trivial — it's just AdaptFormer with depth added."**
Yes. That's the point. The contribution is not the adapter's complexity but showing that this minimal change, within a frozen backbone, produces the largest gain of all tested variants.

**"This is just PEFT for robotics."**
No. The paper's question is not whether PEFT works. It is: why does freezing outperform fine-tuning for grasping, and where should depth enter the frozen VFM? The answer has practical consequences for how future VFM adaptation in robotics should be designed.

**"Why does this beat point-cloud methods if they have explicit 3D geometry?"**
Because their backbones are trained on 30 objects and overfit to training distributions. The VFM's visual priors, preserved by freezing, cover far more object categories. Novel AP is the evidence.
