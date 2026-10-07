RGBDAnchorGateResidualAdapter — 수식 + 설명
표기
$x \in \mathbb{R}^{C}$: ViT patch token (frozen)
$d \in \mathbb{R}^{D}$: depth feature 정렬 후
$\mathrm{LN}$: LayerNorm, $\odot$: element-wise product, $\mathrm{sg}(\cdot)$: stop-gradient
$B$: adapter bottleneck dim
수식
각 transformer 레이어에서 adapter는 token 단위 residual $\Delta(x,d)$를 생성하고, AdaptFormer placement로 MLP에 병렬 합산된다.

$$
\Delta(x,d) ;=; W_{up},\mathrm{GELU}!\big(,v + z + \alpha \odot r(z),\big)
$$

각 항:

$$
\begin{aligned}
v &= W_v,\mathrm{LN}(x) && \text{(visual bottleneck)} \
z &= W_d,d && \text{(depth bottleneck)} \
u_{\text{sep}} &= W_u,\mathrm{LN}!\big(\mathrm{sg}(x)\big) && \text{(detached RGB anchor; $W_u$는 $W_v$와 별개 파라미터)} \
\alpha &= 1 + \tanh!\big(\mathrm{MLP}g\big([z,,u{\text{sep}}]\big)\big) && \text{(anchor-conditioned gate)} \
r(z) &= \mathrm{MLP}_r(z) && \text{(depth-only residual content)}
\end{aligned}
$$

$\mathrm{MLP}_g$의 마지막 layer, $\mathrm{MLP}r$의 마지막 layer, 그리고 $W{up}$은 zero-init이다. 따라서 초기에 $\alpha = 1$, $r(z)=0$, $\Delta \equiv 0$이며, frozen backbone을 perturb하지 않은 상태에서 학습이 시작된다.

설명 (한 문단)
활성화 내부의 세 항은 분리된 역할을 갖는다. $v + z$는 RGBD-AdaptFormer의 additive RGB-D bottleneck과 동일한 baseline 경로를 보존한다. $\alpha \odot r(z)$는 depth로부터 산출된 residual content $r(z)$를 anchor가 channel-wise로 게이팅하는 보정항이다. anchor $u_{\text{sep}}$는 $W_v$와 별도의 projection $W_u$를 가지며 입력에 stop-gradient를 적용하므로, RGB는 gate를 조건짓는 read-only context로만 동작하고 depth-side gradient는 anchor projection을 통해 RGB 경로로 흐르지 않는다.

Paper용 한 문단 압축형
We add a parallel residual $\Delta(x,d) = W_{up},\mathrm{GELU}(v + z + \alpha \odot r(z))$ at each transformer MLP, where $v = W_v\mathrm{LN}(x)$ and $z = W_d d$ are the visual and depth bottlenecks, $r(z)=\mathrm{MLP}r(z)$ is a depth-only residual content, and $\alpha = 1 + \tanh(\mathrm{MLP}g([z, u{\text{sep}}]))$ is an anchor-conditioned gate driven by a detached RGB projection $u{\text{sep}} = W_u\mathrm{LN}(\mathrm{sg}(x))$ that is parameter-distinct from $W_v$. The gate, residual, and output projections are zero-initialized, so $\Delta = 0$ at the start of training. The stop-gradient confines RGB to a read-only conditioning role for the depth residual: it shapes the gate without receiving gradients from $z$ or $r(z)$.

피해야 할 표현 (검토해서 안 쓴 것)
"lightweight", "efficient", "powerful" — 정량 근거 없는 형용사
"geometry-aware backbone" — 백본은 그대로 frozen, 우리가 만드는 건 routing
"anchor learns RGB representation" — sg 때문에 그렇지 않음
"two-stream fusion" — fusion이 아니라 conditioned correction
여기서 멈추는 게 가장 안전한 academic 수준이다.