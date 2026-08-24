# SmolVLM Eagle3 Method Notes

This file summarizes four slide-ready method blocks: band mixing, branch
distillation, visual row compression, and attention matching. Notation is kept
close to the implementation.

## Common Setup

The target model is frozen SmolVLM. For each training sequence
`x = (x_1, ..., x_S)`, the target produces logits `z^T_t` and selected hidden
states from target layers.

Let:

- `H` be hidden size, `V_d` be draft vocab size.
- $h^T_{t,l} \in \mathbb{R}^H$ be the target hidden state at token position `t` and
  target layer `l`.
- $e_t \in \mathbb{R}^H$ be the draft token embedding.
- $z^D_t \in \mathbb{R}^{V_d}$ be the draft logits.
- $P_T(\cdot \mid x_{\le t}) = \operatorname{softmax}(z^T_t / \tau)$ be the teacher distribution,
  sliced to the draft vocab during training.

The draft is trained by the usual Eagle-style autoregressive distillation loss:

```math
L_{CE}
= - \sum_{t \in M}
    \sum_{v \in V_d}
    P_T(v \mid x_{\le t}) \log P_D(v \mid x_{\le t}),
```

where `M` is the assistant/loss mask.

Markdown form: `L_CE = - sum_{t in M} sum_{v in V_d} P_T(v | x_<=t) log P_D(v | x_<=t)`.

## 1. Band Mixing

### Text Figure

```text
Frozen SmolVLM target layers

 early band                 middle band              late band
 [L2 L4 L8 L10]             [L15 L18 L20]            [L26 L28]
      |                          |                       |
      | softmax mix              | softmax mix           | softmax mix
      v                          v                       v
   m_early                    m_mid                   m_late
      |                          |                       |
      +-------------- concat: [m_early | m_mid | m_late] +
                                  |
                             per-stream RMSNorm
                                  |
                              FC: 3H -> H
                                  |
                     Eagle layer 0: [token emb | fused HS]
                                  |
                             draft logits
```

### Motivation

Stock Eagle3 uses a small set of target hidden layers, usually one early, one
middle, and one late layer. For SmolVLM, a single target layer can be noisy:
different visual/text behaviors appear at nearby depths. Band mixing replaces a
single selected layer with a learned convex mixture over a band of layers.

The clean one-layer baseline uses 9 target auxiliary layers grouped into 3
bands:

```text
B_1 = [2, 4, 8, 10]
B_2 = [15, 18, 20]
B_3 = [26, 28]
```

### Formula

For band `b`, with target layer set `B_b`, learn logits
$\alpha_b \in \mathbb{R}^{|B_b|}$ and convert them to simplex weights:

```math
w_{b,i}
= \frac{\exp(\alpha_{b,i})}
       {\sum_{j \in B_b} \exp(\alpha_{b,j})}.
```

The mixed stream at position `t` is:

```math
m_{t,b}
= \sum_{i \in B_b} w_{b,i} h^T_{t,i}.
```

Markdown form: `m_{t,b} = sum_{i in B_b} w_{b,i} h^T_{t,i}`.

For `banded_mix_fc`, the three mixed streams are concatenated and sent through
the stock Eagle3.1 fusion path:

```math
u_t = [m_{t,1}; m_{t,2}; m_{t,3}] \in \mathbb{R}^{3H},
```

with optional per-stream RMSNorm:

```math
\tilde{u}_t
= [\operatorname{RMSNorm}(m_{t,1}); \operatorname{RMSNorm}(m_{t,2}); \operatorname{RMSNorm}(m_{t,3})],
```

then a learned fusion projection:

```math
s_t = W_f \tilde{u}_t \in \mathbb{R}^H.
```

Layer 0 of the draft receives both the token embedding and fused target stream:

```math
y_t^{(0)} = \operatorname{DraftLayer}_0([\operatorname{RMSNorm}(e_t); \operatorname{RMSNorm}(s_t)]).
```

### Why It Helps

Band mixing lets the drafter learn "which depth inside this semantic band is
most predictive" while preserving Eagle3's inference shape: 3 streams become
`3H -> H`, then normal one-layer EAGLE decoding. It improves the mean temp-0
acceptance length from `2.508` for the stock one-layer baseline to `2.706`.

## 2. Branch Distillation

### Text Figure

```text
At a training position t:

teacher top-1:  b_t
draft top-1:    a_t

branch condition:
  a_t != b_t  and  a_t in teacher top-3

                         real path
        x_1 ... x_t ----------------------> Teacher logits z_T(real)
              |                           Draft logits   z_D(real)
              |
              | substitute draft plausible token a_t
              v
                         branch path
        x_1 ... a_t ----------------------> Teacher logits z_T(branch)
                                          Draft logits   z_D(branch)

match the movement, not the full distribution:

  center[z_D(branch) - z_D(real)]
          ~=
  center[z_T(branch) - z_T(real)]
```

### Motivation

Standard distillation teaches the draft to match the teacher on the real token
history. Speculative decoding often fails when the draft's top-1 token is not
the teacher's top-1, even if it is still a plausible teacher top-k token. Branch
distillation trains the draft to know what should happen after taking that
plausible alternative token.

### Branch Selection

At position `t`, let:

```math
a_t = \arg\max_v z^D_t(v),
\qquad
b_t = \arg\max_v z^T_t(v),
```

and let $\operatorname{TopK}_T(t)$ be the teacher's top-k token set. A natural branch is active
when:

```math
c_t =
\mathbf{1}\{
  a_t \ne b_t
  \land
  a_t \in \operatorname{TopK}_T(t)
  \land
  t \in M
\}.
```

The adopted config uses:

```text
branch_distill_top_k = 1
branch_distill_target_top_k = 3
branch_distill_steps = 1
branch_distill_loss_weight = 0.1
```

### Branch-Change Objective

Create a branched sequence `x^{br}` by replacing the real token at branch
positions with the draft's own top-1 token `a_t`. Then run one extra teacher
forward on `x^{br}`.

The key target is not the full branch distribution. It is the teacher's logit
change caused by the branch:

```math
\Delta z^T_t
= \operatorname{center}(z^T_t(x^{br})) - \operatorname{center}(z^T_t(x)),
```

where:

```math
\operatorname{center}(z) = z - \operatorname{mean}_v(z_v).
```

The draft runs the same one-step fork and predicts:

```math
\Delta z^D_t
= \operatorname{center}(z^D_t(x^{br})) - \operatorname{center}(z^D_t(x)).
```

The branch loss is masked MSE:

```math
L_{branch}
=
\frac{
  \sum_t c_t
  \| \Delta z^D_t - \Delta z^T_t \|_2^2
}{
  \sum_t c_t + \epsilon
}.
```

The total loss is:

```math
L = L_{CE} + \lambda_{branch} L_{branch},
\qquad
\lambda_{branch}=0.1.
```

Markdown form: `L = L_CE + lambda_branch L_branch`.

### Why Centered Delta

The base CE already teaches the real-path distribution. The branch term only
teaches the local sensitivity:

```text
If previous token changes from real token to draft-plausible token,
how should the next-token logits move?
```

This avoids relearning the full teacher distribution twice. In the measured
one-layer sweep, branch-change improves mean temp-0 acceptance length from
`2.706` to `2.815`, and mean speedup from `1.673x` to `1.738x`.

## 3. Visual Row Compression

### Text Figure

```text
One image tile before compression

  64 visual rows, each row carries all target aux streams

  row 01: [h_L2 | h_L4 | ... | h_L28]
  row 02: [h_L2 | h_L4 | ... | h_L28]
    ...
  row 64: [h_L2 | h_L4 | ... | h_L28]

                 learned cross-attention routing
                 shared across all 9 aux streams
                              |
                              v

  k summary rows per tile

  k = 1:
    summary row placed at fixed slot row 32

  k = 4:
    summary rows placed at fixed slots 8, 24, 40, 56

                              |
                              v

  compressed prompt for draft:
    text rows unchanged
    grid marker rows unchanged
    image rows: 64*T  ->  k*T
```

### Motivation

SmolVLM image prompts can contain roughly 900 image rows. A one-layer draft has
limited attention capacity and must route over all those rows. Visual row
compression compresses each image tile's 64 rows to `k` learned summaries in
target auxiliary hidden-state space, before `fc_norm`, band mix, and fusion FC.

The goal is not to compress pixels or embeddings. The goal is to compress the
exact target hidden states consumed by the drafter.

### Shapes

For each image tile:

```text
H_tile: [B, T, 64, n, H]
```

where:

- `T` is number of image tiles.
- `64` is rows per tile.
- `n=9` is number of aux streams for banded mix.
- `H=576` for SmolVLM-256M.

The compressor outputs:

```text
C_tile: [B, T, k, n, H]
```

For `k=1`, a 13-17 tile prompt becomes only 13-17 image rows for the draft.

### Shared Routing Formula

First build a reference stream for routing:

```math
\rho_s
= \frac{\exp(\beta_s)}
       {\sum_{r=1}^n \exp(\beta_r)},
\qquad
r_{b,t,j}
= \sum_{s=1}^n \rho_s H_{b,t,j,s}.
```

Here `j` indexes the 64 rows inside a tile.

For learned queries, with query `q_p` and tile embedding `g_t`, project both
query and reference rows into routing key space:

```math
\hat{q}_{t,p} = W_k(q_p + g_t),
\qquad
\hat{r}_{b,t,j} = W_k r_{b,t,j}.
```

The routing weights are:

```math
A_{b,t,p,j}
= \operatorname{softmax}_j
  \left(
    \frac{\hat{q}_{t,p}^{\top}\hat{r}_{b,t,j}}
         {\sqrt{d_k}}
  \right).
```

The same routing weights are applied to every aux stream:

```math
C_{b,t,p,s}
= \sum_{j=1}^{64} A_{b,t,p,j} H_{b,t,j,s}.
```

No value projection is used. Each compressed row is a convex combination of real
target hidden states. Markdown form: `C = sum_j A_j H_j`.


```math
C_{b,t,p,s} \in \operatorname{conv}\{H_{b,t,1,s}, \ldots, H_{b,t,64,s}\}.
```

### Slot Convention

The `k` summaries occupy fixed rows inside each tile, so RoPE and vLLM slot
mapping remain data-independent:

```text
k=1 -> row 32
k=4 -> rows 8, 24, 40, 56
```

Text rows and grid marker rows are untouched.

### Training Objective

The compressor is trained jointly with the draft by the same Eagle CE:

```math
L = L_{CE}
```

or, when combined with branch distillation:

```math
L = L_{CE} + \lambda_{branch} L_{branch}.
```

The compressor has its own parameter group, planned at LR `1e-3`, while the
drafter remains at LR `1e-4`.

## 4. Attention Matching

### Text Figure

```text
Optional routing supervision

          frozen target attention
       text/loss query -> image rows

       row1 row2 row3 ... row64
        |    |    |        |
        v    v    v        v
      A_T = target visual attention distribution

                    compare
                      |
                      v

      A_C = compressor/draft routing distribution
        ^    ^    ^        ^
        |    |    |        |
       row1 row2 row3 ... row64

Loss:
  KL(A_T || A_C)  or  MSE(A_T, A_C)

Intuition:
  keep or emphasize the rows the teacher actually reads
```

### Status

I did not find an implemented `attention_matching` loss in the current clean
branches. The following is the mathematically clean version that fits the same
method family: it uses the frozen target's attention over visual rows as a
teacher signal for the draft or compressor routing.

### Target Signal

For a target layer `l`, head `h`, query position `t`, and key/image row `j`,
the target attention distribution is:

```math
A^T_{l,h,t,j}
= \operatorname{softmax}_j
  \left(
    \frac{
      (q^T_{l,h,t})^\top k^T_{l,h,j}
    }{\sqrt{d_h}}
  \right).
```

For a visual compression method, collapse heads/layers to a tile-level teacher
distribution:

```math
\bar{A}^T_{t,j}
=
\sum_l \gamma_l
\frac{1}{N_h}
\sum_h A^T_{l,h,t,j},
```

where `gamma_l` can be uniform or chosen to match the aux bands.

### Draft/Compressor Distribution

If matching the compressor routing directly:

```math
A^C_{t,p,j}
= \operatorname{softmax}_j
  \left(
    \frac{\hat{q}_{t,p}^{\top}\hat{r}_{t,j}}{\sqrt{d_k}}
  \right).
```

For `k > 1`, aggregate the `k` routing rows:

```math
\bar{A}^C_{t,j}
= \sum_{p=1}^{k} \eta_p A^C_{t,p,j},
\qquad
\eta_p = \frac{1}{k}
```

unless a learned summary weighting is desired.

### Loss Options

KL attention matching:

```math
L_{attn}^{KL}
=
\sum_{t \in M}
\sum_j
\bar{A}^T_{t,j}
\log
\frac{\bar{A}^T_{t,j} + \epsilon}
     {\bar{A}^C_{t,j} + \epsilon}.
```

MSE attention matching:

```math
L_{attn}^{MSE}
=
\sum_{t \in M}
\| \bar{A}^C_t - \bar{A}^T_t \|_2^2.
```

Total loss:

```math
L
= L_{CE}
 + \lambda_{branch} L_{branch}
 + \lambda_{attn} L_{attn}.
```

### Interpretation

Band mixing chooses useful depth streams. Visual compression chooses useful
image rows. Branch distillation teaches local future sensitivity after plausible
wrong tokens. Attention matching would add an explicit routing prior: the rows
kept or emphasized by the draft should align with the rows the frozen target
actually attends to.

## Slide-Level Story

1. Baseline Eagle3: frozen target provides hidden states and logits; one-layer
   draft learns to imitate teacher next-token distributions.
2. Band mixing: replace hard aux-layer choice with learned convex mixtures over
   early/mid/late depth bands.
3. Branch distillation: when the draft chooses a plausible non-teacher-top1
   token, train the draft on the teacher's logit delta after that branch.
4. Visual row compression: compress 64 image rows per tile to `k` target-HS
   summaries before the draft sees them.
5. Attention matching: optional auxiliary loss to make compression/routing
   follow target attention over image rows.

## Compact Formula Summary

```math
m_{t,b} = \sum_{i \in B_b} \operatorname{softmax}(\alpha_b)_i h^T_{t,i}
```

```math
s_t = W_f [\operatorname{RMSNorm}(m_{t,1}); \operatorname{RMSNorm}(m_{t,2}); \operatorname{RMSNorm}(m_{t,3})]
```

```math
L_{branch}
=
\frac{\sum_t c_t
  \|\operatorname{center}(z^D_t(x^{br}))-\operatorname{center}(z^D_t(x))
   -\operatorname{center}(z^T_t(x^{br}))+\operatorname{center}(z^T_t(x))\|_2^2}
{\sum_t c_t + \epsilon}
```

```math
C_{b,t,p,s} =
\sum_{j=1}^{64}
\operatorname{softmax}_j\left(
  \frac{(W_k(q_p + g_t))^\top W_k r_{b,t,j}}{\sqrt{d_k}}
\right)
H_{b,t,j,s}
```

```math
L = L_{CE} + \lambda_{branch} L_{branch} + \lambda_{attn} L_{attn}
```
