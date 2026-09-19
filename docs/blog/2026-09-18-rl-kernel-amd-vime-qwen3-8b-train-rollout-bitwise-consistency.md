# RL-Kernel × vime × AMD: Achieving Bitwise Consistency Between Training and Rollout

By the RL-Kernel Team

On-policy RL assumes that the rollout engine and the training engine evaluate the same policy before a parameter update. In practice, however, generation and training are usually handled by separate engines. Identical models, weights, and inputs do not guarantee identical execution paths: kernels, batch shapes, parallel layouts, reduction order, and intermediate precision can all affect token probabilities, ultimately leading to train-rollout mismatch.

[vime](https://github.com/vllm-project/vime) and [RL-Kernel](https://github.com/RL-Align/RL-Kernel/) address two sides of this problem. vime manages the lifecycle of tokens, state, and weight versions; RL-Kernel aligns reduction and rounding boundaries across RMSNorm, Attention, GEMM, SwiGLU, linear logp, and distributed collectives. vime keeps both engines on the same training timeline; RL-Kernel ensures that they follow the same numerical execution contract.

## Introduction

Training and rollout place different demands on their execution engines. Rollout prioritizes throughput for sampling, prefill, decode, and KV cache operations. Training must support forward and backward passes, optimizer state, and multidimensional parallelism. The two engines run the same model, but they may use different underlying operators.

When these differences affect the importance ratio and clipped objective before a parameter update, they look like a genuine policy shift. With `--use-rollout-logprobs` enabled, vime reuses the rollout logprobs. Verifying bitwise consistency, however, requires holding the comparison targets, numerical contract, and actual execution paths fixed.

We start with floating-point non-associativity and examine Attention, logp, RMSNorm, GEMM, and collectives as nested reductions. We then explain how vime and RL-Kernel divide the work and how controlled operator ablations can pinpoint divergences. Finally, we present 200-step results for ROCm and CUDA.

### Questions This Article Addresses

* Why can training and rollout produce different logprobs even with identical models, weights, and inputs?
* How can a comparability gate and a numerical execution contract help locate the first divergence?
* Which aspects of temporal synchronization and numerical alignment do vime and RL-Kernel each handle?
* How do we confirm that zero mismatch actually comes from the intended backend?
* What do the 200-step ROCm and CUDA results demonstrate?

## Eliminating Train-Rollout Mismatch

### Why the Same Model Does Not Mean the Same Computation

If the model, weights, and inputs are identical, why can the results differ? And why must seemingly separate components such as RMSNorm, Attention, GEMM, logp, and communication be addressed together?

The underlying principle is:

> A model's formulas define the mathematical result, but not a unique execution path. Finite precision breaks some of the equivalences between those paths. Bitwise consistency therefore means ensuring that training and inference follow the same numerical contract.

## Spurious Policy Shifts Before Parameter Updates

The rollout engine generates token $a_t$ given prefix $h_t$ and records

$$
\ell_t^{\mathrm{R}} = \log \mu(a_t \mid h_t)
$$

Before training begins, the training engine rescores the token using the same weight version:

$$
\ell_t^{\mathrm{T}} = \log q(a_t \mid h_t)
$$

If the policy has not yet been updated and both engines are indeed computing the same logical object, the importance ratio should satisfy:

$$
\rho_t = \frac{q_t}{\mu_t}
= \exp\!\left(\ell_t^{\mathrm{T}} - \ell_t^{\mathrm{R}}\right) = 1
$$

Let $\delta_t = \ell_t^{\mathrm{T}} - \ell_t^{\mathrm{R}}$. When $\delta_t$ is small, $\rho_t \approx 1 + \delta_t$. This difference also enters the clipped objective in PPO and GRPO:

$$
J_t = \min\!\left(
\rho_t \widehat{A}_t,
\operatorname{clip}\!\left(\rho_t, 1 - \epsilon_{\mathrm{low}}, 1 + \epsilon_{\mathrm{high}}\right)
\widehat{A}_t
\right)
$$

Here, $\widehat{A}_t$ is the advantage estimate. If $\delta_t$ exceeds $\log(1 + \epsilon_{\mathrm{high}})$ or falls below $\log(1 - \epsilon_{\mathrm{low}})$, the mismatch can even change which clipping branch is taken. It creates an apparent policy shift before any parameter update.

We can further decompose this total error. Let $s_t^{\mathrm{P}}$ and $s_t^{\mathrm{D}}$ denote the probabilities assigned to the same token by serving prefill and an independent decode replay, respectively. Then

$$
\frac{q_t}{\mu_t}
= \frac{q_t}{s_t^{\mathrm{P}}}
\times \frac{s_t^{\mathrm{P}}}{s_t^{\mathrm{D}}}
\times \frac{s_t^{\mathrm{D}}}{\mu_t}
$$

The first term compares training scoring with serving prefill; the second compares prefill with decode; and the third checks the weight version, cache state, and record identity. This decomposition matters: although the final ratio is a single quantity, it spans three interfaces involving cross-engine arithmetic, the two inference paths, and system state. If any of these is not held fixed, we should not loosely attribute the total difference to a kernel error.

At its core, the problem comes down to the non-associativity of floating-point addition. Consider BF16 with round-to-nearest-even:

$$
\operatorname{fl}\!\left(\operatorname{fl}(1 + 2^{-8}) + 2^{-8}\right) = 1
$$

whereas

$$
\operatorname{fl}\!\left(1 + \operatorname{fl}(2^{-8} + 2^{-8})\right) = 1 + 2^{-7}
$$

In real arithmetic, these expressions differ only in their parentheses. In BF16, they produce different answers. Training is designed around packed sequences, backpropagation, and multi-GPU parallelism; inference is designed around prefill, decode, dynamic batching, and the KV cache. Even with shared parameters, these different objectives can lead the engines to choose different partitions, reduction orders, and intermediate precision.

The same weights fix which numbers are added, but not which are added first.

Reusing rollout logprobs and aligning operators are two different things. Reuse determines which recorded values the loss consumes. Operator alignment checks whether the engines obtain the same results when computing independently. Reuse can avoid some consequences of mismatch, but it does not establish alignment.

## Fixing the Mathematical Object and Numerical Contract

Before discussing floating-point error, we must first establish whether the two engines are answering the same question. Write the ideal mathematical object as:

$$
y = F(x, \theta, s, \xi)
$$

Here, $x$ denotes the input, $\theta$ the weights, $s$ the state, including the KV cache, and $\xi$ the random state involved in the computation. What training and rollout actually execute is:

$$
\begin{aligned}
y^{\mathrm{T}} &= \widehat{F}_{C^{\mathrm{T}}}
\left(x^{\mathrm{T}}, \theta^{\mathrm{T}}, s^{\mathrm{T}}, \xi^{\mathrm{T}}\right) \\
y^{\mathrm{R}} &= \widehat{F}_{C^{\mathrm{R}}}
\left(x^{\mathrm{R}}, \theta^{\mathrm{R}}, s^{\mathrm{R}}, \xi^{\mathrm{R}}\right)
\end{aligned}
$$

Here, $C$ is the numerical execution contract.

For logprobs, at a minimum, the following logical objects must match:

* Checkpoint and weight version;
* Prefix, target token, and active mask;
* Position, RoPE, and causal/padding mask;
* Logical K/V obtained from the KV cache mapping;
* Head, sequence, and vocabulary ownership;
* The actual vocabulary range and the random state involved in the computation being compared.

If any of these differs, record `comparable = false`. Physical page numbers and shard layouts may differ, provided that they map back to the same logical tensor. When replaying fixed, previously sampled tokens, there is no need to reproduce the entire history of sampling RNG consumption.

Once this gate is passed, describe each computation node in turn. First, write

$$
y_i = \phi_i(x_{D_i})
$$

Here, $D_i$ is the set of input elements on which output $y_i$ actually depends. If the node can also be written as

$$
y_i = \bigoplus_{k \in \mathcal{R}_i} f(i, k)
$$

then $\mathcal{R}_i$ is its reduction domain. The two sides must first satisfy

$$
D_i^{\mathrm{T}} = D_i^{\mathrm{R}}
\qquad \mathcal{R}_i^{\mathrm{T}} = \mathcal{R}_i^{\mathrm{R}}
$$

Next, partition the reduction domain:

$$
\begin{aligned}
\mathcal{R}_i &= \mathcal{R}_i^{(0)} \cup \mathcal{R}_i^{(1)}
\cup \cdots \cup \mathcal{R}_i^{(m-1)} \\
\Pi_i &= \left\{\mathcal{R}_i^{(0)}, \ldots, \mathcal{R}_i^{(m-1)}\right\}
\end{aligned}
$$

$\Pi_i$ determines which partial summaries are produced first, while the cross-GPU reduction tree determines how they are merged within and across blocks. Even if $\mathcal{R}_i$ is identical, a different cross-GPU reduction tree can produce different bits.

Then record the node's precision tuple

$$
\begin{aligned}
P_v = (&p_{\mathrm{input}}, p_{\mathrm{multiply}}, p_{\mathrm{accumulate}}, \\
       &p_{\mathrm{state}}, p_{\mathrm{output}})
\end{aligned}
$$

and write each rounding operation as

$$
z = Q_p(x)
$$

Collecting the quantities that can independently change the result gives the following minimal arithmetic contract:

$$
C_v = (D_v, \Pi_v, T_v, P_v, R_v, A_v)
$$

Here, $D_v$ captures the dependency sets and reduction domains of the node's outputs; $\Pi_v$ and $T_v$ specify the reduction partition and ordered merge tree; $P_v$ is the precision tuple; $R_v$ records where $Q_p$ occurs; and $A_v$ identifies the specific numerical primitives used, such as exp, log, rsqrt, and SiLU.

Fusion, materialization, and recomputation boundaries are not separate entries in this arithmetic contract. They affect numerical results only when they change $T_v$, $P_v$, $R_v$, or $A_v$, so they are better recorded as execution mechanisms. State and control also remain outside the tuple: the comparability gate checks state such as the cache and RNG, while control conditions such as dispatch and CUDA Graph are treated as triggers. This avoids accounting for the same cause at multiple levels.

$$
\Delta C_v = C_v^{\mathrm{T}} \mathbin{\triangle} C_v^{\mathrm{R}} \ne \varnothing
$$

The expression above represents the set difference between the training and inference arithmetic contracts. This difference identifies candidate root causes of mismatch.

## Nested Reductions in Transformers

RMSNorm, Attention, GEMM, linear logp, and collectives all rely on reductions of this form:

$$
\operatorname{Agg}(\mathcal{R}) = \bigoplus_{i \in \mathcal{R}} u_i
$$

Partition the reduction domain $\mathcal{R}$, compute each local $\operatorname{Agg}(\mathcal{R}^{(j)})$, then merge the results. In real arithmetic, different partitions and parenthesizations are generally considered equivalent. In finite-precision execution, they are not.

| Module | Reduction Domain | Reduction Semantics |
| :---- | :---- | :---- |
| RMSNorm | Hidden dimension | Sum of squares, determining the scale of the entire vector |
| GEMM | K dimension | Sum of products, determining a single output element |
| Attention | Visible keys | Maximum, sum of exponentials, and weighted values |
| Linear logprob | Vocabulary | Maximum, sum of exponentials, and target logit |
| AllReduce, ReduceScatter | Ranks | Local contributions from each GPU |

From this perspective, Split-K, Split-KV, vocabulary sharding, context parallelism, and rank trees simply partition different mathematical axes.

### The Shared Normalization Structure of Attention and Logprob

Given a set of scores $s_i$, let

$$
m = \max_i s_i \qquad l = \sum_i e^{s_i - m}
$$

$m$ is the maximum within the domain, and $l$ is the sum of exponentials relative to that maximum. In real arithmetic, the final result can be expressed using a single log-sum-exp (LSE) value. Actual kernels, however, update and merge $m$ and $l$ separately, so the bitwise contract must retain both intermediate states.

Linear logp needs only $(m, l)$ and the target logit:

$$
\log p(a) = z_a - (m + \log l)
$$

Attention carries an additional vector:

$$
o = \sum_i e^{s_i - m} v_i
\qquad \operatorname{Attn}(q, K, V) = o / l
$$

Both perform the same kind of LSE aggregation, but over different spaces. Attention normalizes over the context to determine which tokens to attend to; logp normalizes over the vocabulary to determine which token to select. Attention's Split-KV merge and logp's vocabulary merge across tensor-parallel (TP) ranks are mathematically two instances of the same problem.

To merge two blocks $(m_1, l_1, o_1)$ and $(m_2, l_2, o_2)$, first set $m = \max(m_1, m_2)$, then compute

$$
\begin{aligned}
l &= e^{m_1 - m} l_1 + e^{m_2 - m} l_2 \\
o &= e^{m_1 - m} o_1 + e^{m_2 - m} o_2
\end{aligned}
$$

In real arithmetic, this merge operation is associative, so any partition can recover the same global result. For an arbitrary number of blocks, the merged result can be written directly as

$$
\begin{aligned}
m &= \max_j m_j \\
l &= \sum_j e^{m_j - m} l_j \\
o &= \sum_j e^{m_j - m} o_j
\end{aligned}
$$

The right-hand side depends only on the set of all blocks, providing a short proof of associativity.

Actual execution, however, uses a merge operation $\widehat{\oplus}$ that includes rounding and approximation. In general,

$$
(\sigma_1 \mathbin{\widehat{\oplus}} \sigma_2)
\mathbin{\widehat{\oplus}} \sigma_3
\ne \sigma_1 \mathbin{\widehat{\oplus}}
(\sigma_2 \mathbin{\widehat{\oplus}} \sigma_3)
$$

where $\sigma_j = (m_j, l_j, o_j)$. The partition $\Pi_i$, reduction tree $T_i$, exp primitive $A_v$, and precision $P_v$ of $m$, $l$, and $o$ are therefore part of the normalization itself.

### Reduction Order in RMSNorm, GEMM, and Communication

RMSNorm uses $\sum_i x_i^2$, and GEMM uses $\sum_k a_{ik} b_{kj}$. After a row-parallel GEMM, AllReduce continues the summation by adding the local sums from each rank.

Suppose the $K$ dimension is partitioned across ranks into disjoint subsets $K_0, \ldots, K_{p-1}$. Then

$$
\begin{aligned}
Y_{ij} &= \sum_{k \in K} X_{ik} W_{kj} \\
       &= \sum_{r=0}^{p-1}
\underbrace{\sum_{k \in K_r} X_{ik} W_{kj}}_{Y_{ij}^{(r)}}
\end{aligned}
$$

Local GEMM produces the inner $Y_{ij}^{(r)}$, while AllReduce computes the outer $\sum_r Y_{ij}^{(r)}$. Mathematically, this is one summation; in the implementation, kernel boundaries divide it into two levels of reduction. The collective is therefore the continuation of the same overall reduction tree beyond a single GPU.

RMSNorm and softmax share another structural feature: both first reduce a domain to a small set of global statistics, then broadcast those statistics back to each local output. For RMSNorm,

$$
\operatorname{RMSNorm}(x)_j
= \gamma_j x_j \left(\epsilon + \frac{1}{d} \sum_{k=1}^{d} x_k^2\right)^{-1/2}
$$

Softmax applies the same $(m, l)$ to every score. Through these shared normalization statistics, a one-bit difference in the reduction can affect the entire hidden vector, an entire Attention row, or the whole vocabulary distribution at once. These operations serve different purposes but share the same numerical structure.

This explains why fixing only one level is insufficient. Disabling Split-K in GEMM removes one class of partial merges within the kernel. If TP AllReduce still uses a different rank tree, however, the complete summation is still parenthesized differently. Conversely, fixing the rank tree cannot replace the warp/CTA reduction contract inside the kernel.

### Fusion and Rounding Boundaries

Suppose one side executes

GEMM → write back as BF16 → SiLU → multiply

while the other executes

GEMM → retain FP32 accumulator → SiLU → multiply → write back as BF16

Both are SwiGLU on paper, but they differ in whether the intermediate tensor is materialized. Fusion, recomputation, and communication staging can affect results by shifting where rounding occurs, as recorded in $R_v$.

Likewise, two implementations labeled FP32 can still produce different bits if they use different exp, log, rsqrt, SiLU, FMA, or fast-math primitives. A dtype describes the container, not the computation in full.

### From Continuous Numerical Error to Discrete Path Divergence

Small score differences can change sampling, argmax, top-k, or threshold decisions, after which the two trajectories are no longer comparable. The relevant discrete boundaries here are masks, positions, cache lookups, selected tokens, and vocabulary ownership.

This is the purpose of fixed replay: freeze the tokens first, so that sampling different outcomes and computing different probabilities for the same token become two separate problems.

## How vime and RL-Kernel Work Together

This framework clarifies how vime and RL-Kernel work together.

A shared training timeline:

prompt → vLLM rollout → token / rollout logp → Megatron scoring → backward → update

vime keeps the engines at the same point in the training process: it tracks which tokens, weight versions, and rollout records feed into each update. RL-Kernel aligns the arithmetic used at that point: how contributions are partitioned, merged, and rounded. In short, vime aligns the training timeline; RL-Kernel aligns the numerical computation.

Without timeline alignment, even fully deterministic kernels may be evaluated against different weights. Without arithmetic alignment, the engines may still compute differently with the same weights. vime constrains the state degrees of freedom in $F$; RL-Kernel constrains the implementation degrees of freedom in $\widehat{F}_C$.

The Qwen3/H100 strict path applies this approach at five boundaries:

| Computation Boundary | What the Strict Path Fixes |
| :---- | :---- |
| RMSNorm | Reduction range, epsilon, residual-add behavior, and output boundary |
| Attention | Position, mask, logical paged KV, split policy, LSE precision, and final cast |
| GEMM / SwiGLU | K-dimension reduction, accumulation precision, epilogue, activation, and materialization boundary |
| Linear logprob | Actual vocabulary range, target ownership, local reduction, and cross-rank LSE merge |
| Distributed collectives | Payload ownership, dtype, and fixed rank reduction order |

`num_splits=1` and no-Split-K are the easiest choices to audit in the current system. A valid contract can also be established as long as both sides fully fix the partition, partial state, and merge tree. Consistency requires only that these choices do not silently alter observable numerical semantics.

### Backward Has Its Own Execution Graph

Rollout has no backward pass, so forward train-rollout parity cannot imply cross-engine backward parity. Backward determinism is an additional training-side requirement, independent of training-inference consistency. Backpropagation introduces new reduction axes, whose execution order can introduce new nondeterminism. For example,

$$
dW = \sum_t dY_t X_t^{\mathsf{T}}
$$

Forward GEMM reduces over the hidden/K dimension; here, the reduction is over the token dimension. Microbatch partitioning, gradient accumulation order, the values saved or recomputed, atomics, and gradient collectives can all change the parenthesization again.

Written as a vector-Jacobian product (VJP), this becomes

$$
(dx, d\theta) = J_F(x, \theta)^{\mathsf{T}}\,dy
$$

The evidence presented here verifies cross-engine consistency of forward logprobs. Reproducibility of the training backward pass must be checked separately: treat the VJP as its own computation graph and examine its domains, partitions, reduction trees, the precision of saved values, and communication. The setting `deterministic_backward=true` is part of this contract.

## Key Triggers of Train-Rollout Mismatch

Batch size, sequence length, prefill/decode, workspace, CUDA Graph, GPU model, and topology often vary alongside mismatch, but they are usually only triggers.

Trigger condition → path selection → $\Delta C_v$ → first numerical divergence

For example:

Batch size changes

→ cuBLASLt heuristic changes

→ number of Split-K partitions changes

→ K-dimension partial merge tree changes

→ GEMM output bits change

Saying that batch size causes mismatch describes a correlation. Saying that batch size triggers a different Split-K reduction tree gets closer to the root cause.

### Single-Variable Ablation

Let $C$ be the fully aligned baseline contract, and change only its $k$th field:

$$
\begin{aligned}
C' &= C - \delta_k \\
\Delta_k(x) &= \widehat{F}(x; C') - \widehat{F}(x; C)
\end{aligned}
$$

Find the first nonzero $\Delta_k$, then trace its propagation through the computation graph into hidden states, logits, selected-token logp, and $\rho_t$. This is more informative than observing a difference in the final logp and suspecting each module in turn.

The implementation choices on the training and rollout sides form a 2×2 matrix, where P denotes the production implementation and R denotes RL-Kernel:

| Combination | Training | Rollout | Purpose |
| :---- | :---- | :---- | :---- |
| P/P | Production | Production | Observe the results of vime's native path |
| P/R | Production | RL-Kernel | Replace only the rollout side and check for divergence |
| R/P | RL-Kernel | Production | Replace only the training side and check for divergence |
| R/R | RL-Kernel | RL-Kernel | Fully aligned strict control path |

When isolating Attention, keep FFN and logp at R/R; apply the same principle when isolating FFN. Any silent fallback on an R side should cause the experiment to fail. Otherwise, the aligned result may not have come from the aligned implementation at all.

R/R, P/R, R/P, and P/P identify the operator implementations used by each engine. The G00-G11 labels used later refer to a different, system-level matrix: whether rollout logprobs are reused and whether aligned operators are enabled.

### Strict Bitwise Consistency Requires Execution Provenance

A credible zero-mismatch result requires at least five layers of evidence:

1. A comparability gate establishes that the objects being compared, state, and ownership match.
2. Fixed-input tests verify repeatability along the same execution path.
3. Training and rollout pass cross-engine parity checks on the same operator input.
4. Selected-token logprobs are compared online under the same weight version.
5. The full workflow archives results for every step and records the backend, device, fallback status, and CUDA and HIP Graph routes.

A configuration file that enables RL-Kernel does not, by itself, prove that RL-Kernel actually ran. Nor does the presence of NCCL in application logs establish whether the target payload used a fixed reduction tree. Numerical results tell us what was computed; execution provenance tells us which path computed it. Both forms of evidence are necessary.

## Progress on Bitwise Alignment with CUDA

The experimental materials for these CUDA results include 200-step records, aggregate tables, bootstrap statistics, validation, and plotting scripts.

### Experimental Configuration

| Item | Configuration |
| :---- | :---- |
| Model / dtype | Qwen3-8B / BF16 |
| Hardware | 1 node, 8× NVIDIA H100 80GB |
| Megatron | TP4 / CP2 / PP1, using 8 GPUs |
| Rollout | 2 vLLM engines, each with TP4 |
| Placement | Actor and rollout colocated |
| Horizon | 200 rollout/training steps |
| Seeds | Training 1234, rollout 1234 |
| Sampling | 8 prompts × 16 samples per step, global batch 128 |
| Response limit | 7,168 tokens |
| Dynamic batching | Maximum 4,096 tokens/GPU |
| vLLM memory utilization | 0.4 |
| CUDA Graph | `FULL_DECODE_ONLY`, retaining the production graph execution path |
| KL loss | Enabled, coefficient 0.001 |
| Snapshot requirement | Exactly 8 rank files per step |

On the strict path, both `mismatch_count` and `max_abs_diff` remained at 0 for all 200 steps.

### Train-Rollout Consistency over 200 Steps

Figure 1 plots train/rollout mismatch count and maximum absolute Δlogp on the same 200-step timeline. RL-Kernel's strict path maintains zero mismatch throughout, while vime's native path exhibits mismatch at every step.

<p align="center" markdown="1">
[![CUDA train-rollout mismatch count and maximum absolute logprob difference][image30]{ width="92%" }][image30]
<br>
*Figure 1: Train-rollout consistency for native vime and vime + RL-Kernel on CUDA.*
</p>

The zero mismatch count and zero maximum absolute logprob difference provide end-to-end evidence that vime + RL-Kernel maintains bitwise train-rollout consistency across all 200 steps.

Figure 2 isolates the mean absolute train/rollout logprob difference over 200 steps. For vime + RL-Kernel, it remains at 0 throughout.

<p align="center" markdown="1">
[![CUDA mean absolute train-rollout logprob difference over 200 steps][image31]{ width="92%" }][image31]
<br>
*Figure 2: Mean absolute train/rollout logprob difference over 200 steps. G10 denotes native vime; G11 denotes vime + RL-Kernel.*
</p>

Figure 3 compares the performance of native vime and vime + RL-Kernel over 200 steps.

<p align="center" markdown="1">
[![CUDA Qwen3-8B performance and consistency comparison][image32]{ width="92%" }][image32]
<br>
*Figure 3: Performance comparison of native vime and vime + RL-Kernel over 200 steps on CUDA.*
</p>

## Progress on Bitwise Alignment with ROCm

We completed a 200-step strict R/R validation on ROCm using the full system of Megatron training and vLLM rollout, with zero mismatch throughout. This validates the integrated stack with all changes incorporated to date, rather than testing a single PR in isolation. The ROCm path uses the same correctness criteria while retaining the native execution paths for AITER, CK, paged KV, HIP Graph, and ROCm collectives. Information read back at runtime verifies the actual backend, execution path, and fallback status.

### Experimental Configuration

| Item | Configuration |
| :---- | :---- |
| Model / dtype | Qwen3-8B / BF16 |
| Hardware | 1 node, 8× AMD Instinct MI300X 192GB |
| Megatron | TP4 / CP2 / PP1, using 8 GPUs |
| Rollout | 2 vLLM engines, each with TP4 |
| Placement | Actor and rollout colocated |
| Horizon | 200 rollout/training steps |
| Seeds | Training 1234, rollout 1234 |
| Sampling | 1 prompt × 8 samples per step, global batch 8 |
| Response limit | 7,168 tokens |
| Dynamic batching | Maximum 4,096 tokens/GPU |
| vLLM memory utilization | 0.38 |
| HIP Graph | `FULL_AND_PIECEWISE`, retaining the production graph execution path |
| KL loss | Enabled, coefficient 0.001 |
| Validation requirement | Frozen inputs and frozen sources must remain unchanged before and after the run; every step must pass runtime provenance and mismatch validation |

On the strict path, both `mismatch_count` and `max_abs_diff` remained at 0 for all 200 steps.

### Train-Rollout Consistency over 200 Steps

Figure 4 plots train/rollout mismatch count and maximum absolute Δlogp on the same 200-step timeline. RL-Kernel's strict path maintains zero mismatch throughout, while vime's native path exhibits mismatch at every step.

<p align="center" markdown="1">
[![ROCm train-rollout mismatch count and maximum absolute logprob difference][image33]{ width="92%" }][image33]
<br>
*Figure 4: Train-rollout consistency for native vime and vime + RL-Kernel on ROCm.*
</p>

The zero mismatch count and zero maximum absolute logprob difference provide end-to-end evidence that vime + RL-Kernel maintains bitwise train-rollout consistency across all 200 steps.

Figure 5 isolates the mean absolute train/rollout logprob difference over 200 steps. For vime + RL-Kernel, it remains at 0 throughout.

<p align="center" markdown="1">
[![ROCm mean absolute train-rollout logprob difference over 200 steps][image34]{ width="92%" }][image34]
<br>
*Figure 5: Mean absolute train/rollout logprob difference over 200 steps. G10 denotes native vime; G11 denotes vime + RL-Kernel.*
</p>

Figure 6 compares the performance of native vime and vime + RL-Kernel over 200 steps.

<p align="center" markdown="1">
[![ROCm Qwen3-8B performance and consistency comparison][image35]{ width="92%" }][image35]
<br>
*Figure 6: Performance comparison of native vime and vime + RL-Kernel over 200 steps on ROCm.*
</p>

## Putting It All Together

These modules may seem unrelated to readers unfamiliar with distributed kernels. The argument connecting them is:

1. A model formula defines only the real-valued function $F$, not a unique implementation $\widehat{F}_C$ governed by a numerical contract.
2. Before a parameter update, differences between training and rollout logprobs create a spurious policy ratio.
3. Before comparing errors, pass the comparability gate to ensure that tokens, weights, positions, masks, cache, and ownership refer to the same object.
4. View the Transformer as nested reductions: RMSNorm aggregates over hidden dimensions, GEMM over features, Attention over keys, logp over the vocabulary, and collectives over ranks.
5. For each reduction, ask the same questions: what participates, how is it partitioned, which tree merges it, when does rounding occur, and which primitives are called?
6. Attention and logp both involve LSE, while collectives perform cross-GPU reductions. These modules therefore need a single, continuous end-to-end contract.
7. vime fixes the temporal relationships among tokens, state, and weights; RL-Kernel fixes how key operators carry out their arithmetic. Together, they turn "the same policy" into conditions the system can enforce.
8. Batch size, sequence length, and CUDA Graph are only triggers. They must be traced to specific partitions, merge trees, or rounding boundaries.
9. Finally, a claim of bitwise consistency requires single-variable replacements to locate the first divergence, along with archived outputs and execution provenance.

Train-rollout mismatch is an abstraction leak: higher layers mistake mathematical equivalence for numerical equivalence. RL-Kernel makes numerical consistency part of the interface between engines, rather than just a property of individual deterministic operators. The numerical contract also specifies which observable behavior optimizations must preserve. Training and inference may still use different memory layouts, parallelism schemes, and scheduling strategies. As long as those changes preserve dependency domains and rounding boundaries, they remain within the same verifiable implementation. What must be eliminated is undeclared arithmetic variation.

## Next Steps

* Expand support to more models and multimodal architectures.
* Continue porting to MUSA, Ascend, and additional hardware platforms.
* Advance integration with Miles and AReaL.

Models, hardware, and execution frameworks for RL post-training will continue to change. RL-Kernel aims to keep explicit correctness criteria in the system so that every kernel replacement, framework upgrade, or hardware migration can be checked for preserved numerical semantics, with any divergence traced to its starting point.

## Acknowledgments

The release of RL-Kernel v0.1.0 would not have been possible without the generous support of our hardware partners, open-source ecosystem partners, and core development team.

### Hardware and Compute Partners

We sincerely thank Liz Li and Yuhan Yang from AMD for providing AMD Instinct GPU compute resources, close technical collaboration, and long-term support for RL-Kernel. We look forward to continuing our work on cross-platform consistency validation, kernel-level performance optimization, and deployment of large-scale RL workloads on ROCm.

We also thank Lei Ding from Moore Threads for advancing MUSA platform support, and Yang Chen from Huawei for advancing Ascend platform support. We are grateful to Embedded LLM for supporting the project's research and development and community collaboration.

Consistent execution and generalization across heterogeneous hardware platforms are central to RL-Kernel's long-term goals. We welcome collaboration with more hardware vendors and open-source communities to build open, efficient RL-Kernel infrastructure together.

### Open-Source Ecosystem and Framework Collaboration

We thank the vLLM community for its close collaboration with RL-Kernel. Special thanks go to Ao Shen, vime maintainer at Inferact, for the trust and support throughout the RL-Kernel and vime integration, community collaboration, and ongoing maintenance. This work builds on the open-source ecosystem of vLLM rollout, vime orchestration, and Megatron training.

### Core Contributors - v0.1.0

Special thanks to the RL-Kernel core contributors for their work on architecture design, kernel implementation, operator-level training-inference consistency for dense models, distributed validation, and community building for v0.1.0.

Chutian Wang:

* Developed inter-node and inter-GPU communication modules for CUDA.
* Built the infrastructure for the cross-platform ablation matrix.

Jiajie Li:

* Led the evaluation of the vime framework, planned the fork's roadmap, and delivered PRs.
* Led vime and RL-Kernel integration on both CUDA and ROCm.
* Led Distributed Attention development.
* Completed `linear_logp` replacement experiments, including TP parallelization, and wrote the accompanying technical blog post.
* Led end-to-end training-and-rollout testing and performance tuning for native vime and RL-Kernel + vime on both CUDA and ROCm.

Siru He:

* Led the setup of the WS1 GTest unit testing framework.
* Led work on distributed support for the GEMM operator on CUDA and ROCm and delivered the corresponding PRs.

Xiaosong Ma:

* Developed the WS1 Attention operator.
* Led the full development and testing workflow for individual WS1 operators and the setup of the GTest framework.
* Optimized GEMM performance in end-to-end training-and-rollout tests on CUDA.
* Delivered communication PRs for ROCm.
* Contributed to code reviews and assisted with end-to-end training-and-rollout testing and performance tuning for native vime and RL-Kernel + vime on ROCm.

Kaijie Lin:

* Led development of the standalone Logprob operator and the distributed implementation PRs.
* Implemented the Triton-based deterministic fused Linear-Logp operator for ROCm.
* Developed Logprob TP parallelization for the vime integration experiments.

Jian Zhang:

* Developed the RoPE operator.
* Contributed to Distributed Attention development.
* Led operator porting to Ascend and delivery of the corresponding PRs.

Huihong Lu:

* Developed the standalone Triton Logprob operator, including ROCm support.
* Contributed to distributed support for Logprob.

Yunxiang Cai:

* Developed the standalone RMSNorm operator.
* Implemented the standalone Attention operator on the Triton path.

Vensen Mu:

* Developed the standalone GEMM operator and optimized its performance in training-and-rollout tests on CUDA.
* Led the work to adapt RL-Kernel to ROCm.
* Delivered communication PRs for ROCm.
* Led end-to-end training-and-rollout benchmarking and performance tuning for native vime and RL-Kernel + vime on ROCm.

Bosong Yang:

* Contributed to research and development of the Distributed Attention operator.

Zhewei Liu:

* Contributed to research and development of the Distributed Attention operator.

Houhong Liang:

* Handled performance profiling, benchmarking, and tuning of the end-to-end training-and-rollout pipeline on CUDA.

Ryan Huang:

* Contributed to development of the standalone Logprob operator and the distributed implementation PRs.

Finally, we thank our community contributors: Xiaopeng Du, Yuepeng Pan, Yiyang Fei, Ziying Tao, Zhifu Liu, Zhengtao Chen, Mengjie Li, Zien Liu, GitHub: haoruilee, GitHub: luoyueyuguang, GitHub: hongleng, GitHub: smarslou.

[image30]: ../assets/blog/rl-kernel-v0.1.0/cuda-training-consistency.png

[image31]: ../assets/blog/rl-kernel-v0.1.0/cuda-mean-logprob-difference.png

[image32]: ../assets/blog/rl-kernel-v0.1.0/cuda-performance.png

[image33]: ../assets/blog/rl-kernel-v0.1.0/rocm-training-consistency.png

[image34]: ../assets/blog/rl-kernel-v0.1.0/rocm-mean-logprob-difference.png

[image35]: ../assets/blog/rl-kernel-v0.1.0/rocm-performance.png
