---
layout: post
title: "DSV4 Flash MoE Train-Inference Consistency Roadmap"
author: "RL-Kernel Contributors"
image: "../../assets/blog/dsv4-moe-consistency-risk-map.png"
summary: "A public roadmap for operator-level train-inference consistency on DeepSeek-V4-Flash-0731 MoE models, focusing on where model architecture creates drift between Megatron training and vLLM inference."
read_time_minutes: 8
tags:
  - reinforcement-learning
  - kernels
  - moe
  - post-training
---

We are developing operator-level train-inference consistency for Qwen3 Dense models, following the WS1 -> WS2 split introduced in RL-Kernel #83. The next step is to extend the same methodology to Flash MoE models such as `DeepSeek-V4-Flash-0731`.

A more detailed v4 roadmap will be published later as an RL-Kernel GitHub issue.

In Dense models, the main consistency surfaces are usually continuous numeric paths such as matmul, attention, and logprob. MoE makes the problem more discrete. A tiny router difference can change the selected top-k expert. A different token order, padding choice, or all2all layout can change the expert input tensor. A different floating-point reduction order in weighted combine can make `dlogp` drift across batch, pack, TP, CP, and EP configurations.

<p align="center">
<img src="../../assets/blog/dsv4-moe-consistency-risk-map.png" alt="DSV4 Flash MoE train-inference drift surfaces" width="92%">
<br>
<em>The main DSV4 Flash MoE drift surfaces: routing, fused materialization, final reduction, and the batch / pack / padding / EP / TP / CP provenance above them.</em>
</p>

## Why DSV4 Flash MoE Drifts More Easily

Public examples suggest that the inference side may place expert parallelism, low-precision caches, chunked prefill, hybrid attention, speculative decoding, and other engineering paths into the same execution graph. That means inference is not a plain PyTorch graph. It is a highly engineered, fused, and parallel execution path.

Our working boundary is:

- Training side: Megatron
- Inference side: vLLM
- Hardware: NVIDIA Hopper
- Consistency definition: follow RL-Kernel #83 / #108, prefer bitwise equality, otherwise tight numerical tolerance
- Scope: forward + backward, deterministic reduction order, batch invariance, padding, and parallel configurations
- Speculative decoding: not the first-phase consistency mainline, but its control-flow impact should be recorded

Within this boundary, DSV4 Flash MoE introduces several important consistency risks.

## Problem 1: Routing Is A Discrete Decision

The MoE router computes router logits from hidden states and then selects top-k experts. The risk is not just a last-bit numeric difference. If two expert scores are close, small differences in accumulation dtype, tie-breaking, expert bias, hash routing, or padding handling can lead training and inference to select different experts.

Once the expert id differs, the following MLP is no longer computing the same token through the same path. A continuous numeric difference has become a discrete path split.

The roadmap approach is deliberately conservative. WS1 should first record and align router logits, top-k ids, top-k weights, tie-breaking rules, and routing metadata. Under the same checkpoint, tokens, mask, and seed, routing metadata should stay stable across batch and padding sweeps. Any mismatch should be reported explicitly instead of being hidden inside downstream `dlogp` statistics.

Router softmax and top-k deserve special attention. The implementation path may be CUDA, Triton, or another backend, but the public roadmap only commits to one principle: first validate precision and stability, then decide whether a fast path is appropriate.

## Problem 2: Single-Operator Boundaries May Not Be Meaningful

A naive MoE decomposition looks like this:

```text
gather -> gate_and_up -> down -> scatter
```

That decomposition does not necessarily match real inference execution boundaries.

vLLM may use `fused_moe`-style paths that organize MoE into a few semantic phases: prepare, expert compute, and weighted combine. Prepare may include activation quantization and dispatch. The expert phase may include permutation, expert MLP, unpermutation, and workspace handling. The combine step may live near the final stage rather than as a separately materialized tensor boundary.

The training side should not be assumed to match this either. Megatron may keep more explicit router, dispatch, expert MLP, and combine boundaries. Some paths may also use grouped GEMM, activation packing, or communication overlap. In other words, both training-side materialization and inference-side fused materialization need investigation. A kernel boundary from either side should not automatically become the consistency contract.

This means that comparing only "gather" or "scatter" may be meaningless. The middle tensors may not exist with the same shape, order, dtype, or semantic ownership. The comparison should instead focus on semantic boundaries:

- Routing boundary: router logits, top-k ids, top-k weights
- Dispatch boundary: token-to-expert map, valid token counts, padding provenance
- Expert boundary: expert inputs and outputs, accumulation dtype, quantization and dequantization points
- Shared-expert boundary: whether shared experts run independently or overlap with combine / communication
- Reduction boundary: weighted combine order, dtype, and downcast point

<p align="center">
<img src="../../assets/blog/dsv4-moe-fused-boundaries.png" alt="DSV4 MoE fused semantic boundaries" width="92%">
<br>
<em>gather / gate_and_up / down / scatter should not be treated as natural acceptance units. The roadmap should validate semantic boundaries instead.</em>
</p>

vLLM is a candidate backend, not the source of truth. The RL-Kernel reference path, semantic trace, and deterministic comparison should define the contract.

## Problem 3: Final Reduction Is The Core Drift Point

The most important place to watch in DSV4 MoE is weighted expert combine. Multiple expert outputs are multiplied by top-k weights and merged back into the token hidden state. This is the final floating-point reduction point of the MoE block.

Looking downward, this point depends on expert MLP matmul accumulation, activation, low-precision scales, and downcast. Looking upward, it is affected by batch, pack, padding, EP all2all, TP shards, CP boundaries, and collective scheduling. Many issues that appear to be batch inconsistency or padding inconsistency eventually show up as numeric drift at this reduction point.

Shared experts, backward propagation, and low-precision paths should be viewed through the same lens. We should first reuse or enable existing determinism controls where available and evaluate mature backend options carefully. If a kernel-level expansion, scheduling choice, or fusion choice changes the reduction order, it should be recorded as an implementation risk rather than a new semantic interpretation of the model.

## Problem 4: Attention Can Pollute MoE

DSV4 Flash is not only MoE. The inference side may also involve SWA, HCA / CSA, chunked prefill, FP8 KV cache, FP4 indexer cache, and long-context compression or hybrid attention paths. The observed drift may appear at MoE, but the hidden-state difference may already have been introduced by an attention block.

The roadmap therefore treats the attention boundary as P0. HCA / CSA names should not directly become acceptance operators. We still need to inspect the real implementation boundary. WS1 should first make the hidden-state boundary before MoE recordable and comparable. If attention is included in the full chain, reports should distinguish attention-induced drift from MoE-induced drift.

## Problem 5: RL Ultimately Observes dlogp

For RL post-training, the most sensitive signal is not a single intermediate tensor. It is the train-rollout logprob difference:

```text
dlogp = training_recomputed_logp - rollout_old_logp
```

MoE routing, expert combine, attention cache, TP vocab-parallel logprob, padding masks, and action masks can all flow into `dlogp`. If we only inspect single-operator numeric error and do not look at `dlogp` on active response / action tokens, we may miss the drift that actually affects PPO / GRPO updates.

## R3 Should Not Be A Prerequisite

R3 routing replay is useful, but we do not treat it as a prerequisite for the DSV4 MoE train-inference consistency roadmap.

R3 can replay rollout-side routed experts and reduce a large class of routing mismatch. But if the roadmap only works under R3, we can no longer tell whether the Megatron router and vLLM router are semantically consistent on their own.

R3 should be an audit, noise-reduction, and provenance tool, not the source of truth for the MoE semantic contract. If R3 is enabled, it must be recorded in provenance. If it is disabled, the same MoE contract should still be independently verifiable.

## RL-Kernel's WS1 -> WS2 Path

We have found a way to structure DSV4 Flash MoE inside the RL-Kernel framework. The goal is not to create a separate numeric standard for DSV4, but to reuse the WS1 -> WS2 design already established for Dense models.

<p align="center">
<img src="../../assets/blog/dsv4-moe-ws-roadmap.png" alt="RL-Kernel DSV4 Flash MoE WS1 WS2 Roadmap" width="92%">
<br>
<em>The DSV4 Flash MoE consistency path: WS1 stabilizes single-GPU semantic boundaries, and WS2 handles EP / TP / CP plus cross-config dlogp.</em>
</p>

### WS1: Single-GPU Semantic Operator Layer

WS1 extends the ground-truth harness, records `ModelProfile`, `LayerContract`, and `SemanticTrace`, runs MoE semantic boundary sweeps, and finally checks single-GPU full-chain forward + backward + `dlogp`.

### WS2: Multi-GPU Distributed Consistency

WS2 writes TP, CP, EP, DP, batch, pack, padding, and collective order into the contract. It then covers deterministic collectives and cross-config `dlogp` alignment between Megatron and vLLM. EP is more likely to expose routing / dispatch issues. TP, DP, and CP are more likely to mix MoE blocks, logprob, and communication overlap. Specific backend changes and kernel wrapping details will not be expanded here.

The experiment matrix should stay restrained. It is used to isolate implementation differences, not to introduce new model semantics into the comparison itself.

## Follow-Up Issue Buckets

| Issue | Focus | Acceptance Signal |
| --- | --- | --- |
| DSV4-WS1-0 | ModelProfile + SemanticTrace | Same checkpoint / tokens / mask / seed; fail closed |
| DSV4-WS1-1 | MoE semantic boundary sweep | Routing / dispatch / combine align across batch and padding sweeps |
| DSV4-WS1-2 | Single-GPU full-chain | Forward + backward; report drift percentiles and dlogp |
| DSV4-WS2-0 | EP / TP / CP provenance | Align under supported layouts; per-rank drift must be reported |
| DSV4-WS2-1 | Deterministic collectives | Unsupported collective / topology combinations fail closed |
| DSV4-WS2-2 | Cross-config dlogp | Align on active tokens; report RL health signals |

## Join Us

DSV4 Flash MoE train-inference consistency is not a single-kernel task. It spans Megatron, vLLM, MoE routing, expert parallelism, Hopper kernels, low-precision numerics, and RL logprob diagnostics. We will continue development around the WS1 / WS2 issue buckets above.

If you are familiar with CUDA / Triton / Hopper, vLLM fused MoE, Megatron MoE, distributed communication, or logprob / KL / ratio diagnostics in RL post-training, we welcome you to join the DSV4 MoE consistency effort. We want to make this work reusable operator-level infrastructure. To join the MoE development work, contact WeChat: `iZzy07Zoey12`.

## Acknowledgments

Thanks to Embedded LLM, Chutian Wang, Jiajie Li, Siru He, Xiaosong Ma, Kaijie Lin, Jian Zhang, Bosong Yang, Yunxiang Cai, Huihong Lu, Zhewei Liu, Zhengtao Chen, and Ziying Tao, Zhipeng Wang, Vensen Mu for their work on Dense model operator-level train-inference consistency and for supporting the DSV4 MoE investigation.
