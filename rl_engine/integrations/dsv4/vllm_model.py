# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4 on vLLM with the RL-Kernel batch-invariant profile (rollout side of verl).

The rollout must give every token the bytes of the actor's ``Dsv4Model``. vLLM's fork
(``ISEEKYAN/vllm@1a6542c``, ``VLLM_BATCH_INVARIANT=1``) already matches it in the
indexer, the compressors, FlashMLA sparse attention and the FP8 projections; this module
keeps those, and vLLM's scheduler, paged KV cache and sampler, and replaces the rest
(design doc section 10):

- mHC: ``mhc_pre`` / ``mhc_post`` / ``hc_head`` (profile ``dsv4-mhc-v1``); the first layer
  reads four identical embedding streams.
- MoE: the whole block (router, SM90 MXFP4 experts, FP8 shared expert, expert parallelism
  over vLLM's EP group) is ``Dsv4MoE``, loaded from the checkpoint by RL-Kernel's loader;
  vLLM never allocates its own experts.
- RoPE: sliding-window-only layers use plain RoPE (vLLM applies YaRN to every layer).
- Logits and log-probs (profile ``dsv4-logprob-v1``): FP32 logits from the BF16 head on
  DeepGEMM; the reported log-prob of each sampled token is recomputed with the 64-tile
  fixed-order log-sum-exp (:class:`RlkSampler`). vLLM's sampling itself is unchanged.
- ``q_norm`` / ``kv_norm``: RL-Kernel's RMSNorm (vLLM's fused pair rounds a rare row
  differently, which then sits in the KV cache for a whole window).
- Indexer query RoPE + FP8 quant: RL-Kernel's ``indexer_q_rope_fp8_cuda``.
- Compressors (main C4 / C128 KV and indexer K): vLLM saves the partial states; the
  compress -> RMSNorm -> RoPE -> FP8 quant -> store step is RL-Kernel's.
- Query norm / RoPE / KV insert: RL-Kernel's ``q_norm_rope_kv_insert`` writes vLLM's paged
  ``fp8_ds_mla`` cache (same layout); vLLM's fused kernel rounds differently in about one
  query row in ten (``scripts/probe_dsv4_vllm_attention_ops.py``).
- Sparse attention: vLLM's DSv4 FlashMLA layer (gathers, indices, sink) on RL-Kernel's
  pinned FlashMLA build (:func:`_use_pinned_flashmla`); ``VLLM_DS4_DECODE_KERNEL=sparse``
  makes decode use the same kernel as prefill.
- Output projection: ``wo_a`` + ``wo_b`` run as RL-Kernel's ``OutputProjection`` (inverse
  RoPE, FP8 quant, ``fp8_einsum``, ``wo_b``), loaded from the checkpoint. vLLM's
  batch-invariant mode on Hopper replaces ``fp8_einsum`` with a Triton block-scaled GEMM per
  group, which rounds differently.
- LoRA (unmerged, profile ``dsv4-lora-v1``): ``wq_a`` / ``wkv`` / ``wq_b`` wrap vLLM's
  projections (the base output is untouched, the delta is added the actor's way); ``wo_b``
  is applied inside the output projection; the MoE adapters live in ``Dsv4MoE``.

No vLLM source changes: :func:`register` (called by RL-Kernel's ``vllm.general_plugins``
entry point when ``RL_KERNEL_DSV4_ROLLOUT=1``) registers :class:`RlkDeepseekV4ForCausalLM`
under vLLM's ``DeepseekV4ForCausalLM`` architecture name.

Supported layout: tensor parallel 1, pipeline parallel 1, no sequence parallel, no MTP, no
EPLB; data parallel attention with expert parallel MoE (``--enable-expert-parallel``).

CUDA graphs: the fork's breakable graphs (auto-enabled for DSv4; attention runs eagerly
inside them). With expert parallelism, a forward of at most ``RL_KERNEL_DSV4_EP_STATIC_TOKENS``
tokens on every data-parallel rank (default: vLLM's largest capture size, 0 when eager) uses
the MoE's static exchange (``Dsv4MoE.ep_static_tokens``: no host sync, fixed shapes, same
bytes); larger forwards keep the dynamic one. The graphs read the adapters in place, so
``RL_KERNEL_DSV4_LORA=rank,alpha`` reserves them before the capture and every sync copies
into them.

Ablation (``RL_KERNEL_DSV4_NON_BI=1`` with ``VLLM_BATCH_INVARIANT=0``): the rollout drops
what it does only for batch invariance and byte equality with the actor, and keeps what
defines the model. vLLM's own paths come back: its non-invariant attention planning
(batched prefill chunks, workspaces sized by the batch), FlashMLA's paged FP8 decode
(``VLLM_DS4_DECODE_KERNEL=paged``), its fused q / kv RMSNorm, query norm / RoPE / KV insert,
indexer query and compressor store kernels, and the aux-stream overlap of the indexer and
compressor. RL-Kernel's mHC, MoE, output projection, LoRA adapters, RoPE tables, FP32 head
and log-prob stay (vLLM has no LoRA path for these layers; the RoPE tables keep the model
the actor trains). The log-probs then differ from the actor's; the point is the rollout's
speed without batch invariance. The actor side has no such switch.

Debug: ``RL_KERNEL_DSV4_DUMP=<dir>`` (once ``<dir>`` exists, so a caller can set it before
the engine's worker processes start and skip their warmup) saves each forward's tokens,
request boundaries,
per-layer streams and final hidden state (``scripts/check_dsv4_vllm_gate.py``).
``RL_KERNEL_DSV4_TEST_LORA=rank,alpha,seed`` attaches the random adapters
``build_lora_adapters(..., zero_b=False)`` makes, for the gate.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import pathlib
import re
import sys
import types
from itertools import islice
from typing import TYPE_CHECKING

import torch
from torch import nn


def _use_pinned_flashmla() -> None:
    """Serve vLLM's vendored FlashMLA interface from RL-Kernel's pinned FlashMLA build.

    Both builds register the ``torch.ops._flashmla_C`` namespace, so only one can be loaded
    per process, and vLLM's (same commit, other build flags) rounds some sparse-attention
    rows differently from the actor's. Loading ours first and aliasing ``vllm._flashmla_C``
    makes vLLM's interface call our kernels. Must run before vLLM imports its attention ops.
    """
    alias = sys.modules.get("vllm._flashmla_C")
    if alias is not None:
        if getattr(alias, "rlk_alias", False):
            return
        raise RuntimeError(
            "vLLM's FlashMLA library is already loaded; import "
            "rl_engine.integrations.dsv4.vllm_model before vLLM's attention ops "
            "(the RL_KERNEL_DSV4_ROLLOUT=1 plugin does)"
        )
    from rl_engine.kernels.ops.cuda.attention.flashmla import _import_interface

    _import_interface()  # loads flash_mla/_flashmla_C.abi3.so
    alias = types.ModuleType("vllm._flashmla_C")
    setattr(alias, "rlk_alias", True)
    sys.modules["vllm._flashmla_C"] = alias


_use_pinned_flashmla()

import vllm.models.deepseek_v4.nvidia.model as vm  # noqa: E402
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config  # noqa: E402
from vllm.v1.sample.sampler import Sampler  # noqa: E402

if TYPE_CHECKING:
    from rl_engine.models.deepseek_v4.lora import MoeLora

ENV_ENABLE = "RL_KERNEL_DSV4_ROLLOUT"
ENV_DUMP = "RL_KERNEL_DSV4_DUMP"
ENV_TEST_LORA = "RL_KERNEL_DSV4_TEST_LORA"
ENV_NON_BI = "RL_KERNEL_DSV4_NON_BI"
ENV_EP_STATIC = "RL_KERNEL_DSV4_EP_STATIC_TOKENS"
ENV_LORA = "RL_KERNEL_DSV4_LORA"


def batch_invariant() -> bool:
    """False under the ``RL_KERNEL_DSV4_NON_BI=1`` ablation (module docstring)."""
    return os.environ.get(ENV_NON_BI, "0") != "1"


class RlkDeepseekV4FP8Config(DeepseekV4FP8Config):
    """vLLM's DeepSeek-V4 FP8 quantization config, unchanged, but defined here.

    vLLM pickles the engine config into each spawned engine-core / worker process, and
    unpickling the quantization config imports its module there before any plugin runs.
    vLLM's own module sits in the DeepSeek-V4 package, whose ``__init__`` loads vLLM's
    FlashMLA build; this one imports this module first, which installs RL-Kernel's build
    (:func:`_use_pinned_flashmla`)."""


def register() -> None:
    """Make vLLM build :class:`RlkDeepseekV4ForCausalLM` for DeepSeek-V4 checkpoints."""
    import vllm.v1.worker.gpu_model_runner as runner
    from vllm import ModelRegistry
    from vllm.model_executor.layers.quantization import register_quantization_config

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM", "rl_engine.integrations.dsv4.vllm_model:RlkDeepseekV4ForCausalLM"
    )
    import vllm.models.deepseek_v4.attention as vattention
    import vllm.models.deepseek_v4.compressor as vcompressor
    import vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl as vcutedsl
    import vllm.v1.worker.gpu.model_runner as runner_v2

    runner.Sampler = RlkSampler
    runner_v2.Sampler = _rlk_sampler_v2()
    if os.environ.get(ENV_DUMP):
        for cls in (runner.GPUModelRunner, runner_v2.GPUModelRunner):
            _mark_dummy_runs(cls)
    register_quantization_config(DeepseekV4FP8Config.get_name())(RlkDeepseekV4FP8Config)
    if not batch_invariant():
        return  # the ablation keeps vLLM's attention ops (module docstring)
    vattention.fused_q_kv_rmsnorm = _rlk_q_kv_rmsnorm
    vattention.fused_indexer_q_rope_quant = _rlk_indexer_q
    vcompressor.compress_norm_rope_store_triton = _rlk_compress_store
    vcompressor.compress_norm_rope_store_two_stage_triton = _rlk_compress_store
    vcutedsl.compress_norm_rope_store_cutedsl = _rlk_compress_store
    _accept_pinned_deepgemm()
    _keep_nccl_bandwidth()
    # Decode through FlashMLA's sparse kernel, as prefill: the paged FP8 decode kernel
    # rounds differently (checked in RlkDeepseekV4Model).
    os.environ.setdefault("VLLM_DS4_DECODE_KERNEL", "sparse")


_IN_DUMMY_RUN = [False]


def _mark_dummy_runs(cls) -> None:
    """Flag the runner's dummy forwards (memory profiling, and a data-parallel rank with no
    requests that joins its peers' collectives), so the debug dump skips them."""
    if getattr(cls._dummy_run, "_rlk_marked", False):
        return
    orig = cls._dummy_run

    @functools.wraps(orig)
    def run(self, *a, **kw):
        _IN_DUMMY_RUN[0] = True
        try:
            return orig(self, *a, **kw)
        finally:
            _IN_DUMMY_RUN[0] = False

    setattr(run, "_rlk_marked", True)
    cls._dummy_run = run


def _rlk_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):
    """vLLM's ``fused_q_kv_rmsnorm`` on RL-Kernel's RMSNorm (the actor's q / kv norms)."""
    from rl_engine.backends.cuda.norm.rmsnorm import rmsnorm_cuda

    return (
        rmsnorm_cuda(qr.contiguous(), q_weight, eps),
        rmsnorm_cuda(kv.contiguous(), kv_weight, eps),
    )


def _rlk_compress_store(
    *,
    state_cache,
    num_actual,
    token_to_req_indices,
    positions,
    block_table,
    cos_sin_cache,
    kv_cache,
    k_cache_metadata,
    head_dim,
    compress_ratio,
    overlap,
    use_fp4_cache,
    rms_norm_weight,
    rms_norm_eps,
    **_unused,
):
    """vLLM's compress -> RMSNorm -> RoPE -> FP8 quant -> cache store step (both the main
    ``fp8_ds_mla`` KV and the indexer K) on RL-Kernel's compressor kernels. vLLM's
    ``save_partial_states`` (a copy and one FP32 add) has filled ``state_cache``."""
    from rl_engine.kernels.ops.cuda.compressor import compress_indexer_cuda, compress_main_cuda

    if use_fp4_cache or kv_cache.dtype != torch.uint8:
        raise NotImplementedError("RL-Kernel DSv4 rollout: FP8 compressed caches only")
    pos = positions[:num_actual]
    slots = k_cache_metadata.slot_mapping[:num_actual].long()
    slots = torch.where((pos + 1) % compress_ratio == 0, slots, -1)
    args = (
        state_cache,
        block_table,
        token_to_req_indices[:num_actual].int(),
        pos,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        kv_cache.view(kv_cache.shape[0], -1),
        kv_cache.shape[1],
        slots,
    )
    if head_dim == 512:
        compress_main_cuda(*args, compress_ratio)
    elif head_dim == 128 and compress_ratio == 4 and overlap:
        compress_indexer_cuda(*args)
    else:
        raise NotImplementedError(
            f"RL-Kernel DSv4 compressor: head {head_dim}, ratio {compress_ratio}"
        )


def _rlk_indexer_q(
    positions,
    index_q,
    cos_sin,
    weights,
    softmax_scale,
    head_scale,
    use_fp4=False,
    output_buffers=None,
):
    """vLLM's DSv4 ``fused_indexer_q_rope_quant`` (FP8 path) on RL-Kernel's kernel."""
    if use_fp4:
        raise NotImplementedError("RL-Kernel DSv4 rollout: MXFP4 indexer keys")
    from rl_engine.kernels.ops.cuda.indexer.indexer_q import indexer_q_rope_fp8_cuda

    return indexer_q_rope_fp8_cuda(index_q, positions, cos_sin, weights, softmax_scale, head_scale)


def _accept_pinned_deepgemm() -> None:
    """Let ``VLLM_BATCH_INVARIANT=1`` start on RL-Kernel's pinned DeepGEMM.

    vLLM fails closed unless DeepGEMM exposes ``set_batch_invariant`` (a fork whose flag makes
    its grouped MoE GEMMs invariant). This model runs no vLLM MoE kernel, and RL-Kernel's
    pinned DeepGEMM (``rl_engine.kernels.ops.cuda.matmul.deepgemm``) is batch-invariant at the DSv4
    shapes as built (``scripts/probe_dsv4_deepgemm_bi.py``), so the check becomes the pin
    check. A DeepGEMM with the flag is left to vLLM."""
    import vllm.utils.deep_gemm as vdg

    if vdg.supports_deep_gemm_batch_invariance():
        return

    def enable() -> None:
        from rl_engine.kernels.ops.cuda.matmul.deepgemm import require_deepgemm

        require_deepgemm()

    vdg.enable_deep_gemm_batch_invariance = enable


ENV_NCCL_DETERMINISTIC = "RL_KERNEL_DSV4_NCCL_DETERMINISTIC"


def _keep_nccl_bandwidth() -> None:
    """Undo the NCCL part of vLLM's batch-invariant environment.

    ``VLLM_BATCH_INVARIANT=1`` makes every worker set one channel, one thread, the Simple
    protocol and tree all-reduce before NCCL starts, so that NCCL's reductions sum in a
    fixed order. This model runs no NCCL reduction on model data (TP 1, checked in
    :class:`RlkDeepseekV4ForCausalLM`; the MoE's EP exchange is ``all_to_all_single``, pure
    byte movement, and DP's synchronisation all-reduces integer token counts), so the
    settings only cut the exchange's bandwidth: about 8x at 64 requests per rank (nsys,
    2026-11-21; design doc 10.3). The NCCL variables get back the values they had before
    vLLM's override. ``RL_KERNEL_DSV4_NCCL_DETERMINISTIC=1`` keeps vLLM's settings."""
    if os.environ.get(ENV_NCCL_DETERMINISTIC, "0") == "1":
        return
    import vllm.model_executor.layers.batch_invariant as vbi

    if getattr(vbi.override_envs_for_invariance, "rlk_wrapped", False):
        return
    override = vbi.override_envs_for_invariance

    def keep_nccl() -> None:
        saved = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
        override()
        for k in [k for k in os.environ if k.startswith("NCCL_")]:
            if k in saved:
                os.environ[k] = saved[k]
            else:
                del os.environ[k]

    setattr(keep_nccl, "rlk_wrapped", True)
    vbi.override_envs_for_invariance = keep_nccl


_WARNED: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        import logging

        logging.getLogger(__name__).warning("RL-Kernel DSv4: %s", msg)


def _rlk_logprobs(z, ids, temps) -> torch.Tensor:
    """FP32 log-prob of ``ids`` under ``softmax(z / T)``; ``temps`` FP32 per row (``None`` =
    1). The kernel scales by ``T`` itself, as the actor's does: one call per distinct ``T``."""
    from rl_engine.kernels.ops.cuda.logprob_dsv4 import logprob_from_logits_cuda

    if temps is None:
        return logprob_from_logits_cuda(z, ids, 1.0)[0]
    t = torch.where(temps < 1e-5, 1.0, temps).float()  # greedy rows: T = 1, as vLLM does
    logp = torch.empty(ids.shape[0], dtype=torch.float32, device=z.device)
    for tv in torch.unique(t).tolist():
        rows = (t == tv).nonzero().flatten()
        logp[rows] = logprob_from_logits_cuda(z[rows], ids[rows], tv)[0]
    return logp


def _rlk_sampler_v2():
    """:class:`RlkSampler` for vLLM's V2 model runner (DeepSeek-V4's default)."""
    from vllm.v1.worker.gpu.sample.sampler import Sampler as SamplerV2

    class RlkSamplerV2(SamplerV2):
        def __call__(self, logits, input_batch):
            mode = self.logprobs_mode
            z = logits.float().clone() if mode in ("processed_logprobs", "raw_logprobs") else None
            out = super().__call__(logits, input_batch)
            lt = out.logprobs_tensors
            if z is None or lt is None or lt.logprobs.dim() != 2 or lt.logprobs.shape[0] == 0:
                return out
            if z.shape[0] != input_batch.idx_mapping_np.shape[0]:
                raise NotImplementedError("RL-Kernel DSv4 log-probs: speculative decoding")
            temps = None
            if mode == "processed_logprobs":
                top_k, top_p = self.sampling_states.get_top_k_top_p(
                    input_batch.expanded_idx_mapping, input_batch.idx_mapping_np
                )
                k_on = top_k is not None and bool((top_k < z.shape[1]).any())
                p_on = top_p is not None and bool((top_p < 1.0).any())
                if k_on or p_on:
                    # vLLM's warmup samples with top-k / top-p; real RL rollouts must not
                    # (verl: top_k=-1, top_p=1). vLLM's own log-probs are kept.
                    _warn_once("top-k / top-p active: log-probs are vLLM's, not RL-Kernel's")
                    return out
                temps = self.sampling_states.temperature.gpu[input_batch.expanded_idx_mapping]
            ids = out.sampled_token_ids[:, 0].long()
            lt.logprobs[:, 0] = _rlk_logprobs(z, ids, temps)
            return out

    return RlkSamplerV2


class RlkSampler(Sampler):
    """vLLM's (V1 model runner) sampler; the sampled token's reported log-prob is RL-Kernel's.

    ``processed_logprobs`` (verl's mode): ``log softmax(z / T)`` of the sampled token, with
    ``T`` vLLM's per-request temperature (1 for greedy requests, as vLLM does), computed as
    ``logprob_from_logits_cuda(z, token, T)`` per distinct ``T``: the same bytes as the
    actor's ``Dsv4Model.selected_logprob(hidden, token, T)``. ``raw_logprobs``: ``T = 1``.
    The top-k alternatives (if requested) stay vLLM's. Penalties, top-k / top-p and token
    masks change vLLM's processed distribution; they are refused in that mode."""

    def forward(
        self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None
    ):
        md = sampling_metadata
        mode = logprobs_mode_override or self.logprobs_mode
        want = md.max_num_logprobs is not None and mode in ("processed_logprobs", "raw_logprobs")
        z = logits.float().clone() if want else None
        out = super().forward(logits, md, predict_bonus_token, logprobs_mode_override)
        lt = out.logprobs_tensors
        if not want or lt is None or lt.logprobs.dim() != 2 or lt.logprobs.shape[0] == 0:
            return out
        ids = out.sampled_token_ids[:, 0].long()
        if mode == "processed_logprobs" and md.temperature is not None:
            if (
                not md.no_penalties
                or md.top_k is not None
                or md.top_p is not None
                or (md.allowed_token_ids_mask is not None or md.bad_words_token_ids)
            ):
                raise NotImplementedError(
                    "RL-Kernel DSv4 log-probs: penalties, top-k / top-p and token masks are "
                    "not supported with processed_logprobs"
                )
            logp = _rlk_logprobs(z, ids, md.temperature)
        else:
            logp = _rlk_logprobs(z, ids, None)
        lt.logprobs[:, 0] = logp
        return out


def add_lora(module: nn.Module, slices: list[tuple[str, int, int]]) -> None:
    """Give a vLLM linear unmerged LoRA adapters on column slices of its output.

    ``slices``: ``(target, start, end)``; the adapter for ``target`` (``module.rlk_adapters``)
    adds its delta to output columns ``[start, end)``, computed from the layer's input
    (``lora.lora_apply``). The module keeps its parameter names (weight loading and
    vLLM's post-load processing see the same module); only its ``forward`` is wrapped."""
    base = module.forward
    module.rlk_adapters = {}

    def forward(x, *args, **kwargs):
        out = base(x, *args, **kwargs)
        if not module.rlk_adapters:
            return out
        from rl_engine.models.deepseek_v4.lora import lora_apply

        y = out[0] if isinstance(out, tuple) else out
        parts, last = [], 0
        for target, lo, hi in slices:
            ad = module.rlk_adapters.get(target)
            if ad is None:
                continue
            parts += [y[:, last:lo], lora_apply(y[:, lo:hi], x, ad)]
            last = hi
        if last == 0:
            return out
        parts.append(y[:, last:])
        y = torch.cat([p for p in parts if p.shape[1]], dim=1)
        return (y, *out[1:]) if isinstance(out, tuple) else y

    module.forward = forward


def model_dir(name: str) -> pathlib.Path:
    """The local checkpoint directory of a vLLM ``model`` argument."""
    path = pathlib.Path(name)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(name, local_files_only=True))


class _PlainRope(nn.Module):
    """Holds the ``cos_sin_cache`` the DSv4 attention ops read (window-only layers)."""

    def __init__(self, cache: torch.Tensor):
        super().__init__()
        self.register_buffer("cos_sin_cache", cache, persistent=False)


def _lora_list(adapters: list[tuple[dict, MoeLora | None]]) -> list:
    """Every ``LoraAdapter`` of a per-layer adapter list, in a fixed order."""
    out = []
    for attn, moe in adapters:
        out += [attn[k] for k in sorted(attn)]
        out += [ad for _, ad in moe.adapters()] if moe is not None else []
    return out


def _same_lora_layout(
    a: list[tuple[dict, MoeLora | None]], b: list[tuple[dict, MoeLora | None]]
) -> bool:
    """Same layers, targets, shapes and alpha."""

    def names(adapters):
        return [
            (sorted(attn), [n for n, _ in moe.adapters()] if moe is not None else None)
            for attn, moe in adapters
        ]

    if len(a) != len(b) or names(a) != names(b):
        return False
    return all(
        x.a.shape == y.a.shape and x.b.shape == y.b.shape and float(x.alpha) == float(y.alpha)
        for x, y in zip(_lora_list(a), _lora_list(b))
    )


class RlkMoE(nn.Module):
    """Stands in for vLLM's ``DeepseekV4MoE``: ``Dsv4MoE`` is attached after load."""

    def __init__(self, vllm_config, prefix: str = "", use_sequence_parallel: bool = False):
        super().__init__()
        if use_sequence_parallel:
            raise NotImplementedError("RL-Kernel DSv4 rollout: sequence parallel MoE")
        self.layer = vm.extract_layer_index(prefix)
        self.hash_routing = self.layer < vllm_config.model_config.hf_config.num_hash_layers
        self.moe = None  # rl_engine.models.deepseek_v4.moe.Dsv4MoE
        self.use_mega_moe = False
        # Largest forward (tokens per data-parallel rank) for the static EP exchange.
        limit: str | int | None = os.environ.get(ENV_EP_STATIC)
        if limit is None:
            eager = vllm_config.model_config.enforce_eager
            limit = 0 if eager else vllm_config.compilation_config.max_cudagraph_capture_size
        self.ep_static_limit = int(limit or 0)

    def _ep_static_tokens(self) -> int | None:
        """The most tokens any data-parallel rank has in this forward (vLLM's DP metadata: a
        host tensor every rank holds the same), if within the static exchange's limit."""
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if self.ep_static_limit <= 0 or not is_forward_context_available():
            return None
        dp = get_forward_context().dp_metadata
        if dp is None:
            return None
        tokens = int(dp.num_tokens_across_dp_cpu.max())
        return tokens if tokens <= self.ep_static_limit else None

    def finalize_mega_moe_weights(self) -> None:
        pass

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.moe is None:
            raise RuntimeError(f"layer {self.layer}: the RL-Kernel MoE is not loaded")
        if self.moe.ep_group is not None:
            self.moe.ep_static_tokens = self._ep_static_tokens()
        return self.moe.forward(x, None, input_ids.long() if self.hash_routing else None)


@contextlib.contextmanager
def _rlk_moe_blocks():
    """Build :class:`RlkMoE` where ``DeepseekV4DecoderLayer`` builds ``DeepseekV4MoE``."""
    saved = vm.DeepseekV4MoE
    vm.DeepseekV4MoE = RlkMoE
    try:
        yield
    finally:
        vm.DeepseekV4MoE = saved


def _rlk_o_proj(attn, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``DeepseekV4Attention._o_proj`` on RL-Kernel's output projection (and ``wo_b`` LoRA)."""
    if attn.rlk_o_proj is None:
        raise RuntimeError("the RL-Kernel output projection is not loaded")
    if not getattr(attn, "rlk_released", False):  # vLLM's copies (never loaded) are unused
        for mod in (attn.wo_a, attn.wo_b):
            for prm in mod.parameters(recurse=False):
                prm.data = prm.data.new_empty(0)
        attn.rlk_released = True
    y = attn.rlk_o_proj(o, positions, attn.rotary_emb.cos_sin_cache, lora=attn.rlk_wo_b_lora)
    rec = getattr(attn, "rlk_record", None)
    if rec is not None:
        rec["o"], rec["y"] = o.clone(), y.clone()
    return y


def _rlk_qnorm_rope_insert(attn, q, kv, positions, attn_metadata):
    """``DeepseekV4Attention._fused_qnorm_rope_kv_insert`` on RL-Kernel's kernel."""
    if not isinstance(attn_metadata, dict):  # profile run: vLLM pads Q, no cache write
        return attn.rlk_vllm_insert(q, kv, positions, attn_metadata)
    from rl_engine.kernels.ops.cuda.attention import dsv4_attention as core

    md = attn_metadata.get(attn.swa_cache_layer.prefix)
    cache = attn.swa_cache_layer.kv_cache
    if cache.dtype != torch.uint8:
        raise NotImplementedError("RL-Kernel DSv4 rollout needs the fp8_ds_mla KV cache")
    return core.q_norm_rope_kv_insert(
        q,
        kv,
        positions,
        attn.rotary_emb.cos_sin_cache,
        cache.view(cache.shape[0], -1),
        md.block_size,
        md.slot_mapping,
        attn.padded_heads,
        attn.eps,
    )


@contextlib.contextmanager
def _float32_default():
    """vLLM loads weights under its model dtype as the default; RL-Kernel code assumes FP32."""
    saved = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(saved)


def _record_outputs(attn, rec: dict) -> None:
    """Debug: record ``fused_wqa_wkv`` and ``wq_b`` outputs into ``rec`` (hooks set once)."""
    if getattr(attn, "rlk_hooks", False):
        return

    def hook(name):
        def fn(_mod, _inp, out):
            r = getattr(attn, "rlk_record", None)
            if r is not None:
                r[name] = (out[0] if isinstance(out, tuple) else out).clone()

        return fn

    attn.fused_wqa_wkv.register_forward_hook(hook("qr_kv"))
    attn.wq_b.register_forward_hook(hook("q"))
    attn.rlk_hooks = True


class RlkDecoderLayer(vm.DeepseekV4DecoderLayer):
    """``DeepseekV4DecoderLayer`` on RL-Kernel's mHC (instances are re-classed)."""

    def forward(self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None):
        from rl_engine.kernels.ops.cuda.mhc import mhc_post_cuda, mhc_pre_cuda

        streams = x
        h, post, comb = mhc_pre_cuda(
            streams, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.attn_norm.weight
        )[:3]
        a = self.attn(positions, h, None)
        streams = mhc_post_cuda(a, streams, post, comb)
        h2, post, comb = mhc_pre_cuda(
            streams, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self.ffn_norm.weight
        )[:3]
        m = self.ffn(h2, input_ids)
        streams = mhc_post_cuda(m, streams, post, comb)
        detail = getattr(self, "rlk_detail", None)
        if detail is not None:
            detail.append([t.clone() for t in (h, a, h2, m)])
        return streams, None, None, None


class RlkDeepseekV4Model(vm.DeepseekV4Model):
    def __init__(self, *, vllm_config, prefix: str = ""):
        with _rlk_moe_blocks():
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        pc = vllm_config.parallel_config
        problems = []
        if self.use_sequence_parallel:
            problems.append("sequence parallel")
        if self.use_mega_moe:
            problems.append("moe_backend deep_gemm_mega_moe")
        if pc.tensor_parallel_size != 1:
            problems.append(f"tensor_parallel_size={pc.tensor_parallel_size}")
        if pc.pipeline_parallel_size != 1:
            problems.append(f"pipeline_parallel_size={pc.pipeline_parallel_size}")
        if pc.enable_eplb:
            problems.append("EPLB")
        if self._mtp_hidden_buffer is not None:
            problems.append("MTP / speculative decoding")
        self.rlk_bi = batch_invariant()
        if not self.rlk_bi:
            if vm.envs.VLLM_BATCH_INVARIANT:
                problems.append(f"{ENV_NON_BI}=1 with VLLM_BATCH_INVARIANT=1 (set it to 0)")
        else:
            if vm.envs.VLLM_DS4_DECODE_KERNEL.lower() != "sparse":
                problems.append("VLLM_DS4_DECODE_KERNEL != sparse")
            if not vm.envs.VLLM_BATCH_INVARIANT:
                problems.append(
                    "VLLM_BATCH_INVARIANT=0 (the attention path must be the BI one; "
                    f"{ENV_NON_BI}=1 is the non-invariant ablation)"
                )
        if "dummy" in str(vllm_config.load_config.load_format):
            # The base weights (and RL-Kernel's MoE / output projection) load from the
            # checkpoint once; a trainer syncs only LoRA adapters (vllm_worker.py).
            problems.append("load_format=dummy (load the checkpoint: load_format=auto)")
        if problems:
            raise NotImplementedError("RL-Kernel DSv4 rollout: " + ", ".join(problems))

        from rl_engine.models.deepseek_v4.config import Dsv4Config
        from rl_engine.models.deepseek_v4.rope import rope_cos_sin

        self.rlk_dir = model_dir(vllm_config.model_config.model)
        cfg = Dsv4Config.from_pretrained(self.rlk_dir)
        if cfg.num_hidden_layers != self.config.num_hidden_layers:  # hf_overrides
            cfg = dataclasses.replace(cfg, num_hidden_layers=self.config.num_hidden_layers)
        cfg.check_supported()
        self.rlk_config = cfg
        self.rlk_max_len = vllm_config.model_config.max_model_len
        plain = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.__class__ = RlkDecoderLayer
            attn = layer.attn
            q, d = attn.q_lora_rank, attn.head_dim
            add_lora(attn.fused_wqa_wkv, [("wq_a", 0, q), ("wkv", q, q + d)])
            add_lora(attn.wq_b, [("wq_b", 0, attn.n_local_heads * d)])
            attn.rlk_o_proj = None
            attn.rlk_wo_b_lora = None
            attn._o_proj = functools.partial(_rlk_o_proj, attn)
            attn.rlk_vllm_insert = attn._fused_qnorm_rope_kv_insert
            if self.rlk_bi:
                # One stream: vLLM overlaps the compressor / indexer on aux streams with the
                # query path, and RL-Kernel's ops there allocate on the current stream only.
                # (The ablation keeps vLLM's ops there, and its streams.)
                attn.aux_stream_list = None
                if attn.indexer is not None:
                    attn.indexer.aux_stream = None
                attn._fused_qnorm_rope_kv_insert = functools.partial(_rlk_qnorm_rope_insert, attn)
            if attn.compress_ratio == 1:
                old = attn.rotary_emb.cos_sin_cache
                if old.dtype != torch.float32:
                    raise TypeError(f"vLLM RoPE cache is {old.dtype}; expected float32")
                if plain is None:
                    cache = rope_cos_sin(self.rlk_config, 1, self.rlk_max_len, old.device)
                    plain = _PlainRope(cache)
                attn.rotary_emb = plain
                attn.indexer_rotary_emb = plain
        self.rlk_dump_step = 0

    def finalize_mhc_broadcast_weights(self) -> None:
        pass  # the first layer reads four streams; no broadcast weights

    def get_expert_mapping(self) -> list:
        return []  # the MoE blocks are not vLLM's (rlk_load_moe)

    def rlk_layers(self):
        return list(islice(self.layers, self.start_layer, self.end_layer))

    def rlk_load(self) -> None:
        """Load what RL-Kernel runs itself from the checkpoint: every layer's ``Dsv4MoE``
        (this EP rank's experts) and output projection. vLLM's ``wo_a`` / ``wo_b`` storage is
        released at the first forward (vLLM post-processes it after loading)."""
        from vllm.distributed import get_ep_group

        from rl_engine.models.deepseek_v4.loader import SafetensorsDir, _moe, load_output_projection

        ep = None
        if self.parallel_config.enable_expert_parallel and get_ep_group().world_size > 1:
            ep = get_ep_group().device_group
        src = SafetensorsDir(self.rlk_dir)
        dev = self.hc_head_fn.device
        for layer in self.rlk_layers():
            i = layer.ffn.layer
            layer.ffn.moe = _moe(src, self.rlk_config, i, dev, ep)
            layer.ffn.moe.check_finite = "defer"
            attn = layer.attn
            attn.rlk_o_proj = load_output_projection(src, self.rlk_config, i, dev)

    def rlk_set_lora(self, adapters: list[tuple[dict, MoeLora | None]]) -> None:
        """Install adapters, one ``(attention dict, MoeLora | None)`` per local layer.

        Adapters with the installed ones' targets, shapes and alpha are copied into them, so
        CUDA graphs captured with the installed ones read the new values. Once
        ``RL_KERNEL_DSV4_LORA`` has reserved adapters (before the capture), others are refused.
        """
        cur = getattr(self, "rlk_lora", None)
        if cur is not None and _same_lora_layout(cur, adapters):
            with torch.no_grad():
                for old, new in zip(_lora_list(cur), _lora_list(adapters)):
                    old.a.copy_(new.a)
                    old.b.copy_(new.b)
            return
        if getattr(self, "rlk_lora_reserved", False):
            raise ValueError(
                f"the adapters differ from the ones {ENV_LORA} reserved (targets, shapes or "
                "alpha), with which the CUDA graphs were captured"
            )
        self.rlk_lora = adapters
        for layer, (attn_lora, moe_lora) in zip(self.rlk_layers(), adapters):
            attn = layer.attn
            for mod, targets in ((attn.fused_wqa_wkv, ("wq_a", "wkv")), (attn.wq_b, ("wq_b",))):
                mod.rlk_adapters = {k: attn_lora[k] for k in targets if k in attn_lora}
            attn.rlk_wo_b_lora = attn_lora.get("wo_b")
            layer.ffn.moe.lora = moe_lora

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        from rl_engine.kernels.ops.cuda.mhc import hc_head_cuda

        h = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        streams = h.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        dump = os.environ.get(ENV_DUMP)
        if dump and not os.path.isdir(dump):
            dump = None  # set before the engine starts; on once the directory exists
        per_layer = [] if dump else None
        detail = [] if dump else None
        inner = [] if dump else None
        for layer in self.rlk_layers():
            layer.rlk_detail = detail
            if dump:
                rec = {}
                inner.append(rec)
                layer.attn.rlk_record = rec
                _record_outputs(layer.attn, rec)
            else:
                layer.attn.rlk_record = None
            streams = layer(streams, positions, input_ids)[0]
            if dump:
                per_layer.append(streams.clone())
        out = hc_head_cuda(
            streams, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm.weight
        )[0]
        if dump:
            self._rlk_dump(dump, input_ids, positions, per_layer, out, detail, inner)
        return out

    def _rlk_dump(self, root, input_ids, positions, per_layer, out, detail, inner) -> None:
        from vllm.forward_context import get_forward_context

        qsl = seq = None
        meta = get_forward_context().attn_metadata
        for m in meta.values() if isinstance(meta, dict) else [meta]:
            if (
                getattr(m, "query_start_loc_cpu", None) is not None
                and getattr(m, "seq_lens_cpu", None) is not None
            ):
                qsl, seq = m.query_start_loc_cpu.clone(), m.seq_lens_cpu.clone()
                break
        if qsl is None or _IN_DUMMY_RUN[0]:
            return  # a dummy forward (profiling, or an idle data-parallel rank): no requests
        from vllm.distributed import get_tensor_model_parallel_rank

        # The rank inside this replica: with data parallelism the global rank counts the
        # other replicas too, and each replica has its own dp<r>/ directory.
        rank = get_tensor_model_parallel_rank()
        path = pathlib.Path(root)
        if self.parallel_config.data_parallel_size > 1:
            path = path / f"dp{self.parallel_config.data_parallel_rank}"
        path = path / f"rank{rank}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_ids": input_ids.cpu(),
                "positions": positions.cpu(),
                "query_start_loc": qsl,
                "seq_lens": seq,
                "streams": [s.cpu() for s in per_layer],
                "hidden": out.cpu(),
                # per layer: attention input, attention output, MoE input, MoE output
                "detail": [[t.cpu() for t in d] for d in detail],
                # per layer: qr_kv, q (wq_b out), o (attention core), y (o-proj out)
                "inner": [{k: v.cpu() for k, v in r.items()} for r in inner],
            },
            path / f"step{self.rlk_dump_step:05d}.pt",
        )
        self.rlk_dump_step += 1


class RlkDeepseekV4ForCausalLM(vm.DeepseekV4ForCausalLM):
    model_cls = RlkDeepseekV4Model

    def set_moe_parameters(self) -> None:
        c = self.config
        self.num_expert_groups = getattr(c, "n_group", 1)
        self.moe_layers: list = []
        self.moe_mlp_layers: list = []
        self.num_moe_layers = 0
        self.num_logical_experts = self.num_physical_experts = c.n_routed_experts
        self.num_routed_experts = c.n_routed_experts
        self.num_local_physical_experts = c.n_routed_experts  # unused without EPLB
        self.num_shared_experts = c.n_shared_experts or 0
        self.num_redundant_experts = 0

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 ``[T, vocab]`` logits ``hidden @ head.T`` (DeepGEMM; ``dsv4-logprob-v1``)."""
        from rl_engine.kernels.ops.cuda.logprob_dsv4 import head_logits

        w = self.lm_head.weight[: self.config.vocab_size]
        return head_logits(hidden_states, w)

    def load_weights(self, weights):
        """vLLM loads everything but the MoE blocks; RL-Kernel then loads those from the
        checkpoint directory (this rank's experts) and installs the test adapters, if any."""
        skip = (".ffn.", ".attn.wo_a.", ".attn.wo_b.")
        n_layers = self.config.num_hidden_layers
        layer = re.compile(r"(?:^|\.)layers\.(\d+)\.")

        def keep(name):
            # A view of the first N layers (config edited, shards shared) also holds the
            # later layers' tensors, which vLLM's loader has no parameter for.
            m = layer.search(name)
            if m is not None and int(m.group(1)) >= n_layers:
                return False
            return not any(k in name for k in skip)

        loaded = super().load_weights((n, w) for n, w in weights if keep(n))
        with _float32_default():
            self.model.rlk_load()
            self._rlk_test_lora()
            self._rlk_reserve_lora()
        return loaded

    def _rlk_reserve_lora(self) -> None:
        """``RL_KERNEL_DSV4_LORA=rank,alpha``: install zero-``B`` adapters on every target
        (the ``rlkernel`` engine's default set) before vLLM captures its CUDA graphs; the
        synced adapters are then copied into them (:meth:`RlkDeepseekV4Model.rlk_set_lora`).
        A zero ``B`` adds an exact zero, so until the first sync the bytes are the base
        model's."""
        spec = os.environ.get(ENV_LORA)
        if not spec or os.environ.get(ENV_TEST_LORA):
            return
        from rl_engine.models.deepseek_v4.model import build_lora_adapters

        rank, alpha = spec.split(",")
        layers = [layer.ffn.layer for layer in self.model.rlk_layers()]
        dev = self.model.hc_head_fn.device
        cfg = self.model.rlk_config
        self.model.rlk_set_lora(
            build_lora_adapters(cfg, layers, int(rank), float(alpha), device=dev, zero_b=True)
        )
        self.model.rlk_lora_reserved = True

    def _rlk_test_lora(self) -> None:
        spec = os.environ.get(ENV_TEST_LORA)
        if spec:
            from rl_engine.models.deepseek_v4.model import build_lora_adapters

            rank, alpha, seed = spec.split(",")
            layers = [layer.ffn.layer for layer in self.model.rlk_layers()]
            self.model.rlk_set_lora(
                build_lora_adapters(
                    self.model.rlk_config,
                    layers,
                    int(rank),
                    float(alpha),
                    device=self.model.hc_head_fn.device,
                    seed=int(seed),
                    zero_b=False,
                )
            )
