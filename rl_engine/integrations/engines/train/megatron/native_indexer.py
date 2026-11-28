# SPDX-License-Identifier: Apache-2.0
"""Actor indexer in the actual production vLLM FP8 Q/K selection domain.

Original BF16/F32 weights remain registered and unchanged. The native inference
FP8 RoPE codec, C4 overlap/state writer, MQA logits and Top-k run independently
on actor operands. The prior eager BF16/Hadamard score was a different numerical
profile, observable only after more than512 compressed candidates. No full-
precision/dequantized-weight fallback or rollout score reuse is involved.
"""

import ast
import inspect
import textwrap
import types

import torch

from rl_engine.backends.rocm.attention.native_indexer import native_fp8_indexer


def packed_indexer_keys(kv, score, ape, norm, cache, epsilon):
    from types import SimpleNamespace

    from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
        compress_norm_rope_store_triton,
    )
    from vllm.models.deepseek_v4.common.ops.save_partial_states import _SAVE_PARTIAL_STATES_KERNEL
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import FP8_DTYPE
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

    t, batch, width = kv.shape
    if (
        width != 256
        or score.shape != kv.shape
        or kv.dtype != score.dtype
        or kv.dtype != torch.float32
    ):
        raise ValueError("original FP32 C4 indexer projections [T,B,256] required")
    if mqa_logits_use_mx_scales():
        raise ValueError("this gfx942 indexer profile requires native float32 key scales")
    count, state_block, key_block = t // 4, 4, 64
    if not count:
        raise ValueError("indexer key codec requires a complete original C4 group")
    state_pages, key_pages = (t + 3) // 4, (count + key_block - 1) // key_block
    device = kv.device
    state = torch.zeros((batch * state_pages, state_block, 512), device=device, dtype=torch.float32)
    positions = torch.arange(t, device=device, dtype=torch.int64).repeat(batch)
    owners = torch.arange(batch, device=device, dtype=torch.int32).repeat_interleave(t)
    slots = positions + owners.long() * state_pages * state_block
    table = torch.arange(batch * state_pages, device=device, dtype=torch.int32).reshape(
        batch, state_pages
    )
    packed = torch.zeros((batch * key_pages, key_block, 132), device=device, dtype=torch.uint8)
    key_slots = positions // 4 + owners.long() * key_pages * key_block
    _SAVE_PARTIAL_STATES_KERNEL(
        kv=kv.permute(1, 0, 2).reshape(-1, width).contiguous(),
        score=score.permute(1, 0, 2).reshape(-1, width).contiguous(),
        ape=ape,
        positions=positions,
        state_cache=state,
        slot_mapping=slots,
        block_size=state_block,
        state_width=width,
        compress_ratio=4,
        pdl_kwargs={},
    )
    compress_norm_rope_store_triton(
        state_cache=state,
        num_actual=t * batch,
        token_to_req_indices=owners,
        positions=positions,
        slot_mapping=slots,
        block_table=table,
        block_size=state_block,
        state_width=width,
        cos_sin_cache=cache,
        kv_cache=packed,
        k_cache_metadata=SimpleNamespace(slot_mapping=key_slots),
        pdl_kwargs={},
        head_dim=128,
        rope_head_dim=64,
        compress_ratio=4,
        overlap=True,
        use_fp4_cache=False,
        rms_norm_weight=norm,
        rms_norm_eps=epsilon,
        quant_block=128,
        token_stride=128,
        scale_dim=4,
    )
    # Preserve the native block-flat byte interpretation and scale convention.
    flat = packed.view(batch, key_pages, -1)
    keys = flat[:, :, : key_block * 128].contiguous().view(FP8_DTYPE)
    keys = keys.reshape(batch, key_pages * key_block, 128)[:, :count].contiguous()
    scales = flat[:, :, key_block * 128 :].contiguous().view(torch.float32)
    scales = scales.reshape(batch, key_pages * key_block, 1)[:, :count].contiguous()
    return keys, scales


def production_indexer_select(q_raw, kv, score, ape, norm, raw_head_weights, cache, epsilon):
    from rl_engine.backends.rocm.attention.deterministic_indexer_topk import (
        deterministic_topk_prefill,
    )
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import fused_indexer_q_rope_quant
    from rl_engine.backends.rocm.attention.indexer_paged_logits import paged_c4_logits_contiguous

    t, batch, heads, dim = q_raw.shape
    if dim != 128 or raw_head_weights.shape != (t, batch, heads) or q_raw.dtype != torch.bfloat16:
        raise ValueError("original BF16 indexer queries/head weights required")
    keys, scales = packed_indexer_keys(kv, score, ape, norm, cache, epsilon)
    starts = torch.zeros(t, dtype=torch.int32, device=q_raw.device)
    ends = torch.div(
        torch.arange(1, t + 1, device=q_raw.device, dtype=torch.int32), 4, rounding_mode="floor"
    )
    positions = torch.arange(t, device=q_raw.device, dtype=torch.int64)
    logits_rows, selections, q_operands, weight_operands = [], [], [], []
    for b in range(batch):
        q, weights = fused_indexer_q_rope_quant(
            positions,
            q_raw[:, b].contiguous(),
            cache,
            raw_head_weights[:, b].contiguous(),
            dim**-0.5,
            heads**-0.5,
            use_fp4=False,
        )
        # Same paged Triton kernel as rollout decode and (wrapped) rollout prefill; starts are 0.
        logits = paged_c4_logits_contiguous(q, weights, keys[b], scales[b], starts, ends)
        indices = torch.full((t, 512), -1, device=q.device, dtype=torch.int32)
        # Same Top-512 rule as the rollout: stable, lower column wins ties, relative ascending.
        deterministic_topk_prefill(logits, starts, ends, indices, 512)
        logits_rows.append(logits)
        selections.append(indices)
        q_operands.append(q)
        weight_operands.append(weights)
    return (
        torch.stack(logits_rows),
        torch.stack(selections),
        dict(
            q=q_operands,
            weights=weight_operands,
            keys=keys,
            key_scales=scales,
        ),
    )


class _OnlyPositiveAux(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_If(self, node):
        if ast.unparse(node.test) == "self.training and torch.is_grad_enabled()" and any(
            isinstance(child, ast.Name) and child.id == "FusedDSAIndexerLoss"
            for statement in node.body
            for child in ast.walk(statement)
        ):
            node.test = ast.BoolOp(
                ast.And(),
                [
                    node.test,
                    ast.parse("(self.config.dsa_indexer_loss_coeff or 0.0) > 0", mode="eval").body,
                ],
            )
            self.changed += 1
        return self.generic_visit(node)


def bind_production_indexer(core, preprocess):
    indexer = core.indexer
    if indexer is None:
        return
    if (
        core.compress_ratio != 4
        or indexer.index_topk != 512
        or (core.config.dsa_indexer_loss_coeff or 0) != 0
    ):
        raise ValueError(
            "production indexer requires C4/Top512 and zero auxiliary-loss coefficient"
        )
    original = core.forward.__func__
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    transform = _OnlyPositiveAux()
    tree = transform.visit(tree)
    if transform.changed != 1:
        raise ValueError("native CSA zero-aux training branch ABI changed")
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    core.forward = types.MethodType(namespace[original.__name__], core)

    def forward(self, x, qr, mask=None):
        t, batch, _ = x.shape
        count = t // 4
        if not count:
            raise ValueError("original C4 indexer requires at least four positions")
        valid = torch.arange(1, t + 1, device=x.device)[:, None] // 4
        expected_mask = torch.where(
            torch.arange(count, device=x.device)[None, :] >= valid, -float("inf"), 0.0
        )
        if (
            mask is None
            or mask.shape != (batch, t, count)
            or not torch.equal(mask, expected_mask[None].expand(batch, -1, -1))
        ):
            raise ValueError("actor indexer requires the exact original full causal mask")
        if count <= 512:
            indices = torch.arange(count, device=x.device).expand(batch, t, count)
            indices = torch.where(indices < valid[None], indices, -1).to(torch.int32)
            return torch.empty((batch, t, 0), dtype=torch.float32, device=x.device), indices
        with torch.no_grad():
            q = self.linear_wq_b(qr)[0].reshape(t, batch, self.index_n_heads, 128)
            kv = self.compressor.linear_wkv(x)[0]
            score = self.compressor.linear_wgate(x)[0]
            weights = self.linear_weights_proj(x)[0]
            logits, indices, _ = production_indexer_select(
                q,
                kv,
                score,
                self.compressor.ape,
                self.compressor.norm.weight,
                weights,
                preprocess.cos_sin_cache,
                self.config.layernorm_epsilon,
            )
        if not getattr(self, "_rlk_production_indexer_executed", False):
            print(
                "RLK_PRODUCTION_FP8_INDEXER_EXECUTED",
                __file__,
                tuple(x.shape),
                count,
                "original weights; native FP8 Q/K/C4/MQA/Top512; aux coefficient0",
                flush=True,
            )
            self._rlk_production_indexer_executed = True
        return logits, indices

    indexer.forward = types.MethodType(forward, indexer)
    core._rlk_indexer_numeric_profile = "native-vllm-c4-fp8-qk-rope-mqa-top512-zero-aux-v1"


# Compatibility entry point for the native auxiliary-loss-preserving actor profile.
def _require_causal_mask(mask, t, batch):
    n = t // 4
    if not isinstance(mask, torch.Tensor) or mask.shape != (batch, t, n):
        raise ValueError("native FP8 indexer requires the actual full-prefix causal mask")
    future = (
        torch.arange(n, device=mask.device)[None, :]
        >= ((torch.arange(t, device=mask.device) + 1) // 4)[:, None]
    )
    if not torch.equal(torch.isneginf(mask), future[None].expand_as(mask)) or not bool(
        ((mask == 0) | torch.isneginf(mask)).all()
    ):
        raise ValueError("native FP8 indexer received a noncanonical compressed causal mask")


def bind_native_fp8_indexer(core, preprocess):
    indexer = core.indexer
    if indexer is None:
        return core
    config = core.config
    if (getattr(config, "dsa_indexer_loss_coeff", 0) or 0) != 0:
        raise ValueError(
            "native FP8 selection requires configured indexer auxiliary coefficient zero"
        )
    if (
        core.compress_ratio != 4
        or indexer.index_topk != 512
        or indexer.index_n_heads != 64
        or indexer.index_head_dim != 128
        or getattr(config, "tensor_model_parallel_size", 1) != 1
        or getattr(config, "context_parallel_size", 1) != 1
    ):
        raise ValueError("native FP8 indexer supports ordinary TP1/CP1 H64/D128/C4 Top512 only")
    if getattr(indexer, "_rlk_native_fp8_indexer", False):
        raise ValueError("native FP8 indexer is already bound")
    original_before = indexer.forward_before_topk
    original_forward = indexer.forward
    pending = None
    last = None
    modules = {
        "q": indexer.linear_wq_b,
        "weights": indexer.linear_weights_proj,
        "kv": indexer.compressor.linear_wkv,
        "score": indexer.compressor.linear_wgate,
    }
    handles = []
    for name, module in modules.items():

        def observe(_module, _args, output, _name=name):
            if pending is not None:
                if _name in pending or not isinstance(output, tuple) or len(output) != 2:
                    raise ValueError("unexpected native indexer projection execution ABI")
                pending[_name] = output[0]

        handles.append(module.register_forward_hook(observe))

    def before(self, x, qr):
        nonlocal pending, last
        if pending is not None or last is not None:
            raise RuntimeError("native FP8 indexer has an unconsumed forward")
        if x.shape[0] // 4 <= 512:
            return original_before(x, qr)
        pending = {}
        try:
            result = original_before(x, qr)
            values = pending
            if set(values) != set(modules):
                raise RuntimeError("missing original indexer projection for native FP8 selection")
            scores, indices = native_fp8_indexer(
                values["q"],
                values["weights"],
                values["kv"],
                values["score"],
                self.compressor.ape,
                self.compressor.norm.weight,
                preprocess.cos_sin_cache,
                config.layernorm_epsilon,
            )
            last = (result[0], result[1], scores, indices)
            if not getattr(self, "_rlk_fp8_executed", False):
                print(
                    "RLK_ACTOR_NATIVE_FP8_INDEXER_EXECUTED",
                    __file__,
                    "original projections; native FP8 Q/K writers + MQA + Top512; "
                    "auxiliary coefficient=0",
                    tuple(values["q"].shape),
                    flush=True,
                )
                self._rlk_fp8_executed = True
            return result
        finally:
            pending = None

    def forward(self, x, qr, mask=None):
        nonlocal last
        if x.shape[0] // 4 <= 512:
            return original_forward(x, qr, mask=mask)
        _require_causal_mask(mask, x.shape[0], x.shape[1])
        self.forward_before_topk(x, qr)
        _, _, scores, indices = last
        last = None
        return scores, indices

    original_core = core.forward.__func__
    if "FusedDSAIndexerLoss" not in original_core.__code__.co_names:
        raise ValueError("unsupported native CSA auxiliary-loss boundary")
    native_loss = original_core.__globals__["FusedDSAIndexerLoss"]

    class SelectionWithNativeAuxiliary:
        @staticmethod
        def apply(q, weights, k, *args, **kwargs):
            nonlocal last
            if q.shape[0] // 4 > 512:
                if kwargs or len(args) < 6 or args[4] != 0:
                    raise ValueError(
                        "native FP8 selection requires the original zero-coefficient auxiliary ABI"
                    )
                _require_causal_mask(args[5], q.shape[0], q.shape[1])
            original_indices, loss = native_loss.apply(q, weights, k, *args, **kwargs)
            if q.shape[0] // 4 <= 512:
                return original_indices, loss
            if last is None or q is not last[0] or k is not last[1]:
                raise RuntimeError(
                    "native FP8 training selection is stale or belongs to another batch"
                )
            indices = last[3]
            if indices.shape != original_indices.shape:
                raise ValueError("native FP8 selected IDs do not match the actual training layout")
            last = None
            return indices, loss

    namespace = dict(original_core.__globals__, FusedDSAIndexerLoss=SelectionWithNativeAuxiliary)
    bound = types.FunctionType(
        original_core.__code__,
        namespace,
        original_core.__name__,
        original_core.__defaults__,
        original_core.__closure__,
    )
    bound.__kwdefaults__ = original_core.__kwdefaults__
    core.forward = types.MethodType(bound, core)
    indexer.forward_before_topk = types.MethodType(before, indexer)
    indexer.forward = types.MethodType(forward, indexer)
    indexer._rlk_native_fp8_indexer = True
    indexer._rlk_native_fp8_handles = handles
    return core
