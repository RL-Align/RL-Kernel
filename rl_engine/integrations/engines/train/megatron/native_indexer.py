# SPDX-License-Identifier: Apache-2.0
"""Explicit native FP8 C4 selection on the actual Megatron CSA indexer.

Keep real projection execution and the original auxiliary-loss autograd call.
When its configured coefficient is zero, replace its nondifferentiable selected
IDs with the same native FP8 contract used by the loaded rollout. Positive
auxiliary-loss coefficients require a separately validated FP8 training bridge.
"""

import types

import torch

from rl_engine.backends.rocm.attention.native_indexer import native_fp8_indexer


def _require_causal_mask(mask, t, batch):
    n = t // 4
    if not isinstance(mask, torch.Tensor) or mask.shape != (batch, t, n):
        raise ValueError(
            "native FP8 indexer requires the actual full-prefix causal mask"
        )
    future = (
        torch.arange(n, device=mask.device)[None, :]
        >= ((torch.arange(t, device=mask.device) + 1) // 4)[:, None]
    )
    if not torch.equal(torch.isneginf(mask), future[None].expand_as(mask)) or not bool(
        ((mask == 0) | torch.isneginf(mask)).all()
    ):
        raise ValueError(
            "native FP8 indexer received a noncanonical compressed causal mask"
        )


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
        raise ValueError(
            "native FP8 indexer supports ordinary TP1/CP1 H64/D128/C4 Top512 only"
        )
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
                if (
                    _name in pending
                    or not isinstance(output, tuple)
                    or len(output) != 2
                ):
                    raise ValueError(
                        "unexpected native indexer projection execution ABI"
                    )
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
                raise RuntimeError(
                    "missing original indexer projection for native FP8 selection"
                )
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
                    "original projections; native FP8 Q/K writers + MQA + Top512; auxiliary coefficient=0",
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
                raise ValueError(
                    "native FP8 selected IDs do not match the actual training layout"
                )
            last = None
            return indices, loss

    namespace = dict(
        original_core.__globals__, FusedDSAIndexerLoss=SelectionWithNativeAuxiliary
    )
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
