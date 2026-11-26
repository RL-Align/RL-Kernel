# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 ``sglang_runtime_adapter_h3``: strict H3 operators inside SGLang's H3 DiT.

SGLang (``sglang.multimodal_gen``, checked at 0.5.21) runs MiniMax-H3 as ``MiniMaxH3DiTModel``.
Its block and token-refiner projections have the paths, fused layouts and ``(output, bias)``
return of vLLM-Omni's, and SGLang's loader maps the Diffusers checkpoint the way
``models/minimax_h3/checkpoint_mapper`` does:

* ``attn.qkv_proj``: ``[Q; K; V]`` rows (``MergedColumnParallelLinear``, each of Q, K, V sharded
  by heads under TP);
* ``mlp.fc1``: ``[gate; up]`` rows (``MergedColumnParallelLinear``);
* ``attn.out_proj`` / ``mlp.fc2``: ``RowParallelLinear``, all-reduced by SGLang.

``install_h3_sglang_rollout`` binds the operators of the strict training graph:

* the projections, as ``install_h3_rollout`` (``vllm/h3_omni.py``) does: ``qkv_proj`` on
  ``h3-projection-gemm-v1``, ``fc1`` on the gate/up row's fixed FP32-tile GEMM
  (``fixed_fp32_gemm``), ``out_proj`` / ``fc2`` on the canonical row-parallel reduction of
  ``h3-parallel-v2`` (``h3_tp.run_tp_block_rows``);
* in every transformer block (``MiniMaxH3DiTBlock``), the AdaLN path: ``adaln_proj.linear`` on the
  deterministic GEMV of ``adaln_projection_3mod`` (on its TP column shard, which is that row's TP
  layout: SGLang's all-gather of the shards is a copy), ``norm1`` / ``norm2`` with their
  modulation on ``h3_rmsnorm`` (``forward_modulated``), and both gated residuals on
  ``adaln_gate_residual``. SGLang applies the norm and the modulation as two steps; the bound
  block applies them as the training block does, so the block's ``forward`` is replaced, and
  ``attn`` / ``mlp`` are still called as SGLang's modules;
* in the final layer (``MiniMaxH3FinalLayer``), ``final_adaln_out``: the same GEMV for
  ``adaln_proj.linear`` and the modulated ``h3_rmsnorm`` for ``norm``; the FP32 output heads
  ``video_out`` / ``audio_out`` stay SGLang's;
* the plain norms of the token refiner (``blocks.N.norm1`` / ``norm2`` and ``final_norm``) on
  ``h3_rmsnorm``.

SGLang's ``adaln_input`` is ``silu(t_emb).to(bfloat16)``, the activation that
``adaln_projection_3mod`` and ``final_adaln_out`` compute from ``temb`` before their GEMV.

Before any forward changes, it refuses what SGLang can put in the way of a bound forward:

* an SGLang version other than those in ``TESTED_SGLANG_VERSIONS``, also without ``strict``: the
  bound block and final-layer forwards replace SGLang's own and call its ``adaln_proj``, ``attn``,
  ``mlp`` and ``_project`` with the signatures of the versions they were validated on. A new
  version is added after ``tools/validation/models/h3_sglang_rollout_report.py`` passes on it;
* a compiled module (``torch.compile`` / ``nn.Module.compile``): the compiled graph keeps
  running the forward it traced;
* a LoRA wrapper (``BaseLayerWithLoRA``) at a bound linear: it adds the LoRA output in its own
  forward, which a bound forward would drop;
* a quantized linear: its weight is not the BF16 checkpoint tensor, and mxfp8 hands the
  projection pre-quantized input (a block with quantized projections takes SGLang's mxfp8
  modulation path);
* a module sharded over another TP size than ``tp``, a column-parallel module that gathers its
  output, or a row-parallel module that does not expect sharded input or reduce its result;
* the AdaLN cache (``adaln_proj is None``): its precomputed tables replace the projection;
* FP32 AdaLN weights (curve-grid checkpoints) and the distilled PDD output heads: these are not
  the released dense H3;
* the VDN-H3 hybrid attention and the compression gate, for the same reason.

``check_sglang_server_args`` refuses the server settings that change what the DiT computes or
bypass a bound forward: ``enable_torch_compile`` / ``regional_compile``,
``enable_breakable_cuda_graph``, ``cache_dit_config``, quantization and an unmerged LoRA, and the
ones that move or shard the DiT's weights: CPU or layerwise offload of the transformer and FSDP
inference.
``check_sglang_sampling_params`` refuses the per-request step skipping of Spectrum and TeaCache.

Under TP, ``MiniMaxH3DiTModel`` batches the AdaLN projections of all blocks and hands each block
its precomputed table; a bound block refuses precomputed tables. The adapter turns that batching
off on the model, so every block runs its bound projection. The bytes are the same: the batched
path calls the same bound ``adaln_proj.linear`` and all-gathers the same column shards, once for
all blocks instead of once per block.

Readback: the handle counts the calls of every bound forward (``handle.calls``) and names the
training kernel each one runs (``handle.implementations``). ``handle.readback(forwards)`` lists
every bound module that did not run exactly once per DiT forward (a forward SGLang bypassed, or a
step it skipped), and ``handle.assert_ran(forwards)`` raises on any.

Attention, rotary, QK norm and SwiGLU stay SGLang's and are reported ``external``, as are the
FP32 boundary projections (patch embeddings, time embedder, output heads). ``handle.remove()``
restores every forward.
"""

from __future__ import annotations

import importlib.metadata
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from rl_engine.integrations.engines.rollout.vllm.h3_omni import (
    _ABSENT,
    H3RolloutHandle,
    install_h3_rollout,
)

ENGINE = "sglang"
# The SGLang releases the bound forwards were validated on (the report tool, every bound output).
TESTED_SGLANG_VERSIONS = ("0.5.21",)
EXTERNAL = [
    "attention",
    "rotary",
    "qk_norm",
    "swiglu",
    "patch_embeddings",
    "time_embedder",
    "output_heads",
]
# What each bound forward runs, by the kind of module (``handle.implementations``).
IMPLEMENTATIONS = {
    "column": "h3-projection-gemm-v1 (Triton)",
    "column_fixed": "h3-parallel-v2 column-parallel, the gate/up row's fixed FP32-tile GEMM "
    "(fixed_fp32_gemm, Triton)",
    "row": "h3-parallel-v2 canonical row-parallel reduction",
    "adaln": "adaln_projection_3mod / final_adaln_out GEMV: rl_engine._C.h3_det_linear_forward",
    "block": "h3_rmsnorm (modulated) + adaln_gate_residual: rl_engine._C.h3_rmsnorm_forward, "
    "rl_engine._C.h3_gate_residual_forward",
    "final": "final_adaln_out norm: rl_engine._C.h3_rmsnorm_forward (modulated)",
    "norm": "h3_rmsnorm: rl_engine._C.h3_rmsnorm_forward",
}
_REFINER_NORM = re.compile(r"(?:^|\.)token_refiner\.(?:blocks\.\d+\.norm[12]|final_norm)$")
_ADALN_MODULES = {"MiniMaxH3DiTBlock": "block", "MiniMaxH3FinalLayer": "final"}


def _sglang_version() -> str | None:
    try:
        return importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_refusal() -> str | None:
    version = _sglang_version()
    if version in TESTED_SGLANG_VERSIONS:
        return None
    found = "SGLang is not installed" if version is None else f"SGLang {version} is installed"
    return (
        f"{found}, but the bound forwards are validated only on SGLang "
        f"{', '.join(TESTED_SGLANG_VERSIONS)}: they replace SGLang's block and final-layer "
        "forwards, so a changed SGLang module would be bypassed without an error. Run "
        "tools/validation/models/h3_sglang_rollout_report.py on the new version and add it to "
        "TESTED_SGLANG_VERSIONS"
    )


@dataclass
class H3SGLangRolloutHandle(H3RolloutHandle):
    """``H3RolloutHandle`` with SGLang's model-level override and the execution readback."""

    # Calls of each bound forward since install (or ``calls.clear()``), by module path.
    calls: Counter = field(default_factory=Counter)
    # Model forwards seen by the SGLang plugin's forward guard (``plugin.py``).
    forwards: int = 0
    implementations: dict[str, str] = field(default_factory=dict)
    overrides: list[str] = field(default_factory=list)
    _attributes: list = field(default_factory=list)  # (object, attribute, its own value or _ABSENT)

    def remove(self) -> None:
        for obj, name, previous in reversed(self._attributes):
            if previous is _ABSENT:
                obj.__dict__.pop(name, None)
            else:
                setattr(obj, name, previous)
        self._attributes.clear()
        super().remove()

    def readback(self, forwards: int) -> list[str]:
        """Every bound module that did not run exactly ``forwards`` times (empty: all did)."""

        return [
            f"{name}: ran {self.calls[name]} times in {forwards} forwards"
            for name in self.patched
            if self.calls[name] != forwards
        ]

    def assert_ran(self, forwards: int) -> None:
        problems = self.readback(forwards)
        if problems:
            raise RuntimeError("strict H3 rollout on SGLang: " + "; ".join(problems))

    def report(self) -> dict:
        out = super().report()
        out["calls"] = {name: self.calls[name] for name in self.patched}
        out["forwards"] = self.forwards
        out["implementations"] = dict(self.implementations)
        out["overrides"] = list(self.overrides)
        return out


def _kind(name: str, module: torch.nn.Module) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "qkv_proj":
        return "column"
    if leaf == "fc1":
        return "column_fixed"
    if leaf in ("out_proj", "fc2"):
        return "row"
    if name.endswith("adaln_proj.linear"):
        return "adaln"
    return _ADALN_MODULES.get(type(module).__name__, "norm")


def _counted(forward: Callable[..., Any], calls: Counter, name: str) -> Callable[..., Any]:
    def counted(*args, **kwargs):
        calls[name] += 1
        return forward(*args, **kwargs)

    return counted


def _never() -> bool:
    return False


def _compiled(module: torch.nn.Module) -> bool:
    if getattr(module, "_compiled_call_impl", None) is not None:  # nn.Module.compile()
        return True
    return type(module).__name__ == "OptimizedModule"  # torch.compile(module)


def _model_refusals(model: torch.nn.Module) -> tuple[list[str], list[str]]:
    """``(compiled, variant)``: modules whose bound forward would not run, and non-H3 attention."""

    compiled, variant = [], []
    for name, module in model.named_modules():
        where = name or "<model>"
        if _compiled(module):
            compiled.append(f"{where}: compiled; the compiled graph keeps the traced forward")
        if name.endswith("attn"):
            if getattr(module, "hybrid", None) is not None:
                variant.append(f"{where}: VDN-H3 hybrid attention, not the released dense H3")
            if getattr(module, "to_gate_compress", None) is not None:
                variant.append(f"{where}: compression gate, not the released dense H3")
    return compiled, variant


def _quantized(module: torch.nn.Module) -> str | None:
    method = getattr(module, "quant_method", None)
    if method is not None and type(method).__name__ != "UnquantizedLinearMethod":
        return type(method).__name__
    return None


def _check_projection(tp: int):
    def check(name: str, module: torch.nn.Module, kind: str) -> str | None:
        if hasattr(module, "base_layer"):
            return (
                "a LoRA wrapper adds the LoRA output in its own forward; merge the adapter into "
                "the checkpoint first"
            )
        method = _quantized(module)
        if method is not None:
            return f"quantized ({method}); the strict kernels read BF16 weights"
        shards = getattr(module, "tp_size", 1)
        if shards != tp:
            return f"sharded over {shards} TP ranks, but tp={tp}"
        if kind == "column" and getattr(module, "gather_output", False):
            return "gathers its output; the strict column kernel returns the local shard"
        if kind == "row" and not getattr(module, "input_is_parallel", True):
            return "expects unsharded input; the strict row kernel takes the local shard"
        if kind == "row" and not getattr(module, "reduce_results", True):
            return "does not reduce its result; the strict row kernel always reduces"
        return None

    return check


def _cuda_bf16(tensor: torch.Tensor | None, what: str) -> str | None:
    if tensor is None:
        return f"{what} is missing"
    if not tensor.is_cuda or tensor.dtype != torch.bfloat16:
        return f"{what} must be CUDA BF16, got {tensor.dtype} on {tensor.device}"
    return None


def _check_norm(norm: Any) -> str | None:
    reasons = [_cuda_bf16(getattr(norm, "weight", None), "the norm weight")]
    if getattr(norm, "eps", None) is None:
        reasons.append("the norm has no explicit eps")
    found = [r for r in reasons if r is not None]
    return "; ".join(found) if found else None


def _check_adaln(module: torch.nn.Module, kind: str, tp: int) -> list[str]:
    """Why the AdaLN path of a block or the final layer cannot be bound (empty: it can)."""

    projection = getattr(module, "adaln_proj", None)
    if projection is None:
        return ["the AdaLN cache replaces adaln_proj; its precomputed tables are not the row"]
    linear = projection.linear
    reason = _check_projection(tp)("adaln_proj.linear", linear, "column")
    reasons = [None if reason is None else f"adaln_proj.linear: {reason}"]
    weight = getattr(linear, "weight", None)
    if weight is not None and weight.dtype == torch.float32:
        reasons.append("FP32 AdaLN weights (a curve-grid checkpoint), not the released H3")
    else:
        reasons.append(_cuda_bf16(weight, "adaln_proj.linear.weight"))
    reasons.append(_cuda_bf16(getattr(linear, "bias", None), "adaln_proj.linear.bias"))
    norms = ("norm1", "norm2") if kind == "block" else ("norm",)
    reasons += [_check_norm(getattr(module, n, None)) for n in norms]
    if kind == "block":
        for path in ("attn.qkv_proj", "mlp.fc1"):
            method = _quantized(module.get_submodule(path))
            if method is not None:
                reasons.append(f"{path} is quantized ({method}): SGLang's mxfp8 modulation path")
    elif getattr(module, "_pdd_heads", None) is not None:
        reasons.append("PDD distilled output heads are loaded, not the released H3")
    return [r for r in reasons if r is not None]


Binder = Callable[[torch.nn.Module, Any], Callable[..., Any]]


def _bind_linear(linear: torch.nn.Module, ops: Any) -> Callable[..., Any]:
    def forward(x):
        shape = x.shape
        (out,) = ops.det_linear_forward(x.reshape(-1, shape[-1]), linear.weight, linear.bias)
        return out.view(*shape[:-1], out.shape[-1]), None

    return forward


def _no_tables(adaln_params):
    if adaln_params is not None:
        raise RuntimeError(
            "strict H3 rollout on SGLang: precomputed AdaLN tables bypass the bound projection"
        )


def _bind_block(block: torch.nn.Module, ops: Any) -> Callable[..., Any]:
    def forward(x, *, adaln_input, combined_indices, adaln_params=None, **attention_kwargs):
        """norm1 -> modulation -> attention -> gated residual, then the same with the MLP."""

        _no_tables(adaln_params)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(
            adaln_input
        )
        index = combined_indices.long()
        h = ops.norm.forward_modulated(
            x, block.norm1.weight, shift_msa, scale_msa, index, block.norm1.eps
        )
        h = block.attn(h, x_prequant=None, **attention_kwargs)
        x = ops.gate(x, h, gate_msa, index)
        h = ops.norm.forward_modulated(
            x, block.norm2.weight, shift_mlp, scale_mlp, index, block.norm2.eps
        )
        return ops.gate(x, block.mlp(h), gate_mlp, index)

    return forward


def _bind_final(layer: torch.nn.Module, ops: Any) -> Callable[..., Any]:
    def forward(x, *, adaln_input, inverse_indices, adaln_params=None):
        """``final_adaln_out``, then SGLang's FP32 output heads."""

        _no_tables(adaln_params)
        shift, scale = layer.adaln_proj(adaln_input)
        h = ops.norm.forward_modulated(
            x, layer.norm.weight, shift, scale, inverse_indices.long(), layer.norm.eps
        )
        return layer._project(h.to(torch.float32))

    return forward


def _bind_norm(norm: torch.nn.Module, ops: Any) -> Callable[..., Any]:
    def forward(x):
        return ops.norm(x, norm.weight, norm.eps)

    return forward


def _select(model: torch.nn.Module, tp: int) -> tuple[list[tuple[str, Any, Binder]], list[str]]:
    """The norm / AdaLN forwards to bind and the refusals, before anything changes."""

    selected: list[tuple[str, Any, Binder]] = []
    refused: list[str] = []
    for name, module in model.named_modules():
        kind = _ADALN_MODULES.get(type(module).__name__)
        if kind is not None:
            reasons = _check_adaln(module, kind, tp)
            if reasons:
                refused.append(f"{name}: " + "; ".join(reasons))
                continue
            selected.append((f"{name}.adaln_proj.linear", module.adaln_proj.linear, _bind_linear))
            selected.append((name, module, _bind_block if kind == "block" else _bind_final))
        elif _REFINER_NORM.search(name):
            reason = _check_norm(module)
            if reason is not None:
                refused.append(f"{name}: {reason}")
            else:
                selected.append((name, module, _bind_norm))
    return selected, refused


class _Ops:
    """The training graph's operators, built once per install."""

    def __init__(self) -> None:
        from rl_engine.backends.cuda.model_specific.minimax_h3.det_linear import det_linear_forward
        from rl_engine.backends.cuda.model_specific.minimax_h3.gate_residual import (
            H3GateResidualCudaOp,
        )
        from rl_engine.backends.cuda.model_specific.minimax_h3.rmsnorm import H3RMSNormCudaOp

        self.det_linear_forward = det_linear_forward
        self.norm = H3RMSNormCudaOp()
        self.gate = H3GateResidualCudaOp()


def check_sglang_server_args(server_args: Any) -> None:
    """Raise if SGLang's server settings change the DiT's arithmetic or bypass a bound forward."""

    refusals = []
    for flag in ("enable_torch_compile", "regional_compile", "enable_breakable_cuda_graph"):
        if getattr(server_args, flag, False):
            refusals.append(f"{flag} is set")
    if getattr(server_args, "cache_dit_config", None):
        refusals.append("cache_dit_config is set (cached blocks are not recomputed)")
    if getattr(server_args, "quantization", None) or getattr(
        server_args, "component_quantizations", None
    ):
        refusals.append("quantization is set")
    if getattr(server_args, "lora_path", None):
        refusals.append("lora_path is set (load a checkpoint with the adapter merged)")
    for flag in ("dit_cpu_offload", "dit_layerwise_offload", "use_fsdp_inference"):
        if getattr(server_args, flag, False):
            refusals.append(f"{flag} is set (the strict kernels read resident CUDA weights)")
    for flag in ("cpu_offload_components", "layerwise_offload_components"):
        if "transformer" in (getattr(server_args, flag, None) or ()):
            refusals.append(f"{flag} names the transformer")
    if refusals:
        raise RuntimeError("strict H3 rollout on SGLang: " + "; ".join(refusals))


def check_sglang_sampling_params(sampling_params: Any) -> None:
    """Raise if a request's sampling settings skip DiT steps (the blocks would not run)."""

    refusals = [
        f"{flag} is set (it skips transformer steps)"
        for flag in ("enable_spectrum", "enable_teacache")
        if getattr(sampling_params, flag, False)
    ]
    if refusals:
        raise RuntimeError("strict H3 rollout on SGLang: " + "; ".join(refusals))


def install_h3_sglang_rollout(
    model: torch.nn.Module, *, collective=None, tp: int = 1, strict: bool = True
) -> H3SGLangRolloutHandle:
    """Bind the strict projections, AdaLN paths and norms of SGLang's H3 DiT.

    ``tp`` is SGLang's TP size and ``collective`` a deterministic collective over SGLang's TP
    group (``tp > 1``). With ``strict``, one error names every refusal and no forward changes;
    without it, refused modules keep SGLang's forward and are listed in ``handle.skipped``, as is
    a VDN-H3 or compression-gate attention (it stays SGLang's, as every attention does; the rest
    of its block is bound). Model-level refusals still raise: an SGLang version the bound
    forwards were not validated on, and a compiled model, whose forwards cannot be bound.
    """

    version = _version_refusal()
    if version is not None:
        raise RuntimeError("H3 rollout on SGLang: " + version)
    compiled, variant = _model_refusals(model)
    refusals = compiled + variant if strict else compiled
    if refusals:
        raise RuntimeError("H3 rollout on SGLang: " + "; ".join(refusals))
    selected, refused = _select(model, tp)
    # Built before any forward changes: it raises if rl_engine._C lacks the H3 norm or gate.
    ops = _Ops() if selected else None
    try:
        base = install_h3_rollout(
            model, collective=collective, tp=tp, strict=strict, check=_check_projection(tp)
        )
    except RuntimeError as error:
        if strict and refused:
            raise RuntimeError("; ".join([str(error), *refused])) from None
        raise
    if strict and refused:
        base.remove()  # the projections were bound; nothing stays changed
        raise RuntimeError("; ".join(refused))
    handle = H3SGLangRolloutHandle(  # ``variant`` is empty here with ``strict`` (it raised)
        patched=base.patched, skipped=variant + base.skipped + refused, _restore=base._restore
    )
    for name, module, bind in selected:
        handle._restore.append((module, module.__dict__.get("forward", _ABSENT)))
        module.forward = bind(module, ops)
        handle.patched.append(name)
    if any(bind is _bind_block for _, _, bind in selected) and callable(
        getattr(model, "_can_batch_block_adaln", None)
    ):
        # The batched path hands the blocks precomputed tables, which a bound block refuses.
        handle._attributes.append(
            (model, "_can_batch_block_adaln", model.__dict__.get("_can_batch_block_adaln", _ABSENT))
        )
        setattr(model, "_can_batch_block_adaln", _never)
        handle.overrides.append(
            "_can_batch_block_adaln: off (each block runs its bound AdaLN projection)"
        )
    for name in handle.patched:
        module = model.get_submodule(name)
        handle.implementations[name] = IMPLEMENTATIONS[_kind(name, module)]
        module.forward = _counted(module.forward, handle.calls, name)
    return handle


def report(model: torch.nn.Module, handle: H3RolloutHandle) -> dict:
    """The handle's report for SGLang: what is bound, skipped and still SGLang's own, and (for an
    ``H3SGLangRolloutHandle``) how often each bound forward ran and which kernel it runs."""

    out = handle.report()
    out["engine"] = ENGINE
    out["sglang"] = _sglang_version()
    out["external"] = list(EXTERNAL)
    return out
