import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.vjp_fp32 import reduce_rows_fp32, rmsnorm_dweight_rows_fp32

_RMSNORM_API_VERSION = 2


def _fold_dweight_rows(rows: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The single left-fold entrypoint for this backend's dweight reductions.

    Both the plain and the gated backward route through here so the file keeps
    one auditable reduction path; the ascending-row fp32 fold is what makes
    dweight independent of the batch layout.
    """
    return reduce_rows_fp32(rows).to(dtype)


def _require_cuda_rmsnorm() -> None:
    """Raise when the compiled RMSNorm bindings are missing or incompatible.

    The registry treats a backend whose construction raises as unavailable and
    falls through to the next candidate, so calling this from ``__init__`` is
    what lets a CUDA-first priority list degrade to the PyTorch reference on a
    build without the extension. Mirrors ``_require_cuda_activation`` in the
    activation ops.
    """
    if not _EXT_AVAILABLE or _C is None:
        raise RuntimeError("CUDA RMSNorm requires the compiled rl_engine._C extension.")
    names = ("rmsnorm_forward", "rmsnorm_backward_dx")
    missing = [name for name in names if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(
            f"CUDA RMSNorm symbols ({', '.join(missing)}) are not compiled into _C. "
            "Rebuild the extension with csrc/cuda/rmsnorm.cu."
        )
    api_version = getattr(_C, "rmsnorm_api_version", None)
    if api_version != _RMSNORM_API_VERSION:
        raise RuntimeError(
            f"CUDA RMSNorm requires rl_engine._C RMSNorm API version {_RMSNORM_API_VERSION} "
            f"(loaded {api_version!r}). Rebuild the extension with csrc/cuda/rmsnorm.cu "
            "for weight_offset support."
        )


class RMSNormCuda(torch.autograd.Function):
    """
    PyTorch autograd wrapper for CUDA RMSNorm.
    """

    @staticmethod
    def forward(ctx, x, weight, mask=None, eps=1e-6, weight_offset=0.0):
        """
        Forward:
          y = x * rsqrt(mean(x^2) + eps) * (weight_offset + weight)

        Input:
          x:      [T, H], fp16/bf16/fp32 CUDA tensor
          weight: [H],    fp16/bf16/fp32 CUDA tensor
          mask:   [T],    bool CUDA tensor
          eps:    float
          weight_offset: float, added to weight in fp32 inside the kernel.
                  1.0 selects the zero-centred (1 + w) convention.

        Output:
          y: [T, H]
        """
        _require_cuda_rmsnorm()
        assert x.is_cuda, "x must be CUDA tensor"
        assert weight.is_cuda, "weight must be CUDA tensor"
        assert x.is_contiguous(), "x must be contiguous"
        assert weight.is_contiguous(), "weight must be contiguous"
        assert x.dim() == 2, "x must be [T, H]"
        assert weight.dim() == 1, "weight must be [H]"
        assert x.shape[1] == weight.shape[0], "hidden size mismatch"
        if mask is None:
            mask = torch.ones((x.shape[0],), device=x.device, dtype=torch.bool)
        else:
            assert mask.is_cuda, "mask must be CUDA tensor"
            assert mask.is_contiguous(), "mask must be contiguous"
            assert mask.dtype == torch.bool, "mask must be bool"
            assert mask.dim() == 1, "mask must be [T]"
            assert mask.shape[0] == x.shape[0], "mask length mismatch"

        y, rstd = _C.rmsnorm_forward(x, weight, float(eps), float(weight_offset))

        ctx.save_for_backward(x, weight, rstd, mask)
        ctx.eps = eps
        ctx.weight_offset = float(weight_offset)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward:
          dx = CUDA row-wise deterministic kernel
          dw = FP32 row contributions followed by an ascending-row left fold
        """
        x, weight, rstd, mask = ctx.saved_tensors
        dy = grad_out.contiguous()

        dx = _C.rmsnorm_backward_dx(dy, x, weight, rstd, ctx.weight_offset)

        # The shape-independent FP32 left fold preserves the C2 Batch/Chunk
        # reduction order while the CUDA reducer executes it in one launch.
        rows = rmsnorm_dweight_rows_fp32(x, dy, rstd=rstd)
        # Multiplication is part of the pre-existing mask contract, including
        # IEEE propagation for non-finite inactive contributions.
        rows = rows * mask.to(dtype=rows.dtype).unsqueeze(-1)
        dw = _fold_dweight_rows(rows, weight.dtype)
        record_backward(
            "rms_norm",
            kernel_id=(
                "rl_engine._C.rmsnorm_backward_dx"
                "+rl_engine.kernels.ops.vjp_fp32.rmsnorm_dweight_rows_fp32"
                "+rl_engine.kernels.ops.vjp_fp32.reduce_rows_fp32"
            ),
            impl="cuda_rmsnorm_dx_declared_fp32_rowfold_dw",
            family="cuda",
        )

        # dw is unchanged by the offset: d/dw (offset + w) == d/dw w.
        return dx, dw, None, None, None


def rmsnorm_cuda(x, weight, eps=1e-6, mask=None, weight_offset=0.0):
    """
    use:
        y = rmsnorm_cuda(x, weight)
        y = rmsnorm_cuda(x, weight, mask=mask)
        y = rmsnorm_cuda(x, weight, weight_offset=1.0)   # zero-centred weight
    """
    return RMSNormCuda.apply(x, weight, mask, eps, weight_offset)


class RMSNormCudaOp:
    """CUDA RMSNorm wrapper compatible with the shared operator harness."""

    backward_impl = "cuda_rmsnorm_dx_declared_fp32_rowfold_dw"

    def __init__(self) -> None:
        _require_cuda_rmsnorm()

    #: Added to the weight in fp32 inside the kernel. Subclasses override it;
    #: 0.0 is the plain convention.
    weight_offset = 0.0

    def __call__(self, x, weight, *, eps=1e-6):
        return self.forward(x, weight, eps=eps)

    def forward(self, x, weight, *, eps=1e-6):
        hidden = x.shape[-1]
        x_2d = x.contiguous().view(-1, hidden)
        y_2d = rmsnorm_cuda(x_2d, weight.contiguous(), eps=eps, weight_offset=self.weight_offset)
        return y_2d.view_as(x)

    def parameter_vjp_contributions_fp32(self, *, x, weight, grad_output, eps=1e-6):
        hidden = x.shape[-1]
        # Only `rstd` is used, and it does not depend on the offset; the offset is
        # passed so this is the same call the forward makes, not because it matters.
        _, rstd = _C.rmsnorm_forward(
            x.contiguous().reshape(-1, hidden),
            weight.contiguous(),
            float(eps),
            float(self.weight_offset),
        )
        rows = rmsnorm_dweight_rows_fp32(x, grad_output, rstd=rstd.reshape(x.shape[:-1]))
        return {"weight": rows}


class Qwen3NextRMSNormCudaOp(RMSNormCudaOp):
    """Zero-centred CUDA RMSNorm: ``y = x * rstd * (1 + weight)``.

    The decoder and final norms of Qwen3-Next (and Gemma) store a zero-centred
    weight. The ``+1`` is applied inside the kernel after the fp32 upcast, so it
    is never rounded through the low-precision weight dtype.
    """

    weight_offset = 1.0


# --------------------------------------------------------------------------- #
# Gated RMSNorm (Qwen3-Next GDN block)
# --------------------------------------------------------------------------- #


def _require_cuda_symbols(what: str, *names: str) -> None:
    """Raise when the compiled kernels backing ``what`` are missing.

    The registry treats a backend whose construction raises as unavailable and
    falls through, so calling this from ``__init__`` is what lets a CUDA-first
    priority list degrade to the PyTorch reference on a build without the
    extension. Mirrors ``_require_cuda_activation`` in the activation ops.
    """
    if not _EXT_AVAILABLE or _C is None:
        raise RuntimeError(f"{what} requires the compiled rl_engine._C extension.")
    missing = [name for name in names if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(
            f"{what} symbols ({', '.join(missing)}) are not compiled into _C. "
            "Rebuild the extension with csrc/cuda/rmsnorm.cu."
        )


#: Gate activations understood by the CUDA kernel, in binding order. ``swish`` is
#: an alias for ``silu``, as in vLLM's GDN block, which maps ``output_gate_type``
#: "swish" to "silu" before constructing ``RMSNormGated``.
_GATE_ACTIVATIONS = {"silu": 0, "swish": 0, "sigmoid": 1}


def _check_gate_activation(activation: str) -> int:
    if activation not in _GATE_ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {sorted(_GATE_ACTIVATIONS)}, got {activation!r}"
        )
    return _GATE_ACTIVATIONS[activation]


def _gate_activation_fp32(gate: torch.Tensor, activation: int) -> torch.Tensor:
    """act(gate) in fp32, matching the kernel's ``gate_activation``."""
    gate32 = gate.float()
    return torch.nn.functional.silu(gate32) if activation == 0 else torch.sigmoid(gate32)


def _gate_activation_grad_fp32(gate: torch.Tensor, activation: int) -> torch.Tensor:
    """d act(gate) / d gate in fp32, matching ``gate_activation_grad``."""
    gate32 = gate.float()
    sigma = torch.sigmoid(gate32)
    if activation == 0:
        return sigma * (1.0 + gate32 * (1.0 - sigma))
    return sigma * (1.0 - sigma)


class RMSNormGatedCuda(torch.autograd.Function):
    """Autograd wrapper for the gated CUDA RMSNorm.

    Forward is the fused kernel. Backward is assembled from deterministic
    pieces: ``dx`` from a row-local CUDA kernel, ``dweight`` from fp32 row
    contributions reduced by the ascending-row left fold, and ``dgate`` purely
    elementwise in fp32 (no reduction, so batch invariance is trivial).
    """

    @staticmethod
    def forward(ctx, x, weight, gate, eps=1e-6, weight_offset=0.0, activation=0):
        """
        Forward:
          y = x * rsqrt(mean(x^2) + eps) * (weight_offset + weight) * act(gate)

        Input:
          x, gate: [T, H], fp16/bf16/fp32 CUDA tensors of matching dtype
          weight:  [H]
          activation: 0 = silu/swish, 1 = sigmoid
        """
        assert x.is_cuda and weight.is_cuda and gate.is_cuda, "inputs must be CUDA tensors"
        assert x.is_contiguous() and weight.is_contiguous() and gate.is_contiguous()
        assert x.dim() == 2, "x must be [T, H]"
        assert weight.dim() == 1, "weight must be [H]"
        assert gate.shape == x.shape, "gate must match x"
        assert _EXT_AVAILABLE and hasattr(
            _C, "rmsnorm_gated_forward"
        ), "Gated RMSNorm CUDA extension is unavailable. Rebuild with csrc/cuda/rmsnorm.cu."

        y, rstd = _C.rmsnorm_gated_forward(
            x, weight, gate, float(eps), float(weight_offset), int(activation)
        )

        ctx.save_for_backward(x, weight, gate, rstd)
        ctx.eps = eps
        ctx.weight_offset = float(weight_offset)
        ctx.activation = int(activation)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, gate, rstd = ctx.saved_tensors
        dy = grad_out.contiguous()
        act = ctx.activation

        dx = _C.rmsnorm_gated_backward_dx(dy, x, weight, gate, rstd, ctx.weight_offset, act)

        # dweight: the gate is a per-element constant here, so the ungated row
        # contributions apply once dy carries act(gate).
        gate_act = _gate_activation_fp32(gate, act)
        rows = rmsnorm_dweight_rows_fp32(x, dy.float() * gate_act, rstd=rstd)
        dw = _fold_dweight_rows(rows, weight.dtype)

        # dgate: row-local and reduction-free.
        normed = x.float() * rstd.unsqueeze(-1)
        # Same guard as the kernels: an unconditional `+ 0.0` turns -0.0 weights into +0.0.
        scale = weight.float()
        if ctx.weight_offset != 0.0:
            scale = scale + ctx.weight_offset
        dgate = (dy.float() * normed * scale * _gate_activation_grad_fp32(gate, act)).to(gate.dtype)

        record_backward(
            "rms_norm_gated",
            kernel_id=(
                "rl_engine._C.rmsnorm_gated_backward_dx"
                "+rl_engine.kernels.ops.vjp_fp32.rmsnorm_dweight_rows_fp32"
                "+rl_engine.kernels.ops.vjp_fp32.reduce_rows_fp32"
            ),
            impl="cuda_rmsnorm_gated_dx_declared_fp32_rowfold_dw",
            family="cuda",
        )

        return dx, dw, dgate, None, None, None


def rmsnorm_gated_cuda(x, weight, gate, eps=1e-6, weight_offset=0.0, activation="silu"):
    """
    use:
        y = rmsnorm_gated_cuda(x, weight, gate)
        y = rmsnorm_gated_cuda(x, weight, gate, activation="sigmoid")
    """
    act = _check_gate_activation(activation)
    return RMSNormGatedCuda.apply(x, weight, gate, eps, weight_offset, act)


class Qwen3NextRMSNormGatedCudaOp:
    """CUDA gated RMSNorm for the Qwen3-Next GDN block.

    Deliberately not a subclass of :class:`RMSNormCudaOp`: it takes an extra
    required tensor, so it cannot stand in for one.

    ``out = x * rstd * weight * silu(gate)``, every multiply in fp32 with a
    single cast at the store. The weight is plain, not zero-centred, matching
    vLLM's ``RMSNormGated`` with ``norm_before_gate=True`` and ``group_size=None``
    -- which is how the GDN block constructs it. The op has no ``norm_before_gate``
    or ``group_size`` parameter, so other configurations are not implemented.

    ``activation`` is fixed at construction; the registry constructs the
    released config's ``"silu"``.
    """

    backward_impl = "cuda_rmsnorm_gated_dx_declared_fp32_rowfold_dw"

    #: The gated weight is plain; kept as an attribute so the surface matches
    #: the ungated op and a zero-centred variant stays one subclass away.
    weight_offset = 0.0

    def __init__(self, activation: str = "silu") -> None:
        _check_gate_activation(activation)
        self.activation = activation
        _require_cuda_symbols(
            "Gated CUDA RMSNorm", "rmsnorm_gated_forward", "rmsnorm_gated_backward_dx"
        )

    def __call__(self, x, weight, gate, *, eps=1e-6):
        return self.forward(x, weight, gate, eps=eps)

    def forward(self, x, weight, gate, *, eps=1e-6):
        if gate.shape != x.shape:
            raise ValueError(f"gate must match x, got {tuple(gate.shape)} vs {tuple(x.shape)}")
        hidden = x.shape[-1]
        x_2d = x.contiguous().view(-1, hidden)
        gate_2d = gate.contiguous().view(-1, hidden)
        y_2d = rmsnorm_gated_cuda(
            x_2d,
            weight.contiguous(),
            gate_2d,
            eps=eps,
            weight_offset=self.weight_offset,
            activation=self.activation,
        )
        return y_2d.view_as(x)

    def parameter_vjp_contributions_fp32(self, *, x, weight, gate, grad_output, eps=1e-6):
        if gate.shape != x.shape:
            raise ValueError(f"gate must match x, got {tuple(gate.shape)} vs {tuple(x.shape)}")
        hidden = x.shape[-1]
        act = _GATE_ACTIVATIONS[self.activation]
        _, rstd = _C.rmsnorm_gated_forward(
            x.contiguous().reshape(-1, hidden),
            weight.contiguous(),
            gate.contiguous().reshape(-1, hidden),
            float(eps),
            float(self.weight_offset),
            act,
        )
        rows = rmsnorm_dweight_rows_fp32(
            x,
            grad_output.float() * _gate_activation_fp32(gate, act),
            rstd=rstd.reshape(x.shape[:-1]),
        )
        return {"weight": rows}
