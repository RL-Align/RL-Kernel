# SPDX-License-Identifier: Apache-2.0
"""T05 CUDA binding for the proposed P6 synthetic BF16 profile.

Ordinary forward checks numeric status before returning. Graph capture uses
prepare/allocate/launch, then check_status after replay. No CPU fallback or autograd.
"""

from dataclasses import dataclass, field

import torch

from .contract import CombinePlan, Context, ContractError, SavedForward, digest, require


def canonical_row_lookup(plan: CombinePlan, context: Context) -> tuple:
    """Validate identity and map opaque token IDs to physical rows."""
    require(type(plan) is CombinePlan, "SCHEMA_MISMATCH", "CombinePlan required")
    require(type(context) is Context, "SCHEMA_MISMATCH", "Context required")
    plan.validate(context)
    require(
        plan.hidden_size < 2**31
        and len(plan.token_ids) * 6 * plan.hidden_size < 2**31
        and len(plan.inverse_map) * plan.hidden_size < 2**31,
        "UNSUPPORTED_GEOMETRY",
        "tensor offsets exceed T05 signed 32-bit index range",
    )
    token_index = {t: i for i, t in enumerate(plan.token_ids)}
    lookup = [[-1] * 6 for _ in plan.token_ids]
    for row, (token, slot, valid) in enumerate(plan.inverse_map):
        if valid:
            lookup[token_index[token]][slot] = row
    return tuple(tuple(r) for r in lookup)


def tensor_bytes(tensor: torch.Tensor) -> str:
    """Explicit diagnostic readback; outside the production launch."""
    if tensor.is_cuda:
        with torch.cuda.device(tensor.device):
            require(
                not torch.cuda.is_current_stream_capturing(),
                "UNSUPPORTED_CAPABILITY",
                "byte readback during graph capture",
            )
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes().hex()


@dataclass(frozen=True)
class CombineBuffers:
    """Reusable launch buffers. Output is valid after check_status succeeds."""

    output: torch.Tensor
    saved_forward: SavedForward
    stages: dict
    _status: torch.Tensor = field(repr=False)
    _owner: object = field(repr=False)
    _block_size: int = field(repr=False)

    def check_status(self):
        with torch.cuda.device(self.output.device):
            require(
                not torch.cuda.is_current_stream_capturing(),
                "UNSUPPORTED_CAPABILITY",
                "check status after graph replay, outside capture",
            )
            values = self._status.cpu().tolist()
        require(
            all(v >= 0 for v in values), "INCOMPLETE_ARTIFACT", "buffers have not been launched"
        )
        require(not any(v & 4 for v in values), "NON_FINITE", "active input or arithmetic result")
        require(
            not any(v & 2 for v in values),
            "UNSUPPORTED_CAPABILITY",
            "subnormal is outside p6.synthetic-bf16.v1",
        )

    def stage_bytes(self):
        self.check_status()
        require(bool(self.stages), "UNSUPPORTED_CAPABILITY", "allocate debug stages first")
        return {
            key: [tensor_bytes(v) for v in value] if type(value) is list else tensor_bytes(value)
            for key, value in self.stages.items()
        }

    def trace(self):
        """Hash actual GPU debug readback using start-kit stage names."""
        stages = self.stage_bytes()
        plan = self._owner.plan
        return {
            "schema": "p6-local-trace.v1",
            "phase": "forward",
            "plan_fingerprint": plan.fingerprint,
            "order_hash": plan.order_hash,
            "boundary_hashes": {k: digest(v) for k, v in stages.items()},
        }


@dataclass(frozen=True)
class PreparedCombine:
    """Validated immutable host plan and private compact CUDA lookup."""

    plan: CombinePlan
    device: torch.device
    saved_forward: SavedForward
    _lookup: torch.Tensor = field(repr=False)

    def _validate_inputs(self, rows, shared, residual):
        n, h, p = len(self.plan.token_ids), self.plan.hidden_size, len(self.plan.inverse_map)
        for name, value, shape in (
            ("rows", rows, (p, h)),
            ("shared", shared, (n, h)),
            ("residual", residual, (n, h)),
        ):
            require(type(value) is torch.Tensor, "SCHEMA_MISMATCH", name + " tensor required")
            require(value.dtype == torch.bfloat16, "DTYPE_MISMATCH", name + " must be BF16")
            require(
                value.layout == torch.strided and value.is_contiguous(),
                "UNSUPPORTED_CAPABILITY",
                name + " must be contiguous strided",
            )
            require(
                not value.is_neg() and not value.is_conj(),
                "UNSUPPORTED_CAPABILITY",
                name + " has unresolved logical storage flags",
            )
            require(tuple(value.shape) == shape, "UNSUPPORTED_GEOMETRY", name + " shape")
            require(value.device == self.device, "UNSUPPORTED_CAPABILITY", name + " device")
            require(
                not value.requires_grad,
                "UNSUPPORTED_CAPABILITY",
                "T05 forward only; provider owner supplies autograd integration",
            )

    def allocate(self, *, debug=False, block_size=256):
        require(type(debug) is bool, "SCHEMA_MISMATCH", "debug must be bool")
        require(
            type(block_size) is int and block_size in (128, 256, 512),
            "UNSUPPORTED_CAPABILITY",
            "block size must be 128/256/512",
        )
        n, h = len(self.plan.token_ids), self.plan.hidden_size
        tiles = (h + block_size - 1) // block_size
        require(tiles <= 65535, "UNSUPPORTED_GEOMETRY", "hidden tiles exceed CUDA grid.y limit")
        with torch.cuda.device(self.device):
            require(
                not torch.cuda.is_current_stream_capturing(),
                "UNSUPPORTED_CAPABILITY",
                "allocate buffers before graph capture",
            )
            output = torch.empty((n, h), dtype=torch.bfloat16, device=self.device)
            status = torch.full(
                (n * tiles,),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            stages = {}
            if debug:

                def fp32(shape):
                    return torch.empty(shape, dtype=torch.float32, device=self.device)

                stages = {
                    "canonical_fp32": fp32((n, 6, h)),
                    "slot_partials_fp32": list(fp32((6, n, h)).unbind(0)),
                    "routed_fp32": fp32((n, h)),
                    "after_shared_fp32": fp32((n, h)),
                    "precast_fp32": fp32((n, h)),
                    "output_bf16": output,
                }
        return CombineBuffers(output, self.saved_forward, stages, status, self, block_size)

    def launch(self, rows, shared, residual, buffers: CombineBuffers):
        """Allocation-free launch; caller checks status after execution/replay.

        Input tensors must stay alive and unchanged until execution completes,
        and may not overlap any output, debug or status write buffer.
        """
        self._validate_inputs(rows, shared, residual)
        require(
            type(buffers) is CombineBuffers and buffers._owner is self,
            "IDENTITY_DRIFT",
            "buffers belong to a different prepared plan",
        )
        # Cross-token unpermute reads can race with another CTA's output stores.
        # Contiguous ranges include storage offsets; empty ranges cannot overlap.
        writes = [buffers.output, buffers._status]
        for stage in buffers.stages.values():
            writes.extend(stage if type(stage) is list else [stage])
        for target in writes:
            output_begin = target.data_ptr()
            output_end = output_begin + target.numel() * target.element_size()
            for value in (rows, shared, residual):
                begin = value.data_ptr()
                end = begin + value.numel() * value.element_size()
                require(
                    max(output_begin, begin) >= min(output_end, end),
                    "UNSUPPORTED_CAPABILITY",
                    "output/input storage overlap is unsupported (including debug/status)",
                )
        n, h = len(self.plan.token_ids), self.plan.hidden_size
        if not n:
            return buffers
        from rl_engine.kernels.ops.triton.moe.combine import combine_kernel

        debug = bool(buffers.stages)
        s = buffers.stages
        # Dummy pointers are eliminated in the compiled debug-off variant.
        canonical = s["canonical_fp32"] if debug else buffers.output
        partials = s["slot_partials_fp32"][0] if debug else buffers.output
        with torch.cuda.device(self.device):
            combine_kernel[(n, (h + buffers._block_size - 1) // buffers._block_size)](
                rows,
                self._lookup,
                shared,
                residual,
                buffers.output,
                buffers._status,
                canonical,
                partials,
                s.get("routed_fp32", buffers.output),
                s.get("after_shared_fp32", buffers.output),
                s.get("precast_fp32", buffers.output),
                T=n,
                H=h,
                BLOCK=buffers._block_size,
                DEBUG=debug,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return buffers

    def forward(self, rows, shared, residual, *, debug=False, block_size=256):
        """Checked eager forward; prepare once for repeated launches."""
        self._validate_inputs(rows, shared, residual)
        buffers = self.allocate(debug=debug, block_size=block_size)
        self.launch(rows, shared, residual, buffers)
        buffers.check_status()
        return buffers


def prepare_combine(plan: CombinePlan, context: Context, device) -> PreparedCombine:
    lookup = canonical_row_lookup(plan, context)
    device = torch.device(device)
    require(device.type == "cuda", "UNSUPPORTED_CAPABILITY", "T05 requires NVIDIA CUDA")
    require(
        torch.cuda.is_available() and torch.version.hip is None,
        "UNSUPPORTED_CAPABILITY",
        "NVIDIA CUDA unavailable",
    )
    device = torch.device(
        "cuda", torch.cuda.current_device() if device.index is None else device.index
    )
    with torch.cuda.device(device):
        require(
            torch.cuda.get_device_capability(device)[0] >= 8,
            "UNSUPPORTED_CAPABILITY",
            "T05 requires sm80+",
        )
        require(
            not torch.cuda.is_current_stream_capturing(),
            "UNSUPPORTED_CAPABILITY",
            "prepare plan outside graph capture",
        )
        try:
            from rl_engine.kernels.ops.triton.moe.combine import combine_kernel  # noqa: F401
        except ImportError as exc:
            raise ContractError("UNSUPPORTED_CAPABILITY", "Triton CUDA backend missing") from exc
        table = torch.tensor(lookup, dtype=torch.int64, device=device).reshape(-1, 6)
    return PreparedCombine(plan, device, SavedForward.capture(plan, context), table)


def fused_moe_combine_fwd(plan, rows, shared, residual, context, *, debug=False):
    """Convenience binding; prepare once for repeated/graph launches."""
    require(type(rows) is torch.Tensor, "SCHEMA_MISMATCH", "rows tensor required")
    return prepare_combine(plan, context, rows.device).forward(rows, shared, residual, debug=debug)
