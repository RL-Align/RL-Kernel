# SPDX-License-Identifier: Apache-2.0
"""Native building blocks for item 26; no new numerical GPU kernels.

These are internal numerical entrypoints, not an implicit autograd fallback.
The training bridge must supply their VJPs and canonical logical metadata.
"""

import hashlib
import importlib
from pathlib import Path

import torch

from rl_engine.backends.rocm.gemm.indexer_projection import AiterIndexerProjectionBackend
from rl_engine.contracts.operators.indexer_projection import IndexerProjectionError


def require(condition, message):
    if not condition:
        raise IndexerProjectionError(message)


class AiterIndexerTrainingMath:
    """Reuse aiter fixed GEMM, vLLM BI batched GEMM, and MHA row reductions.

    Triton's language-level FP32 default selects TF32 on gfx942 unless
    explicitly overridden. A compiler `default_dot_input_precision` argument
    alone does not override that language decision. Require IEEE *before*
    importing/compiling kernels rather than silently accepting TF32 VJPs or
    mutating a process-global precision setting inside an operator.
    """

    def __init__(self):
        self.triton = importlib.import_module("triton")
        self._precision()
        self.linear_backend = AiterIndexerProjectionBackend()
        self.device = self.linear_backend.device
        modules = {
            "vllm.model_executor.layers.batch_invariant": None,
            "aiter.ops.triton._triton_kernels.attention.mha_fused_bwd": None,
        }
        for name in modules:
            modules[name] = importlib.import_module(name)
        self._bmm = modules["vllm.model_executor.layers.batch_invariant"].bmm_kernel
        self._row_dot = modules[
            "aiter.ops.triton._triton_kernels.attention.mha_fused_bwd"
        ]._bwd_preprocess
        with torch.cuda.device(self.device), torch.inference_mode(False):
            self._one = torch.ones((1,), dtype=torch.float32, device=self.device)
            self._ready = torch.cuda.Event()
            self._ready.record()
        self.provenance = {
            "profile": "aiter_vllm.indexer.fp32_ieee.fixed_bmm.rowdot.v1",
            "fp32_input_precision": "ieee",
            "linear": self.linear_backend.provenance,
            "sources": {
                name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for name, module in modules.items()
            },
            "adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def _precision(self):
        require(
            self.triton.knobs.language.fp32_default == "ieee",
            "start with TRITON_F32_DEFAULT=ieee; TF32 Indexer VJPs are not admitted",
        )

    def _check(self, *tensors):
        self._precision()
        require(all(t.device == self.device for t in tensors), "Indexer math device mismatch")

    def linear_fp32(self, x, weight):
        self._check(x, weight)
        require(
            x.ndim == weight.ndim == 2
            and x.shape[1] == weight.shape[1]
            and x.dtype == weight.dtype == torch.float32,
            "FP32 GEMM ABI mismatch",
        )
        with torch.cuda.device(self.device):
            if not x.shape[0] or not weight.shape[0] or not x.shape[1]:
                return torch.zeros(
                    (x.shape[0], weight.shape[0]), dtype=torch.float32, device=self.device
                )
            plan = self.linear_backend.plan(x.shape[1], weight.shape[0], "indexer_vjp", "fp32")
            return self.linear_backend._gemm16(
                x, weight, dtype=torch.float32, config=plan.config, backend="triton"
            )

    def bmm_fp32_output(self, a, b):
        self._check(a, b)
        require(
            a.ndim == b.ndim == 3 and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1],
            "batched GEMM geometry mismatch",
        )
        require(
            a.dtype == b.dtype and a.dtype in (torch.bfloat16, torch.float32),
            "batched GEMM operands must both be BF16 or both FP32",
        )
        batches, m, k = a.shape
        n = b.shape[2]
        with torch.cuda.device(self.device):
            if not batches or not m or not n or not k:
                return torch.zeros((batches, m, n), dtype=torch.float32, device=self.device)
            out = torch.empty((batches, m, n), dtype=torch.float32, device=self.device)
            self._bmm[(batches, self.triton.cdiv(m, 32) * self.triton.cdiv(n, 64))](
                a,
                b,
                out,
                batches,
                m,
                n,
                k,
                *a.stride(),
                *b.stride(),
                *out.stride(),
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=64,
                A_LARGE=a.numel() > 2**31,
                B_LARGE=b.numel() > 2**31,
                C_LARGE=out.numel() > 2**31,
                num_warps=4,
                num_stages=2,
                matrix_instr_nonkdim=16,
                kpack=1,
            )
            return out

    def row_dot(self, a, b):
        """One fixed native reduction per [query, head] row; strides may broadcast."""
        self._check(a, b)
        require(a.ndim == b.ndim == 3 and a.shape == b.shape, "row-dot shape mismatch")
        require(
            a.dtype in (torch.bfloat16, torch.float32)
            and b.dtype in (torch.bfloat16, torch.float32),
            "row-dot dtype mismatch",
        )
        m, h, d = a.shape
        require(d <= 4096, "row-dot feature budget exceeded")
        with torch.cuda.device(self.device):
            if not m or not h or not d:
                return torch.zeros((m, h), dtype=torch.float32, device=self.device)
            out = torch.empty((m, h), dtype=torch.float32, device=self.device)
            self._row_dot[(m, 1, h)](
                a,
                b,
                out,
                0,
                a.stride(1),
                a.stride(0),
                a.stride(2),
                0,
                b.stride(1),
                b.stride(0),
                b.stride(2),
                0,
                out.stride(1),
                out.stride(0),
                0,
                None,
                m,
                None,
                BLOCK_M=1,
                BLOCK_D_MODEL=d,
                BLOCK_D_MODEL_POW2=self.triton.next_power_of_2(d),
                IS_VARLEN=False,
                IS_FP8=False,
                num_warps=4,
            )
            return out

    def row_sum(self, values):
        with torch.cuda.device(self.device):
            torch.cuda.current_stream(self.device).wait_event(self._ready)
            return self.row_dot(values, self._one.expand_as(values))


class AiterIndexerAuxBackend(AiterIndexerTrainingMath):
    """Compose existing Indexer codecs with native sparse batched arithmetic."""

    def __init__(self, contract):
        super().__init__()
        from rl_engine.backends.rocm.attention.indexer_compressor_c4 import AiterIndexerC4Backend
        from rl_engine.backends.rocm.attention.indexer_icv import AiterIndexerICVBackend
        from rl_engine.contracts.operators.indexer_icv import IndexerICVContract

        with torch.cuda.device(self.device):
            self.codec = AiterIndexerC4Backend()
            self.icv = AiterIndexerICVBackend()
        self.contract = contract
        self.icv_contract = IndexerICVContract(
            num_heads=contract.num_heads,
            head_dim=128,
            precision=contract.precision,
            fp8_encoding="e4m3fnuz" if contract.precision == "fp8" else None,
            query_tile_rows=1,
            key_tile_rows=contract.selected_capacity,
            max_output_bytes=contract.max_workspace_bytes,
            max_prepared_key_bytes=contract.max_workspace_bytes,
        )
        self._softmax = importlib.import_module(
            "vllm.model_executor.layers.batch_invariant"
        ).softmax_batch_invariant
        self.provenance["codec"] = self.codec.provenance
        self.provenance["icv"] = self.icv.provenance
        self.provenance["aux_contract"] = contract.fingerprint
        rope = importlib.import_module("aiter.ops.rope")
        self._rope_forward, self._rope_backward = rope.rope_cached_fwd, rope.rope_cached_bwd
        self.provenance["q_rope_sources"] = {
            name: hashlib.sha256(
                Path(importlib.import_module(name).__file__).read_bytes()
            ).hexdigest()
            for name in (
                "aiter.ops.rope",
                "aiter.jit.module_rope_1c_cached_fwd",
                "aiter.jit.module_rope_1c_cached_bwd",
            )
        }

    def query_transform(self, q, cos, sin):
        self._check(q, cos, sin)
        if not q.shape[0]:
            return torch.empty_like(q)
        with torch.cuda.device(self.device):
            rotated = self._rope_forward(q[:, None], cos, sin, 1, True, True).squeeze(1)
            return self.codec._rotate(rotated.reshape(-1, 128)).reshape_as(q)

    def query_transform_vjp(self, grad, cos, sin):
        self._check(grad, cos, sin)
        if not grad.shape[0]:
            return torch.empty_like(grad)
        with torch.cuda.device(self.device):
            unrotated = self.codec._rotate(grad.bfloat16().reshape(-1, 128)).reshape_as(grad)
            return self._rope_backward(unrotated[:, None], cos, sin, 1, True, True).squeeze(1)

    def enable_backward(self):
        """Reuse item 25's verified CSR gather, not a scatter-add gradient."""
        if hasattr(self, "csr"):
            return
        from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend
        from rl_engine.backends.rocm.attention.joint_backward import AiterJointBackwardBackend

        with torch.cuda.device(self.device):
            self.csr = AiterJointBackwardBackend(AiterJointAttentionBackend())
        self.backward_provenance = {
            "forward": self.provenance,
            "csr": self.csr.provenance,
            "sink_reducer": "existing fixed32 canonical row reducer",
        }

    @torch.no_grad()
    def encode(self, values, *, role, version, packed=None):
        from rl_engine.models.deepseek_v4.indexer_icv import QuantizedICVRows

        self._check(values)
        require(
            values.dtype == torch.bfloat16 and values.is_contiguous() and values.shape[-1] == 128,
            "quantization master must be contiguous BF16 D128",
        )
        require(values.ndim == (3 if role == "q" else 2), "quantization role/shape mismatch")
        require(bool(torch.isfinite(values).all()), "non-finite quantization master")
        with torch.cuda.device(self.device), torch.inference_mode(False):
            if packed is None:
                data, scales = self.codec.quantize(
                    values.detach().reshape(-1, 128), self.icv_contract
                )
                data = data.reshape(*values.shape[:-1], data.shape[-1])
                scales = scales.reshape(*values.shape[:-1], scales.shape[-1])
            else:
                require(
                    isinstance(packed, tuple)
                    and len(packed) == 2
                    and all(isinstance(t, torch.Tensor) for t in packed),
                    "packed operands must be a data/scale tensor pair",
                )
                data, scales = packed
                require(
                    not data.requires_grad and not scales.requires_grad,
                    "packed/cache data and scales are frozen; differentiate the BF16 master",
                )
                require(
                    data.shape[:-1] == values.shape[:-1] and scales.shape[:-1] == values.shape[:-1],
                    "packed/master geometry mismatch",
                )
            operand = QuantizedICVRows.bind(
                data,
                scales,
                role=role,
                version=version,
                layout=self.icv_contract.layout,
                contract=self.icv_contract,
                ownership="snapshot",
            )
            require(operand.data.shape[:-1] == values.shape[:-1], "packed/master geometry mismatch")
            prepared, scale = self.icv.prepare(operand.data, operand.scales, self.icv_contract)
            real = prepared.float() if scale is None else prepared.float() * scale
            require(bool(torch.isfinite(real).all()), "non-finite decoded training operand")
            return operand, prepared, scale, real

    @staticmethod
    def gather_rows(values, indices):
        if not values.shape[0]:
            return torch.zeros(
                (*indices.shape, values.shape[-1]), dtype=values.dtype, device=values.device
            )
        rows = values.index_select(0, indices.clamp_min(0).long().flatten())
        return rows.reshape(*indices.shape, values.shape[-1])

    def raw_sparse_icv(self, q_prepared, k_prepared, q_scale, k_scale, indices):
        selected = self.gather_rows(k_prepared, indices)
        if self.contract.precision == "mxfp4":
            return self.bmm_fp32_output(q_prepared, selected.transpose(1, 2))
        # Preserve item 16's FP8 dot-then-independent-scale multiplication,
        # rather than silently changing it to BF16 or dequantize-before-dot.
        if not q_prepared.shape[0]:
            return torch.empty(
                (0, self.contract.num_heads, indices.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
        scales = self.gather_rows(k_scale, indices)
        with torch.cuda.device(self.device):
            return torch.cat(
                [
                    self.icv.dot(
                        q_prepared[i : i + 1],
                        selected[i],
                        q_scale[i : i + 1],
                        scales[i],
                        self.icv_contract,
                    )
                    for i in range(q_prepared.shape[0])
                ],
                0,
            )

    def masked_softmax(self, logits, valid):
        if not logits.shape[0]:
            return torch.empty_like(logits)
        # All-invalid rows are defined to have zero mass, not NaN softmax.
        active = valid.any(-1)
        masked = logits.masked_fill(~valid, float("-inf"))
        masked = torch.where(active[..., None], masked, 0.0)
        probabilities = self._softmax(masked, -1)
        return torch.where(valid, probabilities, 0.0)

    def forward_sparse(
        self, q_prepared, k_prepared, q_scale, k_scale, weights, teacher_q, teacher_k, metadata
    ):
        c = self.contract
        indices = metadata.selected_indices
        valid = (indices >= 0) & metadata.valid_rows[:, None]
        raw = self.raw_sparse_icv(q_prepared, k_prepared, q_scale, k_scale, indices)
        raw = torch.where(valid[:, None, :], raw, 0.0)
        require(bool(torch.isfinite(raw).all()), "non-finite active ICV")
        relu = torch.relu(raw * (128**-0.5))
        weighted = torch.where(metadata.valid_rows[:, None], weights, 0.0) * (c.num_heads**-0.5)
        scores = self.row_dot(
            relu.transpose(1, 2).contiguous(), weighted[:, None, :].expand(-1, indices.shape[1], -1)
        )
        require(bool(torch.isfinite(scores).all()), "non-finite active Indexer score")
        teacher_selected = self.gather_rows(teacher_k, indices)
        target_logits = self.bmm_fp32_output(teacher_q, teacher_selected.transpose(1, 2))
        target_logits = target_logits * c.softmax_scale
        require(
            bool(torch.isfinite(torch.where(valid[:, None, :], target_logits, 0.0)).all()),
            "non-finite active teacher logit",
        )
        target_heads = self.masked_softmax(
            target_logits, valid[:, None, :].expand_as(target_logits)
        )
        target = self.row_sum(target_heads.transpose(1, 2).contiguous()) / c.teacher_heads
        predicted = self.masked_softmax(scores, valid)
        terms = target * (torch.log(target + 1e-10) - torch.log(predicted + 1e-10))
        losses = self.row_sum(terms[:, None, :]).squeeze(1)
        losses = losses * (c.loss_coeff / c.global_query_count)
        return losses, raw, relu, predicted, target

    @torch.no_grad()
    def backward_parts(self, record, grad_loss):
        """Row-local VJPs; dK query/slot contributions are reduced later."""
        self._check(grad_loss)
        c = self.contract
        require(
            grad_loss.shape == record.per_query_loss.shape
            and grad_loss.dtype == torch.float32
            and bool(torch.isfinite(grad_loss).all()),
            "auxiliary loss gradient must be finite FP32 [queries]",
        )
        with torch.cuda.device(self.device):
            predicted, target = record.predicted, record.target
            upstream = -target / (predicted + 1e-10)
            dot = self.row_dot(upstream[:, None, :], predicted[:, None, :]).squeeze(1)
            ds = predicted * (upstream - dot[:, None])
            ds = ds * (grad_loss * (c.loss_coeff / c.global_query_count))[:, None]
            indices = record.metadata.selected_indices
            valid = (indices >= 0) & record.metadata.valid_rows[:, None]
            ds = torch.where(valid, ds, 0.0)
            dw = self.row_dot(record.relu, ds[:, None, :].expand_as(record.relu))
            dw = dw * (c.num_heads**-0.5)
            weights = torch.where(record.metadata.valid_rows[:, None], record.weights, 0.0)
            ddot = ds[:, None, :] * (weights * (c.num_heads**-0.5))[:, :, None]
            ddot = torch.where(record.raw_icv > 0, ddot * (128**-0.5), 0.0)
            selected_k = self.gather_rows(record.k.real, indices)
            dq = self.bmm_fp32_output(ddot, selected_k)
            dk_selected = self.bmm_fp32_output(ddot.transpose(1, 2), record.q.real)
            return dq, dk_selected, dw

    def reduce_keys(self, rows, indices, key_count):
        self.enable_backward()
        with torch.cuda.device(self.device):
            ptr, data = self.csr.build_csr(indices, key_count)
            out = torch.empty((key_count, 128), dtype=torch.float32, device=self.device)
            self.csr.gather_intermediate_into(rows, indices, ptr, data, out)
            return out
