# SPDX-License-Identifier: Apache-2.0
"""Unmodified upstream aiter forward and Megatron functional VJP, no new GPU kernel."""

import importlib
import importlib.metadata
import importlib.util
import os

import torch

from rl_engine.contracts.operators.kv_compressor_c128 import (
    AITer_HCA_SHA256,
    AITer_HCA_LEGACY_SHA256,
    C128_BACKEND_ID,
    C128Error,
    MLITE_BACKWARD_SHA256,
    require,
    source_hash,
)
from rl_engine.ops.autograd.backward_runtime import record_backward


class AiterC128Backend:
    backend_id = C128_BACKEND_ID
    device_type = "cuda"

    def __init__(self, *, megatron_backward_path=None):
        require(torch.version.hip is not None and torch.cuda.is_available(), "C128 requires ROCm")
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        require(arch == "gfx942", f"unvalidated C128 architecture: {arch}")
        name = "aiter.ops.flydsl.kernels.fused_compress_attn_hca"
        try:
            self.hca = importlib.import_module(name)
            provider_hash = source_hash(self.hca.__file__)
            require(
                provider_hash in (AITer_HCA_SHA256, AITer_HCA_LEGACY_SHA256),
                "unvalidated aiter HCA source revision",
            )
            self.legacy_hca = provider_hash == AITer_HCA_LEGACY_SHA256
            # The packaged 0.1.16 ABI is self-contained and exposes the
            # two-kernel composite launcher. Newer aiter splits out helpers.
            names = [name, "aiter.ops.flydsl.kernels.tensor_shim"]
            if not self.legacy_hca:
                names.extend(
                    [
                        "aiter.ops.flydsl.kernels.fused_compress_attn_common",
                        "aiter.ops.flydsl.kernels.kernels_common",
                        "aiter.ops.flydsl.kernels.quant_utils",
                    ]
                )
            modules = [importlib.import_module(item) for item in names]
            self.shim = modules[1]
            self.Stream = importlib.import_module("flydsl.expr.typing").Stream
            # Validate the ABI before admitting a backend identity.
            require(
                all(
                    callable(fn)
                    for fn in (
                        (
                            self.hca.flydsl_hca_compress_attn
                            if self.legacy_hca
                            else self.hca.flydsl_hca_compress_forward
                        ),
                        self.hca.compile_hca_norm_rope_scatter,
                        self.shim._run_compiled,
                    )
                ),
                "aiter HCA functions are not callable",
            )
        except (ImportError, AttributeError) as exc:
            raise C128Error("pinned aiter HCA / FlyDSL ABI unavailable") from exc
        sources = {
            name: source_hash(getattr(m, "__file__", None))
            for name, m in zip(names, modules, strict=True)
        }
        path = megatron_backward_path or os.environ.get("RLK_MLITE_DSV4_BACKWARD_SOURCE")
        if path is None:
            try:
                graph_module = importlib.import_module(
                    "megatron.lite.model.deepseek_v4.vllm.primitive.attention.backward"
                )
                path = graph_module.__file__
            except (ImportError, AttributeError) as exc:
                raise C128Error(
                    "provide the pinned mLite backward.py source or install mLite"
                ) from exc
        require(source_hash(path) == MLITE_BACKWARD_SHA256, "unvalidated mLite backward source")
        spec = importlib.util.spec_from_file_location("_rlk_pinned_mlite_c128_backward", path)
        require(spec is not None and spec.loader is not None, "mLite source loader unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.functional = module.compressed_compact_graph
        self.provenance = {
            "backend": self.backend_id,
            "arch": arch,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": importlib.import_module("triton").__version__,
            "flydsl": importlib.metadata.version("flydsl"),
            "sources": sources,
            "hca_abi": "packaged_composite" if self.legacy_hca else "split_forward",
            "mlite_backward": MLITE_BACKWARD_SHA256,
            "adapter": source_hash(__file__),
        }

    def forward(self, bound, ape, norm, cos, sin, contract):
        groups = bound.group_ids.shape[0]
        if groups == 0:
            return torch.empty((0, 512), dtype=torch.bfloat16, device=bound.values.device)
        values = bound.values
        state = torch.zeros((1, 128, 512), dtype=torch.float32, device=values.device)
        if self.legacy_hca:
            output = torch.empty((groups, 1, 512), dtype=torch.bfloat16, device=values.device)
            # Explicit values bypass the library's capacity-based dispatch.
            # This pinned ABI's norm/RoPE kernel is intrinsically one-wave.
            self.hca.flydsl_hca_compress_attn(
                kv_in=values[:, :512],
                score_in=values[:, 512:],
                kv_state=state,
                score_state=state,
                state_slot_mapping=bound.state_slots,
                plan_gpu=bound.plan,
                ape=ape,
                rms_weight=norm,
                rms_eps=contract.rms_eps,
                cos_cache=cos,
                sin_cache=sin,
                kv_cache=output,
                block_tables=bound.block_table,
                k_per_block=1,
                ratio=128,
                head_dim=512,
                rope_head_dim=64,
                k_split_num_waves=8,
                slice_size=64,
                stream=torch.cuda.current_stream(self.device),
            )
            return output[:, 0, :].contiguous()
        pooled = self.hca.flydsl_hca_compress_forward(
            kv_in=values[:, :512],
            score_in=values[:, 512:],
            kv_state=state,
            score_state=state,
            state_slot_mapping=bound.state_slots,
            plan_gpu=bound.plan,
            ape=ape,
            ratio=128,
            head_dim=512,
            k_split_num_waves=8,
            slice_size=64,
        )
        # Do NOT use the launcher's capacity-based norm-wave or pooling tuning.
        # Reuse its compiled norm/RoPE/scatter kernel with a fixed one-wave ABI.
        output = torch.empty((groups, 1, 512), dtype=torch.bfloat16, device=values.device)
        kernel = self.hca.compile_hca_norm_rope_scatter(
            head_dim=512,
            rope_head_dim=64,
            ratio=128,
            k_per_block=1,
            rms_weight_is_bf16=False,
            rms_eps=contract.rms_eps,
            quant=False,
            k_waves=1,
        )
        self.shim._run_compiled(
            kernel,
            pooled,
            512,
            bound.plan,
            norm,
            cos,
            sin,
            output,
            512,
            512,
            bound.block_table,
            bound.block_table.shape[1],
            output,
            0,
            0,
            groups,
            self.Stream(torch.cuda.current_stream()),
        )
        # This is a private compact output, not a write into a live KV cache.
        return output[:, 0]

    def vjp(self, bound, ape, norm, cos, sin, grad, contract):
        rows, ape_rows, norm_rows = [], [], []
        cache = torch.cat((cos, sin), dim=-1)
        # Fixed one-group shapes avoid batch-dependent backward reductions.
        # The graph itself is imported unchanged from the pinned mLite source.
        with torch.inference_mode(False), torch.enable_grad():
            for index in range(bound.group_ids.shape[0]):
                x = bound.values[index * 128 : (index + 1) * 128].float().detach().requires_grad_()
                a = ape.detach().requires_grad_()
                n = norm.detach().requires_grad_()
                y = self.functional(
                    x,
                    a,
                    n,
                    bound.group_ids[index : index + 1, 1],
                    cache,
                    ratio=128,
                    head_dim=512,
                    rope_dim=64,
                    eps=contract.rms_eps,
                )
                dx, da, dn = torch.autograd.grad(
                    y,
                    (x, a, n),
                    grad_outputs=grad[index : index + 1].float(),
                )
                rows.append(dx)
                ape_rows.append(da)
                norm_rows.append(dn)
        record_backward(
            "kv_compressor_c128",
            kernel_id="megatron.compressed_compact_graph.fp32.group1+rlk.canonical_rows",
            impl="upstream_functional_vjp",
            family="c128",
        )
        return torch.cat(rows), torch.stack(ape_rows), torch.stack(norm_rows)
