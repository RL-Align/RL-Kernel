# SPDX-License-Identifier: Apache-2.0
"""Reuse aiter C4 overlap compression and the pinned mLite functional VJP."""

import importlib
import importlib.metadata
import importlib.util
import os

import torch

from rl_engine.contracts.operators.kv_compressor_c4 import (
    AITer_C4_SHA256,
    AITer_C4_LEGACY_SHA256,
    C4_BACKEND_ID,
    C4Error,
    MLITE_BACKWARD_SHA256,
    require,
    source_hash,
)
from rl_engine.ops.autograd.backward_runtime import record_backward


class AiterC4Backend:
    backend_id = C4_BACKEND_ID
    device_type = "cuda"

    def __init__(self, *, megatron_backward_path=None):
        require(torch.version.hip is not None and torch.cuda.is_available(), "C4 requires ROCm")
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        require(arch == "gfx942", f"unvalidated C4 architecture: {arch}")
        names = ("aiter.ops.flydsl.kernels.fused_compress_attn",)
        try:
            self.aiter = importlib.import_module(names[0])
            require(
                callable(self.aiter.flydsl_fused_compress_attn),
                "aiter C4 entrypoint is not callable",
            )
        except (ImportError, AttributeError) as exc:
            raise C4Error("pinned aiter C4 ABI unavailable") from exc
        provider_hash = source_hash(getattr(self.aiter, "__file__", None))
        require(
            provider_hash in (AITer_C4_SHA256, AITer_C4_LEGACY_SHA256),
            "unvalidated aiter C4 source revision",
        )
        # Two explicitly identified upstream ABIs, not a silent numerical fallback.
        self.launch_options = (
            {"quant_mode": "none", "k_split_num_waves": 1}
            if provider_hash == AITer_C4_SHA256
            else {}
        )
        path = megatron_backward_path or os.environ.get("RLK_MLITE_DSV4_BACKWARD_SOURCE")
        if path is None:
            try:
                module = importlib.import_module(
                    "megatron.lite.model.deepseek_v4.vllm.primitive.attention.backward"
                )
                path = module.__file__
            except (ImportError, AttributeError) as exc:
                raise C4Error(
                    "provide the pinned mLite backward.py source or install mLite"
                ) from exc
        require(source_hash(path) == MLITE_BACKWARD_SHA256, "unvalidated mLite backward source")
        spec = importlib.util.spec_from_file_location("_rlk_pinned_mlite_c4_backward", path)
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
            "aiter_c4": provider_hash,
            "launch_options": self.launch_options,
            "padding": "native state-phase negative-position masking",
            "mlite_backward": MLITE_BACKWARD_SHA256,
            "adapter": source_hash(__file__),
        }

    def forward(self, bound, ape, norm, cos, sin, contract):
        groups = len(bound.logical_keys)
        dim = contract.head_dim
        if groups == 0:
            return torch.empty((0, dim), dtype=torch.bfloat16, device=bound.values.device)
        blocks = max((block for block, _slot in bound.cache_slots_host), default=-1) + 1
        cache = torch.empty((blocks, 64, dim), dtype=torch.bfloat16, device=bound.values.device)
        provider_values = bound.context_values.reshape(-1, bound.context_values.shape[-1])
        state_kv = bound.kv_state
        state_score = bound.score_state
        state_slots = bound.state_slots
        plan = bound.plan.contiguous()
        block_table = bound.block_table
        self.aiter.flydsl_fused_compress_attn(
            kv_in=provider_values[:, : 2 * dim],
            score_in=provider_values[:, 2 * dim :],
            kv_state=state_kv,
            score_state=state_score,
            plan_gpu=plan,
            state_slot_mapping=state_slots,
            ape=ape,
            rms_weight=norm,
            rms_eps=contract.rms_eps,
            cos_cache=cos,
            sin_cache=sin,
            kv_cache=cache,
            block_tables=block_table,
            k_per_block=64,
            overlap=True,
            ratio=4,
            head_dim=dim,
            rope_head_dim=64,
            quant=False,
            use_ue8m0=True,
            preshuffle=True,
            **self.launch_options,
        )
        return torch.stack(
            [cache[block, slot] for block, slot in bound.cache_slots_host],
            dim=0,
        ).contiguous()

    def vjp(self, bound, ape, norm, cos, sin, grad, contract):
        contributions, ape_rows, norm_rows = [], [], []
        cache = torch.cat((cos, sin), dim=-1)
        with torch.inference_mode(False), torch.enable_grad():
            for index, (_request, group) in enumerate(bound.logical_keys):
                context = bound.context_values[index].float().detach().requires_grad_()
                a = ape.detach().requires_grad_()
                n = norm.detach().requires_grad_()
                group_ids = torch.tensor(
                    [group - 1, group] if group > 0 else [0],
                    dtype=torch.int64,
                    device=context.device,
                )
                source = context if group > 0 else context[4:]
                y = self.functional(
                    source,
                    a,
                    n,
                    group_ids,
                    cache,
                    ratio=4,
                    head_dim=contract.head_dim,
                    rope_dim=64,
                    eps=contract.rms_eps,
                )
                if group > 0:
                    grad_out = torch.cat((torch.zeros_like(y[:1]), grad[index : index + 1].float()))
                    dx, da, dn = torch.autograd.grad(y, (context, a, n), grad_outputs=grad_out)
                else:
                    dx, da, dn = torch.autograd.grad(
                        y, (source, a, n), grad_outputs=grad[index : index + 1].float()
                    )
                    dx = torch.cat((torch.zeros_like(context[:4]), dx))
                contributions.append((bound.context_indices_host[index], dx))
                ape_rows.append(da)
                norm_rows.append(dn)
        record_backward(
            "indexer_compressor_c4" if contract.family == "indexer_c4" else "kv_compressor_c4",
            kernel_id="megatron.compressed_compact_graph.ratio4+rlk.fixed_order_rows",
            impl="upstream_functional_vjp",
            family=contract.family,
        )
        return contributions, torch.stack(ape_rows), torch.stack(norm_rows)
