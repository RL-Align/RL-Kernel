# SPDX-License-Identifier: Apache-2.0
"""Existing aiter compression/cache scatter and vLLM FP32 partial-state writes."""

import importlib
import importlib.metadata

import torch

from rl_engine.contracts.operators.kv_compressor_c4 import (
    AITer_C4_LEGACY_SHA256,
    AITer_C4_SHA256,
    C4Error,
    require,
    source_hash,
)


class AiterCompressorStateBackend:
    device_type = "cuda"
    supports_graph = True

    def __init__(self):
        require(torch.version.hip is not None and torch.cuda.is_available(), "ROCm GPU required")
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        require(arch == "gfx942", "unvalidated compressor-state architecture")
        try:
            native = importlib.import_module("aiter.ops.flydsl.kernels.fused_compress_attn")
            writer = importlib.import_module(
                "vllm.models.deepseek_v4.common.ops.save_partial_states"
            )
            self.compress = native.flydsl_fused_compress_attn
            self.save_partial = writer.save_partial_states
        except (ImportError, AttributeError) as exc:
            raise C4Error("required aiter/vLLM state ABI unavailable") from exc
        h = source_hash(native.__file__)
        require(h in (AITer_C4_LEGACY_SHA256, AITer_C4_SHA256), "unvalidated compressor source")
        self.options = (
            {"quant_mode": "none", "k_split_num_waves": 1} if h == AITer_C4_SHA256 else {}
        )
        self.provenance = {
            "arch": arch,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": importlib.metadata.version("triton"),
            "flydsl": importlib.metadata.version("flydsl"),
            "compressor": h,
            "state_writer": source_hash(writer.__file__),
            "tensor_shim": source_hash(
                importlib.import_module("aiter.ops.flydsl.kernels.tensor_shim").__file__
            ),
            "bridge": source_hash(__file__),
            "recipe": "fixed single-wave online FP32",
        }

    def execute(self, arena):
        with torch.cuda.device(self.device):
            self._execute(arena)

    def _execute(self, arena):
        c = arena.contract
        # Layout adaptation only. Both destination buffers are preallocated,
        # so eager and captured execution use the same reader representation.
        arena._read_kv.copy_(arena._state[..., : c.width])
        arena._read_score.copy_(arena._state[..., c.width :])
        self.compress(
            kv_in=arena._input[:, : c.width],
            score_in=arena._input[:, c.width :],
            kv_state=arena._read_kv,
            score_state=arena._read_score,
            plan_gpu=arena._plan,
            state_slot_mapping=arena._state_slots,
            ape=arena._ape,
            rms_weight=arena._norm,
            rms_eps=arena.rms_eps,
            cos_cache=arena._cos,
            sin_cache=arena._sin,
            kv_cache=arena._cache,
            block_tables=arena._blocks,
            k_per_block=64,
            overlap=c.ratio == 4,
            ratio=c.ratio,
            head_dim=512,
            rope_head_dim=64,
            quant=False,
            **self.options,
        )
        # Ordering is essential: compression consumes PREVIOUS partial state.
        # Duplicate ring writers are removed by prepare() before this launch.
        self.save_partial(
            kv=arena._input[:, : c.width],
            score=arena._input[:, c.width :],
            ape=arena._ape,
            positions=arena._positions,
            state_cache=arena._state,
            slot_mapping=arena._write_slots,
            block_size=c.state_rows,
            state_width=c.width,
            compress_ratio=c.ratio,
        )
