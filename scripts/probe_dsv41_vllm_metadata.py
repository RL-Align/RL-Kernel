# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Dump what vLLM main's DSv4.1 model hands its attention layers (for the rollout plugin).

Loads a checkpoint (the toy one from ``make_dsv41_toy_checkpoint.py``) in vLLM, in process,
runs two requests for a few decode steps and prints, for the first attention calls of the
first layers:
- every cache tensor the layer owns or reads (SWA, compressed KV, indexer K, compressor
  state): shape, stride, dtype;
- the forward context's attention metadata per cache prefix: dataclass fields, tensor shapes
  and leading values.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=. python scripts/probe_dsv41_vllm_metadata.py \\
        --model TOY_DIR --out probe.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

LINES: list[str] = []


def emit(s: str = "") -> None:
    LINES.append(s)
    print(s, flush=True)


def describe(value, indent: str = "    ") -> None:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        emit(f"{indent}{type(value).__name__}")
        for f in dataclasses.fields(value):
            describe_field(f.name, getattr(value, f.name), indent + "  ")
    elif isinstance(value, dict):
        for k, v in value.items():
            describe_field(str(k), v, indent + "  ")
    else:
        emit(f"{indent}{type(value).__name__}: {str(value)[:200]}")


def describe_field(name: str, v, indent: str) -> None:
    if isinstance(v, torch.Tensor):
        head = v.flatten()[:12].tolist() if v.numel() and v.dtype != torch.uint8 else ""
        emit(f"{indent}{name}: {v.dtype} {tuple(v.shape)} stride {v.stride()} {head}")
    elif dataclasses.is_dataclass(v) and not isinstance(v, type):
        emit(f"{indent}{name}:")
        describe(v, indent + "  ")
    elif isinstance(v, (int, float, bool, str, type(None))):
        emit(f"{indent}{name}: {v}")
    elif isinstance(v, (list, tuple)):
        emit(f"{indent}{name}: {type(v).__name__}[{len(v)}] {str(v)[:160]}")
    else:
        emit(f"{indent}{name}: <{type(v).__name__}>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=4, help="dump the first N layers")
    ap.add_argument("--calls", type=int, default=3, help="forward steps to dump")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out")
    args = ap.parse_args()

    import importlib

    import vllm.utils.deep_gemm as vllm_dg

    # vLLM prefers a pip-installed deep_gemm, which here is RL-Kernel's pin (e21c821, whose
    # paged MQA wants block_kv 64); vLLM main's builders need its vendored build.
    vllm_dg._import_deep_gemm = lambda: importlib.import_module("vllm.third_party.deep_gemm")

    from vllm import LLM, SamplingParams
    from vllm.forward_context import get_forward_context
    from vllm.models.deepseek_v41 import attention as v41_attention

    base = v41_attention.DeepseekV4Attention
    step = {"n": 0}
    original = base.forward

    def forward(self, positions, hidden_states, *a, **k):
        layer = getattr(self, "layer_id", None)
        if layer == 0:
            step["n"] += 1
        if step["n"] <= args.calls and layer is not None and layer < args.layers:
            emit(f"=== step {step['n']} layer {layer} {type(self).__name__} tokens "
                 f"{hidden_states.shape[0]} positions {positions[:16].tolist()}")
            for attr in ("compress_ratio", "is_kv_source", "kv_source_layer_id",
                         "compressed_cache_prefix", "window_size", "head_dim",
                         "n_local_heads", "padded_heads", "kv_cache_dtype",
                         "swa_bytes_per_token", "compressed_bytes_per_token",
                         "kv_page_alignment", "prefix"):
                if hasattr(self, attr):
                    emit(f"  {attr}: {getattr(self, attr)}")
            for name, mod in self.named_modules():
                kv = getattr(mod, "kv_cache", None)
                if isinstance(kv, torch.Tensor) and kv.numel():
                    emit(f"  cache {name or '<self>'} ({type(mod).__name__}, "
                         f"prefix {getattr(mod, 'prefix', '?')}, block_size "
                         f"{getattr(mod, 'block_size', '?')}): {kv.dtype} "
                         f"{tuple(kv.shape)} stride {kv.stride()}")
            md = get_forward_context().attn_metadata
            if isinstance(md, dict):
                for key, value in md.items():
                    if f"layers.{layer}." in key:
                        emit(f"  metadata[{key}]")
                        describe(value)
            else:
                describe(md)
        return original(self, positions, hidden_states, *a, **k)

    base.forward = forward
    for sub in base.__subclasses__():
        if "forward" in sub.__dict__:
            emit(f"note: {sub.__name__} overrides forward (not hooked)")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_len,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=0.6,
        max_num_seqs=8,
        trust_remote_code=True,
    )
    prompts = [{"prompt_token_ids": list(range(100, 170))},
               {"prompt_token_ids": list(range(300, 305))}]
    outs = llm.generate(prompts, SamplingParams(max_tokens=3, temperature=0.0))
    for o in outs:
        emit(f"generated {o.outputs[0].token_ids}")
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
