# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare the DSv4 model with DeepSeek's official inference code, layer by layer.

A semantic (wiring) check, not a byte check: the reference runs its own TileLang
kernels and FP8/FP4 simulation. The two differ in rounding, so the per-layer errors
should be small (around 1e-2 relative) and should not grow abruptly. A wrong
weight, fusion order or RoPE base shows up as an O(1) error at the layer where it
enters.

Keep ``--tokens`` <= 4 * index_topk (2048): then the indexer selects every compressed
entry on both sides, and its scoring precision cannot change the result.

    python scripts/check_dsv4_model_ref.py --ckpt ~/dsv4-toy --ref <dir with model.py, kernel.py>

The reference imports ``fast_hadamard_transform`` for the indexer's rotation; when it
is missing, an exact PyTorch Walsh-Hadamard transform stands in.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

import torch


def _hadamard_stub() -> None:
    try:
        import fast_hadamard_transform  # noqa: F401

        return
    except ImportError:
        pass

    def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        n = x.shape[-1]
        h = torch.ones(1, 1, dtype=torch.float32, device=x.device)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        return ((x.float() @ h) * scale).to(x.dtype)

    mod = types.ModuleType("fast_hadamard_transform")
    mod.hadamard_transform = hadamard_transform
    sys.modules["fast_hadamard_transform"] = mod


def _reference(ref_dir: pathlib.Path, ckpt: pathlib.Path, cfg: dict, tokens: int):
    _hadamard_stub()
    sys.path.insert(0, str(ref_dir))
    import model as ref  # DeepSeek's inference/model.py

    from rl_engine.models.deepseek_v4.loader import SafetensorsDir

    rs = cfg["rope_scaling"]
    args = ref.ModelArgs(
        max_batch_size=1,
        max_seq_len=max(tokens, 256),
        dtype="fp8",
        scale_fmt="ue8m0",
        expert_dtype=cfg.get("expert_dtype"),
        scale_dtype="fp8",
        vocab_size=cfg["vocab_size"],
        dim=cfg["hidden_size"],
        moe_inter_dim=cfg["moe_intermediate_size"],
        n_layers=cfg["num_hidden_layers"],
        n_hash_layers=cfg["num_hash_layers"],
        n_mtp_layers=0,
        n_heads=cfg["num_attention_heads"],
        n_routed_experts=cfg["n_routed_experts"],
        n_shared_experts=cfg["n_shared_experts"],
        n_activated_experts=cfg["num_experts_per_tok"],
        score_func=cfg["scoring_func"],
        route_scale=cfg["routed_scaling_factor"],
        swiglu_limit=cfg["swiglu_limit"],
        q_lora_rank=cfg["q_lora_rank"],
        head_dim=cfg["head_dim"],
        rope_head_dim=cfg["qk_rope_head_dim"],
        norm_eps=cfg["rms_norm_eps"],
        o_groups=cfg["o_groups"],
        o_lora_rank=cfg["o_lora_rank"],
        window_size=cfg["sliding_window"],
        compress_ratios=tuple(cfg["compress_ratios"][: cfg["num_hidden_layers"]]),
        compress_rope_theta=cfg["compress_rope_theta"],
        original_seq_len=rs["original_max_position_embeddings"],
        rope_theta=cfg["rope_theta"],
        rope_factor=rs["factor"],
        beta_fast=rs["beta_fast"],
        beta_slow=rs["beta_slow"],
        index_n_heads=cfg["index_n_heads"],
        index_head_dim=cfg["index_head_dim"],
        index_topk=cfg["index_topk"],
        hc_mult=cfg["hc_mult"],
        hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"],
        hc_eps=cfg["hc_eps"],
    )
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        m = ref.Transformer(args)
    torch.set_default_dtype(torch.float32)
    src = SafetensorsDir(ckpt)
    state = m.state_dict()
    for name, p in state.items():
        if name not in src:
            continue
        t = src[name]
        if p.dtype in (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, torch.float8_e4m3fn):
            p.view(torch.uint8).copy_(t.view(torch.uint8))
        elif t.dtype == torch.float8_e4m3fn:
            # A BF16 parameter over an FP8 checkpoint tensor (wo_a): dequantize, as
            # DeepSeek's convert script does.
            s = src[name[: -len(".weight")] + ".scale"].float()
            n, k = t.shape
            w = t.float().view(n // 128, 128, k // 128, 128) * s[:, None, :, None]
            p.copy_(w.view(n, k))
        else:
            p.copy_(t)
    missing = [n for n in state if n not in src and not n.endswith(("kv_cache", "freqs_cis"))]
    missing = [n for n in missing if "_state" not in n]
    if missing:
        raise RuntimeError(f"reference parameters not in the checkpoint: {missing[:8]}")
    return ref, m


@torch.inference_mode()
def _reference_streams(m, ids: torch.Tensor):
    # The reference allocates outputs and index tables with the defaults its __main__ sets.
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    x = ids[None]
    h = m.embed(x).unsqueeze(2).repeat(1, 1, m.hc_mult, 1)
    streams = []
    for layer in m.layers:
        h = layer(h, 0, x)
        streams.append(h[0])
    out = m.norm(m.head.hc_head(h, m.hc_head_fn, m.hc_head_scale, m.hc_head_base))
    return out[0], streams


def _err(a: torch.Tensor, b: torch.Tensor) -> str:
    a, b = a.float(), b.float()
    rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
    cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    return f"rel L2 {rel:.3e}  cos {cos:.6f}  max|d| {float((a - b).abs().max()):.3e}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=pathlib.Path, required=True)
    p.add_argument("--ref", type=pathlib.Path, required=True)
    p.add_argument("--tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--window-rope",
        choices=("vllm", "plain"),
        default="vllm",
        help="RoPE of our sliding-window-only layers: vLLM's YaRN cache (default) or plain "
        "RoPE as in the reference and Megatron-Core",
    )
    args = p.parse_args()
    if args.tokens > 2048:
        raise SystemExit("--tokens must be <= 2048 (see the module docstring)")

    from rl_engine.models.deepseek_v4.attention import Dsv4Batch
    from rl_engine.models.deepseek_v4.loader import load_model

    cfg = json.loads((args.ckpt / "config.json").read_text())
    ours = load_model(args.ckpt, max_model_len=4096, load_head=False)
    if args.window_rope == "plain":
        dim, base = cfg["qk_rope_head_dim"], cfg["rope_theta"]
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        ang = torch.outer(torch.arange(4096, dtype=torch.float32), inv)
        plain = torch.cat([ang.cos(), ang.sin()], -1).cuda()
        for layer in ours.layers:
            if layer.attn.c.compress_ratio == 1:
                layer.attn.cos_sin = plain
    gen = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(cfg["vocab_size"], (args.tokens,), generator=gen).cuda()
    batch = Dsv4Batch.from_lens([args.tokens], [args.tokens], cfg["sliding_window"])
    out, streams = ours.forward_batch(ids, batch, ours.new_caches(1, 4096), return_streams=True)
    del ours
    torch.cuda.empty_cache()

    _, m = _reference(args.ref, args.ckpt, cfg, args.tokens)
    r_out, r_streams = _reference_streams(m, ids)
    print(f"tokens {args.tokens}, window-layer RoPE: {args.window_rope}")
    for i, (a, b) in enumerate(zip(streams, r_streams)):
        print(f"layer {i} (ratio {max(1, cfg['compress_ratios'][i]):>3}) streams: {_err(a, b)}")
    print(f"final hidden (hc_head + norm):    {_err(out, r_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
