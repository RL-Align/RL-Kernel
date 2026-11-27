# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare the DSv4.1 model with DeepSeek's reference inference code, layer by layer.

A wiring check, not a byte check. Both models load one random checkpoint in the release's
formats (the toy config: 7 layers holding every role, an Engram layer; experts at
F = 256 to fit both models on one GPU). The reference runs its TileLang kernels and its
quantization simulation (MXFP8 activations, an FP8 sliding-window KV with the RoPE dims
quantized too, FP4 compressed KV and indexer, design doc D11 / D12), so the per-layer
errors should be small and should not jump; a wrong weight, fusion, RoPE or wiring shows
up as an O(1) error where it enters.

Both sides round every dense layer's input to MXFP8 (D12 with activation rounding), so the
dense arithmetic agrees up to accumulation order. ``--match-storage`` turns off the
reference's simulation of the storage formats we do not follow (D11 / D15): the FP8
rounding of the sliding-window KV (RoPE dims included) and the FP4 rounding of the
compressed latent and the indexer q / k. What remains is our FP8 KV and indexer storage.

Keep ``--tokens`` <= index_topk (512): every compressed position is then selected on both
sides, so the indexers' precision cannot change the result.

    python scripts/check_dsv41_model_ref.py --meta <dir with tokenizer.json, inference/>
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch


def _err(a: torch.Tensor, b: torch.Tensor) -> str:
    a, b = a.float(), b.float()
    rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
    cos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    return f"rel L2 {rel:.3e}  cos {cos:.6f}  max|d| {float((a - b).abs().max()):.3e}"


class _Tok:
    """What DeepSeek's build_compressed_token_map reads from a tokenizer."""

    def __init__(self, path: pathlib.Path):
        from tokenizers import Tokenizer

        self.backend_tokenizer = Tokenizer.from_file(str(path))
        self._n = self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    def __len__(self) -> int:
        return self._n


def _reference(
    ref_dir: pathlib.Path, cfg, ckpt: dict, tokenizer, tokens: int, match_storage: bool = False
):
    sys.path.insert(0, str(ref_dir))
    import model as ref  # DeepSeek's inference/model.py

    if match_storage:
        aq = ref.act_quant

        def act_quant(x, *a, **k):  # the window KV is [b, s, d]; the output o is 4-D
            inplace = k.get("inplace", a[3] if len(a) > 3 else False)
            return x if inplace and x.dim() == 3 else aq(x, *a, **k)

        def fp4_act_quant(x, *a, **k):
            return x

        ref.act_quant, ref.fp4_act_quant = act_quant, fp4_act_quant

    r = cfg.rope_for_layer(next(i for i in range(cfg.num_hidden_layers) if cfg.compress_ratio(i)))
    n = cfg.num_hidden_layers
    args = ref.ModelArgs(
        max_batch_size=1,
        max_seq_len=max(tokens, 256),
        dtype="fp8",
        expert_dtype="fp4",
        vocab_size=cfg.vocab_size,
        dim=cfg.hidden_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_layers=n,
        n_mtp_layers=0,
        n_heads=cfg.num_attention_heads,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=cfg.n_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok,
        score_func=cfg.scoring_func,
        route_scale=cfg.routed_scaling_factor,
        swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora_rank,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        norm_eps=cfg.rms_norm_eps,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora_rank,
        window_size=cfg.sliding_window,
        compress_ratios=tuple(cfg.compress_ratios[:n]),
        kv_source_layers=cfg.kv_source_layer_ids,
        index_source_layers=cfg.index_source_layer_ids,
        compress_rope_theta=cfg.compress_rope_theta,
        original_seq_len=r["original_max_position_embeddings"],
        rope_theta=cfg.rope_theta,
        rope_factor=r["factor"],
        beta_fast=r["beta_fast"],
        beta_slow=r["beta_slow"],
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk,
        candidate_source_layer=cfg.candidate_source_layer_id,
        candidate_topk_blocks=cfg.candidate_topk_blocks,
        candidate_block_size=cfg.candidate_block_size,
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        engram_layer_ids=cfg.engram_layer_ids,
        engram_num_embeddings=cfg.engram_num_embeddings,
        engram_max_ngram_size=cfg.engram_max_ngram_size,
        engram_vocab_size=cfg.engram_vocab_size,
        engram_n_heads=cfg.engram_n_heads,
        engram_head_dim=cfg.engram_head_dim,
        engram_pad_id=cfg.engram_pad_token_id,
        engram_compressed_vocab_size=cfg.engram_compressed_vocab_size,
        vision_n_layers=0,
    )
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        m = ref.Transformer(args, tokenizer)
    torch.set_default_dtype(torch.float32)
    state = m.state_dict()
    missing = []
    for name, p in state.items():
        src_name = name.replace(".bias_vl", ".bias")  # text-only: the VL bias is unused
        if src_name not in ckpt:
            missing.append(name)
            continue
        t = ckpt[src_name]
        if p.dtype in (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, torch.float8_e4m3fn):
            p.view(torch.uint8).copy_(t.view(torch.uint8))
        elif t.dtype == torch.float8_e4m3fn:  # a BF16 parameter (wo_a): convert.py dequantizes
            from rl_engine.models.deepseek_v41.loader import mxfp8_dequant

            p.copy_(mxfp8_dequant(t, ckpt[src_name[: -len(".weight")] + ".scale"]))
        else:
            p.copy_(t)
    if missing:
        raise RuntimeError(f"reference parameters not in the checkpoint: {missing[:8]}")
    return ref, m


@torch.inference_mode()
def _reference_streams(ref, m, ids: torch.Tensor):
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    x = ids[None]
    hashes = m.engram_hash(x, 0, None) if m.engram_hash is not None else None
    h = m.embed(x).unsqueeze(2).repeat(1, 1, m.hc_mult, 1)
    pre = ref.make_identity_pre_mix(h, m.hc_mult)
    streams = []
    for layer in m.layers:
        if layer.engram is not None:
            h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
        h, pre = layer(h, 0, pre, None)
        streams.append(h[0])
    out = m.norm(m.layers[-1].hc_pre(h, pre))
    torch.set_default_device("cpu")
    return out[0], streams


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--meta", type=pathlib.Path, required=True)
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--moe-intermediate", type=int, default=256)
    p.add_argument("--match-storage", action="store_true")
    args = p.parse_args()

    from rl_engine.models.deepseek_v41 import Dsv41Config, toy_config
    from rl_engine.models.deepseek_v41.attention import Dsv41Batch
    from rl_engine.models.deepseek_v41.engram import build_compressed_token_map, ngram_windows
    from rl_engine.models.deepseek_v41.loader import load_model, random_checkpoint

    cfg = Dsv41Config.from_dict({**toy_config(), "moe_intermediate_size": args.moe_intermediate})
    if args.tokens > cfg.index_topk:
        raise SystemExit(f"--tokens must be <= {cfg.index_topk} (see the module docstring)")
    ckpt = random_checkpoint(cfg, seed=args.seed)
    token_map, size = build_compressed_token_map(str(args.meta / "tokenizer.json"))
    ours = load_model(ckpt, cfg, max_model_len=4096, load_head=False, token_map=token_map)
    gen = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(cfg.vocab_size, (args.tokens,), generator=gen).cuda()
    batch = Dsv41Batch.from_lens([args.tokens], [args.tokens], cfg.sliding_window)
    pos = torch.arange(args.tokens, device="cuda")
    windows = ngram_windows([ids], [pos], cfg.engram_max_ngram_size)
    out, streams = ours.forward_batch(
        ids, batch, ours.new_caches(1, 4096), engram_windows=windows, return_streams=True
    )
    del ours
    torch.cuda.empty_cache()

    tok = _Tok(args.meta / "tokenizer.json")
    ref, m = _reference(args.meta / "inference", cfg, ckpt, tok, args.tokens, args.match_storage)
    r_out, r_streams = _reference_streams(ref, m, ids)
    print(f"tokens {args.tokens}, compressed vocabulary {size}")
    for i, (a, b) in enumerate(zip(streams, r_streams)):
        role = cfg.layer_role(i)
        print(f"layer {i} ({role.mode:7s} r{role.ratio}) streams: {_err(a, b)}")
    print(f"final hidden (collapse + norm):   {_err(out, r_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
