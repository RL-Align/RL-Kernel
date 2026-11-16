# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The ``rlkernel`` verl engine (``rl_engine/integrations/dsv4/verl_engine.py``) on verl's
data contract, without Ray: data-parallel ranks under torchrun, each with its own batch.

1. Log-probs are batch-invariant: one micro-batch, many micro-batches, and each sequence
   alone through ``Dsv4Model`` give the same bytes; per-sample temperatures too.
2. Training steps (verl's ``train_batch``) reach every adapter, keep the ranks' adapters
   identical, and change the log-probs; ``disable_adapter`` gives the base model's bytes.
3. After training, (1) still holds (nonzero adapters).
4. ``get_per_tensor_param`` exports the adapters' bytes; a checkpoint round trip restores
   the log-probs byte for byte.

    torchrun --nproc-per-node 2 scripts/check_dsv4_verl_engine.py --model <dsv4 checkpoint>
"""

from __future__ import annotations

import argparse
import os
import tempfile
import types

import torch
import torch.distributed as dist


def _nested(xs):
    return torch.nested.as_nested_tensor(xs, layout=torch.jagged)


def make_batch(lens, vocab, seed, temperature, max_tokens, entropy=True):
    from verl.utils import tensordict_utils as tu

    gen = torch.Generator().manual_seed(seed)
    ids = [torch.randint(vocab, (n,), generator=gen) for n in lens]
    pos = [torch.arange(n) for n in lens]
    mask = [(torch.arange(n) >= n // 2).to(torch.int64) for n in lens]
    tensors = {"input_ids": _nested(ids), "position_ids": _nested(pos), "loss_mask": _nested(mask)}
    meta = {
        "use_dynamic_bsz": True,
        "max_token_len_per_gpu": max_tokens,
        "use_remove_padding": True,
        "calculate_entropy": entropy,
    }
    if isinstance(temperature, list):
        tensors["temperature"] = torch.tensor(temperature, dtype=torch.float32)
    else:
        meta["temperature"] = temperature
    return tu.get_tensordict(tensor_dict=tensors, non_tensor_dict=meta), ids


def infer(engine, td):
    with engine.eval_mode():
        out = engine.infer_batch(td, loss_function=None)
    return [t.clone() for t in out["model_output"]["log_probs"].unbind()], out


def loss_fn(model_output, data, dp_group=None):
    lp = model_output["log_probs"].values()
    mask = data["loss_mask"].values().to(lp.dtype)
    loss = -(lp * mask).sum() / data["batch_num_tokens"] * data["dp_size"]
    return loss, {}


def same(a, b):
    """Bitwise, per sequence, without the last position: its target (the next token of the
    packed batch) belongs to the following sequence, and verl never reads it."""
    return all(
        torch.equal(x[:-1].view(torch.int32), y[:-1].view(torch.int32)) for x, y in zip(a, b)
    )


def alone(engine, ids, temps):
    """Each sequence alone through Dsv4Model (one request, full prefill)."""
    model = engine.model
    out = []
    for x, t in zip(ids, temps):
        x = x.cuda()
        caches = model.new_caches(1, x.numel(), seed=11)
        hidden = model.forward(x, [x.numel()], [x.numel()], caches)
        out.append(model.selected_logprob(hidden, torch.roll(x, -1), t)[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/dsv4-work/dsv4-toy256"))
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    from verl.workers.config.optimizer import FSDPOptimizerConfig

    from rl_engine.integrations.dsv4.verl_engine import RlkEngineConfig, RlkernelEngine

    model_config = types.SimpleNamespace(
        path=args.model, local_path=args.model, lora_rank=0, lora={}
    )
    engine = RlkernelEngine(
        model_config=model_config,
        engine_config=RlkEngineConfig(lora_rank=args.rank, lora_alpha=32, lora_seed=3),
        optimizer_config=FSDPOptimizerConfig(lr=2e-3, total_training_steps=10, clip_grad=1.0),
        checkpoint_config=None,
    )
    engine.initialize()
    vocab = engine.model.config.vocab_size
    lens = [300, 140, 517, 64, 1000, 33] if rank == 0 else [77, 410, 256, 900, 129]
    T = 0.8
    ok = True

    def check(name, cond):
        nonlocal ok
        flag = torch.tensor([1 if cond else 0], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        good = bool(flag.item())
        ok &= good
        if rank == 0:
            print(f"{'PASS' if good else 'FAIL'}  {name}")

    def invariance(tag):
        one, out = infer(engine, make_batch(lens, vocab, 100 + rank, T, 1 << 16)[0])
        many, _ = infer(engine, make_batch(lens, vocab, 100 + rank, T, 1024)[0])
        td, ids = make_batch(lens, vocab, 100 + rank, T, 1 << 16)
        ref = alone(engine, ids, [T] * len(lens))
        check(f"[{tag}] one micro-batch == many micro-batches", same(one, many))
        check(f"[{tag}] packed == each sequence alone", same(one, ref))
        temps = [0.8, 1.0, 0.6, 1.0, 0.8, 1.3][: len(lens)]
        mixed, _ = infer(engine, make_batch(lens, vocab, 100 + rank, temps, 1 << 16)[0])
        check(f"[{tag}] per-sample temperatures", same(mixed, alone(engine, ids, temps)))
        ent = out["model_output"]["entropy"].values()
        good = bool(torch.isfinite(ent).all() and (ent > 0).all())
        check(f"[{tag}] entropy finite, positive", good)
        return one

    base = invariance("zero-B adapters")

    # training steps
    norms = []
    for step in range(3):
        td, _ = make_batch(lens, vocab, 200 + 10 * step + rank, T, 1100, False)
        with engine.train_mode():
            out = engine.train_batch(td, loss_function=loss_fn)
        engine.lr_scheduler_step()
        norms.append(out["metrics"]["grad_norm"])
    if rank == 0:
        print("grad norms", [round(n, 4) for n in norms])
    check("grad norms finite and nonzero", all(n == n and 0 < n < float("inf") for n in norms))
    params = engine.model.lora_parameters()
    nonzero = all(float(p.detach().float().abs().max()) > 0 for p in params)
    check("every adapter tensor nonzero", nonzero)
    flat = torch.cat([p.detach().flatten() for p in params])
    ref = flat.clone()
    dist.broadcast(ref, 0)
    identical = torch.equal(flat.view(torch.int16), ref.view(torch.int16))
    check("adapters identical on every rank", identical)

    trained = invariance("trained adapters")
    check("training changed the log-probs", not same(trained, base))
    with engine.disable_adapter():
        off, _ = infer(engine, make_batch(lens, vocab, 100 + rank, T, 1 << 16)[0])
    check("disable_adapter == base model", same(off, base))

    gen, peft = engine.get_per_tensor_param(base_sync_done=True)
    exported = dict(gen)
    check("export: 2 tensors per adapter", len(exported) == 2 * len(engine.adapters))
    check(
        "export: the adapters' bytes",
        all(
            torch.equal(exported[f"{n}.lora_A"], ad.a)
            and torch.equal(exported[f"{n}.lora_B"], ad.b)
            for n, ad in engine.adapters
        ),
    )
    check("export: peft config", peft["r"] == args.rank and peft["lora_alpha"] == 32)

    path = [tempfile.mkdtemp(prefix="rlk-ckpt-") if rank == 0 else None]
    dist.broadcast_object_list(path, 0)
    engine.save_checkpoint(path[0], global_step=3)
    td, _ = make_batch(lens, vocab, 999 + rank, T, 1100, False)
    with engine.train_mode():
        engine.train_batch(td, loss_function=loss_fn)
    moved, _ = infer(engine, make_batch(lens, vocab, 100 + rank, T, 1 << 16)[0])
    engine.load_checkpoint(path[0])
    back, _ = infer(engine, make_batch(lens, vocab, 100 + rank, T, 1 << 16)[0])
    check("checkpoint round trip", same(back, trained) and not same(moved, trained))

    if rank == 0:
        print("VERL ENGINE", "PASS" if ok else "FAIL")
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
