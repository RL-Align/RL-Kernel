# SPDX-License-Identifier: Apache-2.0
"""Real PP2 (optionally DP2) model, explicit residual/delayed-pre P2P VJP.

This is model execution, not a collective-only test. The tested PP cut is
between original layers 1 and 2, so KV/index source 2 and consumer 3 coexist.
"""

import argparse
import csv
import datetime
import json
import os
import traceback
from pathlib import Path

import torch
from .gpu_visibility import rank_assignment
import torch.distributed as dist
from rl_engine.models.deepseek_v41.model import DeepSeekV41TextModel
from rl_engine.models.deepseek_v41.parallel import AttentionTP
from rl_engine.models.deepseek_v41.bitwise import compare_bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dp", type=int, default=1, choices=(1, 2))
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument("--optimizer-steps", type=int, default=0, choices=(0, 2))
    parser.add_argument("--second-baseline", type=Path)
    args = parser.parse_args()
    gpu_assignment = rank_assignment()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=8))
    if dist.get_world_size() != 2 * args.dp * args.tp:
        raise ValueError("world size must equal TP*PP2*DP")
    stage = (rank // args.tp) % 2
    tp_rank = rank % args.tp
    replica = rank // (2 * args.tp)
    peer = rank + args.tp if stage == 0 else rank - args.tp
    tp_groups = [
        dist.new_group(
            list(range(d * 2 * args.tp + s * args.tp, d * 2 * args.tp + (s + 1) * args.tp))
        )
        for d in range(args.dp)
        for s in range(2)
    ]
    tp = AttentionTP(args.tp, tp_rank, tp_groups[replica * 2 + stage])
    dp_groups = [
        dist.new_group([d * 2 * args.tp + s * args.tp + t for d in range(args.dp)])
        for s in range(2)
        for t in range(args.tp)
    ]
    if args.tp > 1:
        args.baseline = args.baseline / f"tp-rank{tp_rank}" / "baseline.pt"
    device = torch.device("cuda", local)
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=True)
    dp_expected = baseline["gradients"]
    if args.second_baseline is not None:
        if args.dp != 2 or args.tp != 1:
            raise ValueError("distinct-prompt baselines currently require TP1 PP2 DP2")
        second = torch.load(args.second_baseline, map_location="cpu", weights_only=True)
        if baseline["gradients"].keys() != second["gradients"].keys():
            raise ValueError("replica baselines have different gradient parameter sets")
        dp_expected = {
            key: (value + second["gradients"][key]) / 2
            for key, value in baseline["gradients"].items()
        }
        if replica == 1:
            baseline = second
    ids = baseline["input_ids"].to(device)
    b, s = ids.shape
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {
        "rank": rank,
        "status": "RUNNING",
        "gpu_assignment": gpu_assignment,
        "tp": args.tp,
        "tp_scope": "attention heads/output groups sharded; FFN replicated",
        "pp": 2,
        "dp": args.dp,
        "stage": stage,
        "mode_parity": "shared-kernel parity",
        "dp_batch": (
            "two distinct real-token prompts; equal mean-loss replica weighting"
            if args.second_baseline is not None
            else "identical real-token microbatch on each replica"
        ),
        "checkpoint": "real original layers 0,1,2,3",
        "independent_model_oracle": "NOT RUN",
    }

    def check(category, values, expected, subset=False):
        if not subset and values.keys() != expected.keys():
            raise AssertionError(f"{category}: tensor keys differ")
        for key, value in values.items():
            row = compare_bits(value, expected[key].to(device), rank=rank, operator=key)
            row["category"] = category
            rows.append(row)
            if row["status"] != "PASS":
                (args.output / f"first_mismatch-rank{rank}.json").write_text(
                    json.dumps(row, indent=2) + "\n"
                )
                raise AssertionError(f"{category}: {key}: bit mismatch")

    def execute(training, backward=True):
        boundary = None
        if stage == 1:
            x = torch.empty((b, s, 4, 5120), device=device, dtype=torch.bfloat16)
            pre = torch.empty((b, s, 4), device=device, dtype=torch.float32)
            dist.recv(x, peer)
            dist.recv(pre, peer)
            boundary = (x.requires_grad_(training), pre.requires_grad_(training))
        out = model(ids, collect_trace=True, boundary=boundary)
        if stage == 0:
            dist.send(out.residual.detach().contiguous(), peer)
            dist.send(out.previous_pre.detach().contiguous(), peer)
        if training and backward:
            if stage == 1:
                (-out.raw_token_logprobs.mean()).backward()
                dist.send(boundary[0].grad.contiguous(), peer)
                dist.send(boundary[1].grad.contiguous(), peer)
            else:
                dx = torch.empty_like(out.residual)
                dp = torch.empty_like(out.previous_pre)
                dist.recv(dx, peer)
                dist.recv(dp, peer)
                torch.autograd.backward((out.residual, out.previous_pre), (dx, dp))
        return out

    try:
        model = DeepSeekV41TextModel.from_checkpoint(
            args.checkpoint,
            layer_ids=((0, 1) if stage == 0 else (2, 3)),
            device=device,
            first_stage=stage == 0,
            last_stage=stage == 1,
            attention_tp=tp,
        )
        traces = []
        gradients = []
        for repeat in range(2):
            model.train()
            model.zero_grad(set_to_none=True)
            out = execute(True)
            torch.cuda.synchronize()
            grad = {
                k: p.grad.detach().clone()
                for k, p in model.named_parameters()
                if p.grad is not None
            }
            if any(not bool(torch.isfinite(g).all()) for g in grad.values()):
                raise AssertionError("nonfinite pipeline gradients")
            check(
                f"same_tp_unpartitioned_forward_{repeat}", out.trace, baseline["trace"], subset=True
            )
            check(
                f"same_tp_unpartitioned_backward_{repeat}", grad, baseline["gradients"], subset=True
            )
            if args.dp > 1:
                parameters = dict(model.named_parameters())
                for name, value in grad.items():
                    dist.all_reduce(value, group=dp_groups[stage * args.tp + tp_rank])
                    value.div_(args.dp)
                    parameters[name].grad.copy_(value)
                check(f"dp_gradient_mean_{repeat}", grad, dp_expected, subset=True)
            if args.tp > 1:
                # Q/sparse heads and wo_a/sink gradients are intentionally sharded.
                common = {
                    k: v
                    for k, v in out.trace.items()
                    if not k.endswith((".attention.q", ".attention.sparse"))
                }
                common_grad = {
                    k: v for k, v in grad.items() if not k.endswith((".attn.wo_a", ".attn.sink"))
                }
                for category, values in (
                    ("tp_replicated_state", common),
                    ("tp_replicated_gradient", common_grad),
                ):
                    expected = {}
                    for key, value in values.items():
                        other = value.clone()
                        dist.broadcast(
                            other, src=replica * 2 * args.tp + stage * args.tp, group=tp.group
                        )
                        expected[key] = other
                    check(category, values, expected)
            traces.append(out.trace)
            gradients.append(grad)
            print("PP_MODEL_REPEAT_COMPLETE", rank, repeat, flush=True)
        check("deterministic_forward", traces[1], traces[0])
        check("deterministic_backward", gradients[1], gradients[0])
        model.eval()
        with torch.inference_mode():
            out = execute(False)
        check("train_inference_shared_kernel", out.trace, traces[0])
        if stage == 1:
            torch.save(
                {"input_ids": ids.cpu(), "raw_fp32_token_logprobs": out.raw_token_logprobs.cpu()},
                args.output / f"logprobs-rank{rank}.pt",
            )
        if args.optimizer_steps:
            del out, traces, gradients
            parameters = dict(model.named_parameters())
            initial = {name: p.detach().clone() for name, p in parameters.items()}
            expected = []
            for replay in range(2):
                with torch.no_grad():
                    for name, p in parameters.items():
                        p.copy_(initial[name])
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, foreach=False)
                for step in range(args.optimizer_steps):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    out = execute(True)
                    del out
                    grad = {
                        name: p.grad.detach().clone()
                        for name, p in parameters.items()
                        if p.grad is not None
                    }
                    if any(not bool(torch.isfinite(value).all()) for value in grad.values()):
                        raise AssertionError("nonfinite optimizer-step gradient")
                    if args.dp > 1:
                        for name, value in grad.items():
                            dist.all_reduce(value, group=dp_groups[stage * args.tp + tp_rank])
                            value.div_(args.dp)
                            parameters[name].grad.copy_(value)
                    optimizer.step()
                    weights = {name: p.detach().clone() for name, p in parameters.items()}
                    if args.dp > 1:
                        synchronized = {}
                        for name, p in parameters.items():
                            dist.broadcast(
                                p.data,
                                src=stage * args.tp + tp_rank,
                                group=dp_groups[stage * args.tp + tp_rank],
                            )
                            synchronized[name] = p.detach().clone()
                        check(
                            f"optimizer.dp_weight_sync.repeat{replay}.step{step}",
                            synchronized,
                            weights,
                        )
                    if step == 0:
                        changed = sum(
                            not torch.equal(
                                value.contiguous().view(torch.uint8),
                                initial[name].contiguous().view(torch.uint8),
                            )
                            for name, value in weights.items()
                        )
                        if not changed:
                            raise AssertionError("no model parameter changed after optimizer step")
                        summary["updated_parameter_tensors_step0"] = changed
                    model.train()
                    trained = execute(True, backward=False).trace
                    model.eval()
                    with torch.inference_mode():
                        inference = execute(False).trace
                    check(
                        f"optimizer.updated_weight_modes.repeat{replay}.step{step}",
                        inference,
                        trained,
                    )
                    if replay == 0:
                        expected.append((grad, weights, trained))
                    else:
                        eg, ew, et = expected[step]
                        check(f"optimizer.gradient_replay.step{step}", grad, eg)
                        check(f"optimizer.parameter_replay.step{step}", weights, ew)
                        check(f"optimizer.sequence_replay.step{step}", trained, et)
                    print("PP_MODEL_OPTIMIZER_STEP_COMPLETE", rank, replay, step, flush=True)
            summary["optimizer_steps"] = "PASS"
            summary["optimizer_steps_per_replay"] = args.optimizer_steps
        else:
            summary["optimizer_steps"] = "NOT RUN"
        summary.update(
            status="PASS",
            forward="PASS",
            backward="PASS",
            compared_elements=sum(r["compared_elements"] for r in rows),
            mismatched_elements=sum(r["mismatched_elements"] for r in rows),
        )
    except Exception as error:
        summary.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (args.output / f"summary-rank{rank}.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / f"comparisons-rank{rank}.json").write_text(json.dumps(rows, indent=2) + "\n")
        with (args.output / f"consistency-rank{rank}.csv").open("w") as stream:
            fields = [
                "category",
                "operator",
                "rank",
                "dtype",
                "compared_elements",
                "mismatched_elements",
                "status",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in fields})
    print("PP_MODEL_RESULT", json.dumps(summary), flush=True)
    if summary["status"] == "PASS":
        dist.barrier()
    dist.destroy_process_group()
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
