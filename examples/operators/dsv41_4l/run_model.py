# SPDX-License-Identifier: Apache-2.0
"""Real original-layer model forward/VJP, replay and shared-kernel mode parity."""

import argparse
import csv
import datetime
import json
import os
import time
import traceback
from pathlib import Path

import torch
from .gpu_visibility import rank_assignment
import torch.distributed as dist
from rl_engine.models.deepseek_v41.parallel import AttentionTP
from rl_engine.models.deepseek_v41.model import DeepSeekV41TextModel
from rl_engine.models.deepseek_v41.bitwise import compare_bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2, 4))
    args = parser.parse_args()
    gpu_assignment = rank_assignment()
    tp = AttentionTP()
    if args.tp > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=8))
        tp = AttentionTP(args.tp, dist.get_rank(), dist.group.WORLD)
        args.output = args.output / f"tp-rank{tp.rank}"
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {
        "status": "RUNNING",
        "gpu_assignment": gpu_assignment,
        "scope": "real checkpoint original layers 0,1,2,3 text model",
        "mode_parity": "shared-kernel parity, not independent backends",
        "training_profile": DeepSeekV41TextModel.profile,
        "topology": {"tp": args.tp, "pp": 1, "dp": 1},
        "tp_scope": "attention heads/output groups sharded; FFN replicated",
        "full_40_layers": "NOT RUN",
        "independent_model_oracle": "NOT RUN",
        "frozen_parameters": ["packed FP8/FP4 weights and scales", "token embedding", "FP32 head"],
        "backward_contract": (
            "identity activation-quantization STE; discrete index/dispatch selection"
        ),
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "comparisons.json").write_text(json.dumps(rows, indent=2) + "\n")
        with (args.output / "consistency.csv").open("w") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "category",
                    "operator",
                    "dtype",
                    "compared_elements",
                    "mismatched_elements",
                    "status",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in writer.fieldnames})

    def compare(category, actual, expected):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{category}: tensor keys differ")
        for key in actual:
            result = compare_bits(
                actual[key],
                expected[key],
                operator=key,
                rank=tp.rank,
                token_dims=2 if key in ("hidden", "logits", "raw_token_logprobs") else None,
            )
            result["category"] = category
            rows.append(result)
            if result["status"] != "PASS":
                (args.output / "first_mismatch.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print("FIRST_MISMATCH", json.dumps(result), flush=True)
                raise AssertionError(f"{category}: {key} bit mismatch")

    try:
        torch.manual_seed(1234)
        torch.use_deterministic_algorithms(True)
        summary["device"] = torch.cuda.get_device_name()
        cases = json.loads((args.checkpoint / "cases.json").read_text())
        ids = torch.tensor(
            [cases[args.case]["input_ids"][: args.tokens]], device="cuda", dtype=torch.int64
        )
        summary["input_ids"] = ids.tolist()
        summary["checkpoint_revision"] = json.loads(
            (args.checkpoint / "manifest.json").read_text()
        )["revision"]
        print("LOAD_START", flush=True)
        start = time.monotonic()
        model = DeepSeekV41TextModel.from_checkpoint(args.checkpoint, attention_tp=tp)
        summary["load_seconds"] = time.monotonic() - start
        print("LOAD_COMPLETE", summary["load_seconds"], flush=True)
        gradients = []
        outputs = []
        for repeat in range(2):
            model.train()
            model.zero_grad(set_to_none=True)
            start = time.monotonic()
            out = model(ids, collect_trace=True)
            torch.cuda.synchronize()
            print("FORWARD_COMPLETE", repeat, time.monotonic() - start, flush=True)
            if not bool(
                torch.isfinite(out.logits).all() & torch.isfinite(out.raw_token_logprobs).all()
            ):
                raise AssertionError("model produced nonfinite logits/logprobs")
            loss = -out.raw_token_logprobs.mean()
            loss.backward()
            torch.cuda.synchronize()
            print(
                "BACKWARD_COMPLETE",
                repeat,
                time.monotonic() - start,
                "loss",
                float(loss),
                flush=True,
            )
            grad = {
                k: p.grad.detach().clone()
                for k, p in model.named_parameters()
                if p.grad is not None
            }
            if not grad or any(not bool(torch.isfinite(g).all()) for g in grad.values()):
                raise AssertionError("missing or nonfinite parameter gradients")
            summary[f"repeat_{repeat}"] = {
                "seconds": time.monotonic() - start,
                "loss": float(loss),
                "gradient_parameters": len(grad),
                "gradient_elements": sum(g.numel() for g in grad.values()),
                "parameters_without_gradient": [
                    k for k, p in model.named_parameters() if p.requires_grad and p.grad is None
                ],
            }
            gradients.append(grad)
            outputs.append(out.trace)
            del out, loss
        compare("deterministic_forward", outputs[1], outputs[0])
        compare("deterministic_backward", gradients[1], gradients[0])
        torch.save(
            {
                "input_ids": ids.cpu(),
                "trace": {k: v.cpu() for k, v in outputs[0].items()},
                "gradients": {k: v.cpu() for k, v in gradients[0].items()},
            },
            args.output / "baseline.pt",
        )
        del gradients
        model.eval()
        with torch.inference_mode():
            out = model(ids, collect_trace=True)
        compare("train_inference_shared_kernel", out.trace, outputs[0])
        torch.save(
            {
                "input_ids": ids.cpu(),
                "hidden": out.hidden.cpu(),
                "logits": out.logits.cpu(),
                "raw_fp32_token_logprobs": out.raw_token_logprobs.cpu(),
            },
            args.output / "outputs.pt",
        )
        summary.update(
            status="PASS",
            forward="PASS",
            backward="PASS",
            deterministic_replay="PASS",
            shared_kernel_train_inference_parity="PASS",
            compared_elements=sum(r["compared_elements"] for r in rows),
            mismatched_elements=sum(r["mismatched_elements"] for r in rows),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        )
    except Exception as error:
        summary.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    print("MODEL_RESULT", json.dumps(summary), flush=True)
    if args.tp > 1:
        if summary["status"] == "PASS":
            dist.barrier()
        dist.destroy_process_group()
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
