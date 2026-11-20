# SPDX-License-Identifier: Apache-2.0
"""Real four-layer optimizer steps, gradient accumulation and updated-weight parity."""

import argparse
import csv
import json
import traceback
from pathlib import Path

import torch
from rl_engine.models.deepseek_v41.model import DeepSeekV41TextModel
from rl_engine.models.deepseek_v41.bitwise import compare_bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {
        "status": "RUNNING",
        "scope": "real original four layers; frozen quantized weights/embedding/head",
        "steps": 2,
        "seeds": [17, 1234],
        "batch_sizes": [1, 2],
        "sequence_lengths": [7, 8],
        "mode_parity": "shared-kernel parity",
        "optimizer": "SGD lr=1e-5 foreach=False",
        "gradient_accumulation": "two distinct prompt microbatches; sum of NLL means divided by 2",
    }

    def check(category, a, b):
        assert a.keys() == b.keys()
        for name in a:
            r = compare_bits(a[name], b[name], operator=name)
            r["category"] = category
            rows.append(r)
            if r["status"] != "PASS":
                (args.output / "first_mismatch.json").write_text(json.dumps(r, indent=2) + "\n")
                raise AssertionError(r)

    try:
        torch.use_deterministic_algorithms(True)
        model = DeepSeekV41TextModel.from_checkpoint(args.checkpoint)
        cases = json.loads((args.checkpoint / "cases.json").read_text())
        micro = [
            torch.tensor([cases[i]["input_ids"][:n]], device="cuda") for i, n in ((0, 7), (1, 8))
        ]
        batch = torch.tensor([case["input_ids"][:8] for case in cases[:2]], device="cuda")
        params = dict(model.named_parameters())
        initial = {k: p.detach().clone() for k, p in params.items()}
        for seed in (17, 1234):
            expected = []
            for replay in range(2):
                with torch.no_grad():
                    for name, p in params.items():
                        p.copy_(initial[name])
                torch.manual_seed(seed)
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, foreach=False)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    model.train()
                    for ids in micro:
                        out = model(ids)
                        (-out.raw_token_logprobs.mean() / len(micro)).backward()
                    del out
                    gradients = {
                        k: p.grad.detach().clone() for k, p in params.items() if p.grad is not None
                    }
                    assert gradients and all(
                        bool(torch.isfinite(v).all()) for v in gradients.values()
                    )
                    optimizer.step()
                    weights = {k: p.detach().clone() for k, p in params.items()}
                    if step == 0:
                        changed = sum(
                            not torch.equal(
                                v.contiguous().view(torch.uint8),
                                initial[k].contiguous().view(torch.uint8),
                            )
                            for k, v in weights.items()
                        )
                        assert changed > 0, "optimizer failed to update any parameter"
                        summary["updated_parameter_tensors_step0"] = changed
                    model.train()
                    out = model(batch, collect_trace=True)
                    trained = out.trace
                    del out
                    model.eval()
                    with torch.inference_mode():
                        out = model(batch, collect_trace=True)
                    check(
                        f"seed{seed}.repeat{replay}.step{step}.updated_weight_modes",
                        out.trace,
                        trained,
                    )
                    model.train()
                    again = model(batch, collect_trace=True)
                    check(
                        f"seed{seed}.repeat{replay}.step{step}.updated_weight_replay",
                        again.trace,
                        trained,
                    )
                    del again, out
                    if replay == 0:
                        expected.append((gradients, weights, trained))
                    else:
                        eg, ew, et = expected[step]
                        check(f"seed{seed}.step{step}.gradient_accumulation_replay", gradients, eg)
                        check(f"seed{seed}.step{step}.optimizer_weight_replay", weights, ew)
                        check(f"seed{seed}.step{step}.whole_training_sequence_replay", trained, et)
                    print("OPTIMIZER_MODEL_STEP_PASS", seed, replay, step, flush=True)
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
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "comparisons.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output / "consistency.csv").open("w") as stream:
        fields = [
            "category",
            "operator",
            "dtype",
            "compared_elements",
            "mismatched_elements",
            "status",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    print("OPTIMIZER_MODEL_RESULT", json.dumps(summary), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
