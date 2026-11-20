# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint odd prefill + multi-step decode with training across caches."""

import argparse
import json
import traceback
from pathlib import Path
import torch
from .gpu_visibility import rank_assignment
from rl_engine.models.deepseek_v41.model import DeepSeekV41TextModel
from rl_engine.models.deepseek_v41.bitwise import compare_bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpu_assignment = rank_assignment()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    summary = {
        "status": "RUNNING",
        "gpu_assignment": gpu_assignment,
        "scope": "real checkpoint original layers 0,1,2,3 odd prefill and decode",
        "mode_parity": "shared-kernel parity",
        "chunked_vs_monolithic_bit_parity": "NOT RUN",
    }

    def check(category, a, b):
        assert a.keys() == b.keys()
        for key in a:
            row = compare_bits(a[key], b[key], operator=key)
            row["category"] = category
            records.append(row)
            if row["status"] != "PASS":
                (args.output / "first_mismatch.json").write_text(json.dumps(row, indent=2) + "\n")
                raise AssertionError(row)

    try:
        torch.use_deterministic_algorithms(True)
        model = DeepSeekV41TextModel.from_checkpoint(args.checkpoint)
        cases = json.loads((args.checkpoint / "cases.json").read_text())
        for case in cases:
            ids = torch.tensor([case["input_ids"]], device="cuda")
            count = min(ids.shape[1] - 1, 8)

            def run(training, ids=ids, count=count):
                model.train(training)
                model.zero_grad(set_to_none=True)
                model.reset()
                traces = {}
                loss = 0
                for start, stop in [(0, 3), *[(p, p + 1) for p in range(3, count)]]:
                    output = model(
                        ids[:, start:stop],
                        start_pos=start,
                        target_ids=ids[:, start + 1 : stop + 1],
                        collect_trace=True,
                    )
                    assert output.raw_token_logprobs.dtype == torch.float32
                    assert bool(torch.isfinite(output.raw_token_logprobs).all())
                    for key, value in output.trace.items():
                        traces[f"positions.{start}-{stop}.{key}"] = value
                    loss = loss - output.raw_token_logprobs.sum() / count
                if training:
                    loss.backward()
                    for key, p in model.named_parameters():
                        if p.grad is not None:
                            assert bool(torch.isfinite(p.grad).all())
                            traces["gradient." + key] = p.grad.detach().clone()
                return traces

            a = run(True)
            b = run(True)
            check(case["name"] + ".replay", a, b)
            del b
            with torch.inference_mode():
                inf = run(False)
            check(case["name"] + ".shared_kernel_modes", inf, {key: a[key] for key in inf})
            print("STATE_CASE_PASS", case["name"], count, flush=True)
            del a, inf
        summary.update(
            status="PASS",
            cases=len(cases),
            forward="PASS",
            backward="PASS",
            compared_elements=sum(r["compared_elements"] for r in records),
            mismatched_elements=sum(r["mismatched_elements"] for r in records),
        )
    except Exception as error:
        summary.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "comparisons.json").write_text(json.dumps(records, indent=2) + "\n")
    print("STATE_RESULT", json.dumps(summary), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
