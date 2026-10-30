# SPDX-License-Identifier: Apache-2.0
"""Optional PyTorch reference smoke, NOT the T05/T06 production Triton kernels."""

from .contract import ContractError, CombinePlan, require
from .oracle import forward


def tensor_hex(tensor):
    import torch

    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes().hex()


def run_case(case, expected, device, graph=False, noncontiguous=False):
    import torch

    plan = CombinePlan.from_dict(case["plan"])
    plan.validate(plan.context)
    # Validate host input/profile before any GPU arithmetic.
    forward(plan, case["rows"], case["shared"], case["residual"], plan.context)
    n, h, p = len(plan.token_ids), plan.hidden_size, len(plan.inverse_map)
    token_index = {t: i for i, t in enumerate(plan.token_ids)}
    indices = [[p] * 6 for _ in range(n)]
    for i, (t, s, valid) in enumerate(plan.inverse_map):
        if valid:
            indices[token_index[t]][s] = i
    index = torch.tensor(indices, dtype=torch.int64, device=device).reshape(n, 6)
    masks = torch.tensor(plan.valid_slots, dtype=torch.bool, device=device).reshape(n, 6)

    def materialize(values, shape, dtype):
        t = torch.tensor(values, dtype=dtype, device=device).reshape(shape)
        return t.T.contiguous().T if noncontiguous else t

    rows = materialize(case["rows"] + [[0.0] * h], (p + 1, h), torch.bfloat16)
    shared = materialize(case["shared"], (n, h), torch.bfloat16)
    residual = materialize(case["residual"], (n, h), torch.bfloat16)
    dx = materialize(case["dx_rows"] + [[0.0] * h], (p + 1, h), torch.float32)
    dx_shared = materialize(case["dx_shared"], (n, h), torch.float32)

    def calculate(source, branch, residual_input=None):
        acc = torch.zeros((n, h), dtype=torch.float32, device=device)
        seen = torch.zeros((n, 1), dtype=torch.bool, device=device)
        partials = []
        canonical = []
        for s in range(6):
            value = source[index[:, s]].float()
            valid = masks[:, s : s + 1]
            canonical.append(torch.where(valid, value, torch.zeros_like(value)))
            next_acc = torch.where(seen, acc + value, value)
            acc = torch.where(valid, next_acc, acc)
            seen = seen | valid
            partials.append(acc.clone())
        routed = acc
        after_shared = routed + branch.float()
        if residual_input is None:
            return {
                "canonical_fp32": torch.stack(canonical, dim=1),
                "slot_partials_fp32": partials,
                "routed_fp32": routed,
                "output_fp32": after_shared,
            }
        precast = after_shared + residual_input.float()
        return {
            "canonical_fp32": torch.stack(canonical, dim=1),
            "slot_partials_fp32": partials,
            "routed_fp32": routed,
            "after_shared_fp32": after_shared,
            "precast_fp32": precast,
            "output_bf16": precast.to(torch.bfloat16),
        }

    def compare(actual, target):
        for key, values in actual.items():
            got = [tensor_hex(t) for t in values] if type(values) is list else tensor_hex(values)
            require(got == target[key], "BYTE_MISMATCH", f"{case['name']}:{key}")

    compare(calculate(rows, shared, residual), expected["forward"])
    compare(calculate(dx, dx_shared), expected["backward"])
    # Separate dy gather mock validates the same forward index on the device.
    dy = materialize(case["dy"], (n, h), torch.float32)
    gathered = [
        dy[token_index[t]] if valid else torch.zeros(h, dtype=torch.float32, device=device)
        for t, _, valid in plan.inverse_map
    ]
    gathered = (
        torch.stack(gathered)
        if gathered
        else torch.empty((0, h), dtype=torch.float32, device=device)
    )
    require(
        tensor_hex(gathered) == expected["gradient_dispatch_fp32"],
        "BYTE_MISMATCH",
        "gradient dispatch",
    )
    if graph:
        require(device.type == "cuda", "UNSUPPORTED_CAPABILITY", "CUDA Graph requires CUDA")
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                calculate(rows, shared, residual)
                calculate(dx, dx_shared)
        torch.cuda.current_stream(device).wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            captured_f = calculate(rows, shared, residual)
            captured_b = calculate(dx, dx_shared)
        for _ in range(2):
            g.replay()
            compare(captured_f, expected["forward"])
            compare(captured_b, expected["backward"])
        # Change fixed-address input to prove replay does not just expose stale output.
        changed = dict(case)
        changed["rows"] = [[-v for v in row] for row in case["rows"]]
        rows.copy_(materialize(changed["rows"] + [[0.0] * h], (p + 1, h), torch.bfloat16))
        expected_changed = forward(
            plan, changed["rows"], case["shared"], case["residual"], plan.context
        )
        g.replay()
        compare(captured_f, expected_changed["stages"])
        compare(captured_b, expected["backward"])
    return {
        "case": case["name"],
        "verdict": "REFERENCE_BYTES_PASS",
        "graph": graph,
        "noncontiguous": noncontiguous,
        "source_stride": list(rows.stride()),
    }


def conformance(records, device_name, require_h100=False, graph=False):
    try:
        import torch
    except ImportError as exc:
        raise ContractError(
            "UNSUPPORTED_CAPABILITY", "PyTorch is required for device reference smoke"
        ) from exc
    device = torch.device(device_name)
    require(device.type in ("cpu", "cuda"), "UNSUPPORTED_CAPABILITY", str(device))
    if require_h100:
        require(
            device.type == "cuda" and torch.cuda.is_available(),
            "UNSUPPORTED_CAPABILITY",
            "H100 required",
        )
        require(
            torch.version.hip is None and "H100" in torch.cuda.get_device_name(device),
            "UNSUPPORTED_CAPABILITY",
            "requested NVIDIA H100 not present",
        )
    if device.type == "cuda":
        require(torch.cuda.is_available(), "UNSUPPORTED_CAPABILITY", "CUDA unavailable")
        torch.cuda.set_device(device)
    provenance = {
        "provider": "pytorch-reference-only",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "compute_capability": list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None,
        "compiled_production_kernel": False,
    }
    outcomes = []
    with torch.no_grad():
        for record in records:
            for noncontiguous in (False, True):
                outcomes.append(
                    run_case(record["input"], record["expected"], device, graph, noncontiguous)
                )
    return provenance, outcomes
