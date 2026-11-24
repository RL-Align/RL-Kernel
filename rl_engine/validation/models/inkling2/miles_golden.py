# SPDX-License-Identifier: Apache-2.0
"""INK-03: debug capture from the pinned, real-weight Miles text model."""

import argparse
import hashlib
import inspect
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from rl_engine.models.inkling2.fingerprint import (
    HF_REVISION,
    MILES_REVISION,
    snapshot_fingerprint,
)
from rl_engine.validation.models.inkling2.visit_trace import VisitTrace, tensor_digest, visits


def checkpoint_digest(directory: Path) -> tuple[str, dict[str, str]]:
    """Stream the actual indexed checkpoint bytes, not just its config hash."""
    index = directory / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    filenames = sorted(set(weight_map.values()))
    if not filenames or any(Path(n).name != n or not n.endswith(".safetensors") for n in filenames):
        raise ValueError("invalid checkpoint shard names")
    digest = hashlib.sha256()
    for name in [index.name, *filenames]:
        part = hashlib.sha256()
        with (directory / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                part.update(chunk)
        digest.update(f"{name}\0{part.hexdigest()}\n".encode())
    return digest.hexdigest(), weight_map


@contextmanager
def capture_block_visits(model, trace: VisitTrace):
    """Hook actual module executions; repeated physical blocks get new visit IDs."""
    layers = list(model.layers.values())
    if len(layers) != 24 or tuple(model.core) != tuple(range(6, 18)) or model.num_passes != 3:
        raise ValueError("golden capture requires the released 24-layer fixed loop")
    if model.activation_checkpoint:
        raise ValueError("disable activation checkpointing for this debug capture")
    schedule = visits()
    cursor = 0
    grad_handles = []
    handles = []

    def hook_for(physical):
        def hook(module, args, output):
            nonlocal cursor
            if cursor >= len(schedule) or schedule[cursor].physical != physical:
                raise RuntimeError(f"unexpected physical layer {physical} at visit {cursor}")
            logical = cursor
            cursor += 1
            _, partial = output
            if not bool(torch.isfinite(partial).all()):
                raise RuntimeError(f"nonfinite partial at logical visit {logical}")
            provenance = dict(backend="miles-native", kernel="Inkling2Block.forward:boundary")
            trace.record(logical, "partial", partial, **provenance)
            if not partial.requires_grad:
                raise RuntimeError("golden capture requires autograd-enabled layer outputs")

            def grad_hook(grad):
                if not bool(torch.isfinite(grad).all()):
                    raise RuntimeError(f"nonfinite gradient at visit {logical}")
                trace.record(logical, "partial", grad, direction="backward", **provenance)

            grad_handles.append(partial.register_hook(grad_hook))

        return hook

    try:
        handles = [layer.register_forward_hook(hook_for(i)) for i, layer in enumerate(layers)]
        yield
        if cursor != 48:
            raise RuntimeError(f"incomplete forward: observed {cursor}/48 visits")
        backward = [r for r in trace.report()["records"] if r["direction"] == "backward"]
        if len(backward) != 48:
            raise RuntimeError(f"incomplete backward: observed {len(backward)}/48 visits")
    finally:
        for handle in handles + grad_handles:
            handle.remove()


def _check_miles_source(root: Path) -> None:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("rev-parse", "HEAD") != MILES_REVISION or git("status", "--porcelain", "-uno"):
        raise ValueError("Miles checkout must be clean and at the pinned source revision")


def load_miles_model(directory: Path, weight_map: dict[str, str], miles_root: Path, device):
    """Use Miles' model builder and HF state-dict adapter; reject missing parameters."""
    from safetensors import safe_open
    from torchtitan.config.parallelism import ParallelismConfig

    from miles.backends.torchtitan_utils.config import build_model_config

    _check_miles_source(miles_root)
    if not Path(inspect.getfile(build_model_config)).resolve().is_relative_to(miles_root.resolve()):
        raise RuntimeError("imported Miles does not come from the pinned checkout")
    args = SimpleNamespace(
        backend=SimpleNamespace(
            titan_model_name="inkling2", titan_model_flavor="looped", seq_length=256
        )
    )
    config = build_model_config(
        args, parallelism=ParallelismConfig(), hf_assets_path=str(directory)
    )
    model = config.build()
    adapter = config._owner.state_dict_adapter_cls(config, str(directory))
    hf = {}
    for name in sorted(set(weight_map.values())):
        with safe_open(directory / name, framework="pt", device="cpu") as shard:
            for key in sorted(k for k, shard_name in weight_map.items() if shard_name == name):
                if key.startswith("model.llm."):
                    value = shard.get_tensor(key)
                    if value.dtype not in (torch.bfloat16, torch.float32):
                        raise ValueError(f"non-BF16/FP32 text checkpoint tensor {key}")
                    hf[key] = value
    mapped = adapter.from_hf(hf)
    missing, unexpected = model.load_state_dict(mapped, strict=False, assign=True)
    if unexpected or any("tokens_per_expert" not in key for key in missing):
        raise RuntimeError(f"unmapped model state: missing={missing}, unexpected={unexpected}")
    if any(name not in mapped for name, _ in model.named_parameters()):
        raise RuntimeError("refusing an uninitialized parameter in golden capture")
    model = model.to(device).eval()
    model.activation_checkpoint = False
    return model


def _hardware(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-weight golden capture needs an actual GPU")
    props = torch.cuda.get_device_properties(device)
    if torch.version.hip is not None:
        arch = props.gcnArchName.split(":")[0]
        if arch != "gfx950":
            raise RuntimeError("initial ROCm golden requires MI350X/MI355X (gfx950)")
        return f"rocm:{arch}:{props.name}"
    return f"cuda:sm{props.major}{props.minor}:{props.name}"


def capture_checkpoint(
    directory: Path, miles_root: Path, inputs: dict, device: torch.device
) -> dict:
    """Single-device forward/backward golden; no optimizer or engine parity claim."""
    hardware = _hardware(device)
    fingerprint = snapshot_fingerprint(directory, revision=HF_REVISION, converted=True)
    weight_hash, weight_map = checkpoint_digest(directory)
    tokens, positions, targets = [
        torch.tensor(inputs[key], device=device) for key in ("tokens", "positions", "targets")
    ]
    if any(t.dtype != torch.int64 or t.ndim != 1 for t in (tokens, positions, targets)):
        raise ValueError("inputs must be one-dimensional integer arrays")
    if (
        not 1 <= tokens.numel() <= 256
        or tokens.shape != positions.shape
        or tokens.shape != targets.shape
    ):
        raise ValueError("expected 1-256 aligned tokens, positions and targets")
    if positions[0] != 0 or bool(
        ((positions[1:] != 0) & (positions[1:] != positions[:-1] + 1)).any()
    ):
        raise ValueError("packed positions must start at zero and increment or reset")
    if bool(((tokens < 0) | (tokens >= 200058) | (targets < 0) | (targets >= 200058)).any()):
        raise ValueError("golden inputs require valid vocabulary IDs")
    input_digest = hashlib.sha256(
        json.dumps(
            [tensor_digest(t) for t in (tokens, positions, targets)], sort_keys=True
        ).encode()
    ).hexdigest()
    metadata = {
        **fingerprint,
        "weights_sha256": weight_hash,
        "input_sha256": input_digest,
        "token_count": tokens.numel(),
        "hardware": hardware,
        "torch_version": str(torch.__version__),
        "runtime_revision": MILES_REVISION,
        "engine": "miles-native",
        "capture_mode": "real-weight",
        "trace_scope": "block-boundary",
    }
    model = load_miles_model(directory, weight_map, miles_root, device)
    trace = VisitTrace(metadata)
    with torch.enable_grad(), capture_block_visits(model, trace):
        logits = model(tokens, positions)
        if logits.shape != (tokens.numel(), 200058) or logits.dtype != torch.float32:
            raise RuntimeError("unexpected Miles valid-vocabulary FP32 logits")
        logprob = F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
        trace.record_logprob(logprob)
        logprob.sum().backward()
    return trace.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--miles-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = capture_checkpoint(
        args.checkpoint,
        args.miles_root,
        json.loads(args.inputs.read_text()),
        torch.device(args.device),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
