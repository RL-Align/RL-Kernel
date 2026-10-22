# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""RFC #420 ``ws1_full_h3_transformer``: the whole pinned 33B H3 DiT, forward and backward.

Runs the released BF16 checkpoint on one GPU three ways on a small packed input:

* ``provider``  - diffusers' own arithmetic (cuBLAS / SDPA);
* ``candidate`` - the 50 real blocks through the RL-Kernel block graph, and (default) the
  audio projections, the text projection and the token refiner through their rows
  (``--provider-boundaries`` keeps those on the provider);
* ``candidate`` again - to prove the strict run is bitwise repeatable.

Reports forward/backward agreement with the provider, bitwise repeatability of the outputs,
of every parameter gradient and of the input gradients, and which boundary modules are bound
to rows. Gradients are hashed or compared on the fly and released, so memory stays bounded.
This is integration evidence for a pinned checkpoint; interim block bindings are listed in
the report and are not final-kernel qualification.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rl_engine.validation.models.h3_full_model import FullModelTrace, full_model_execution
from tools.validation.models.h3_full_model import packed_inputs, provider_variant_context


def tensor_sha(t: torch.Tensor) -> str:
    """Bit-level fingerprint computed on the GPU: two position-weighted sums of the raw 16-bit words.

    Not cryptographic; equality of two fingerprints means the bytes agree for any realistic
    difference (any single changed word changes the plain sum or the weighted sum).
    """

    t = t.detach().contiguous()
    words = t.view(torch.uint8).view(-1)
    words = words[: words.numel() // 2 * 2].view(torch.int16).to(torch.int64)
    weights = torch.arange(words.numel(), device=words.device, dtype=torch.int64) % 65521 + 1
    plain, weighted = int(words.sum()), int((words * weights).sum())
    return f"{t.dtype}:{tuple(t.shape)}:{plain:x}:{weighted:x}"


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def run_once(model, inputs, *, candidate, strict_boundaries, reference_grads=None, keep_grads=False,
             variant="default"):
    """One forward+backward. Returns outputs, input grads, per-parameter (sha, rel-to-reference)."""

    trace = FullModelTrace()
    stats, kept, handles = {}, {}, []
    for name, parameter in model.named_parameters():

        def hook(p, name=name):
            g = p.grad.detach()
            if not torch.isfinite(g).all():
                raise ValueError(f"non-finite gradient: {name}")
            entry = dict(sha=tensor_sha(g))
            if reference_grads is not None:
                entry["rel_vs_reference"] = rel(g, reference_grads[name].to(g.device))
            if keep_grads:
                kept[name] = g.cpu()
            stats[name] = entry
            p.grad = None

        handles.append(parameter.register_post_accumulate_grad_hook(hook))
    grad_inputs = ("hidden_states", "audio_hidden_states", "encoder_hidden_states")
    fresh = {k: (v.detach().clone().requires_grad_() if k in grad_inputs else v) for k, v in inputs.items()}
    torch.cuda.synchronize()
    begin = time.perf_counter()
    with full_model_execution(
        model, candidate=candidate, trace=trace, strict_boundaries=strict_boundaries
    ), provider_variant_context(variant):
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            video, audio = model(**fresh)
            torch.cuda.synchronize()
            forward_s = time.perf_counter() - begin
            loss = video.float().square().mean() + audio.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
    total_s = time.perf_counter() - begin
    for h in handles:
        h.remove()
    in_grads = {k: fresh[k].grad.detach().cpu() for k in ("hidden_states", "audio_hidden_states", "encoder_hidden_states")}
    return dict(
        video=video.detach().cpu(), audio=audio.detach().cpu(), in_grads=in_grads, stats=stats,
        kept=kept, trace=trace.validate(backward=True), forward_s=forward_s, total_s=total_s,
        loss=float(loss),
    )


def summarise(cand, ref):
    rels = [v["rel_vs_reference"] for v in cand["stats"].values() if "rel_vs_reference" in v]
    return dict(
        video_rel=rel(cand["video"], ref["video"]),
        audio_rel=rel(cand["audio"], ref["audio"]),
        input_grad_rel={k: rel(cand["in_grads"][k], ref["in_grads"][k]) for k in cand["in_grads"]},
        param_grad_rel_max=max(rels) if rels else None,
        param_grad_rel_median=sorted(rels)[len(rels) // 2] if rels else None,
        param_grad_count=len(cand["stats"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="dir holding transformer/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-boundaries", action="store_true")
    args = parser.parse_args()
    from diffusers import MiniMaxH3Transformer3DModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    model, loading = MiniMaxH3Transformer3DModel.from_pretrained(
        str(args.checkpoint / "transformer"), torch_dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True, output_loading_info=True,
    )
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("checkpoint did not load every tensor exactly")
    model = model.to("cuda").eval()
    model.enable_gradient_checkpointing()
    load_s = time.perf_counter() - started
    inputs = packed_inputs("cuda")
    strict = not args.provider_boundaries

    provider = run_once(model, inputs, candidate=False, strict_boundaries=False, keep_grads=True)
    provider_again = run_once(model, inputs, candidate=False, strict_boundaries=False)
    cand = run_once(model, inputs, candidate=True, strict_boundaries=strict, reference_grads=provider["kept"])
    cand2 = run_once(model, inputs, candidate=True, strict_boundaries=strict)
    # How far an independent provider kernel choice moves the same numbers (calibration).
    variants = {}
    for name in ("sdpa-efficient", "blas-cublaslt"):
        try:
            other = run_once(model, inputs, candidate=False, strict_boundaries=False,
                             reference_grads=provider["kept"], variant=name)
            variants[name] = summarise(other, provider)
        except Exception as exc:  # a forced kernel may be unavailable; record, do not fail
            variants[name] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            torch.backends.cuda.preferred_blas_library("default")

    def same(a, b):
        return (
            torch.equal(a["video"], b["video"]) and torch.equal(a["audio"], b["audio"])
            and all(torch.equal(a["in_grads"][k], b["in_grads"][k]) for k in a["in_grads"])
            and all(a["stats"][k]["sha"] == b["stats"][k]["sha"] for k in a["stats"])
        )

    report = dict(
        status="execution_complete",
        device=torch.cuda.get_device_name(), torch=torch.__version__,
        parameters=sum(p.numel() for p in model.parameters()),
        load_seconds=load_s, strict_boundaries=strict,
        candidate_vs_provider=summarise(cand, provider),
        provider_repeat_vs_provider=summarise(provider_again, provider),
        provider_variants_vs_provider=variants,
        provider_bitwise_repeatable=same(provider, provider_again),
        candidate_bitwise_repeatable=same(cand, cand2),
        seconds=dict(provider=provider["total_s"], candidate=cand["total_s"],
                     provider_forward=provider["forward_s"], candidate_forward=cand["forward_s"]),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        trace=dict(bindings=cand["trace"]["bindings"], boundary_bindings=cand["trace"]["boundary_bindings"],
                   candidate_calls=len(cand["trace"]["candidate_calls"])),
        loss=dict(provider=provider["loss"], candidate=cand["loss"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
