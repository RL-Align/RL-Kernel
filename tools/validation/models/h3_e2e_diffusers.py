# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""RFC #420 ``final_e2e_diffusers_cuda``: pinned Diffusers against RL-Kernel, full trajectory.

The workload is fixed by ``examples/models/minimax_h3/e2e-t2va-contract.json``: the verl-omni
H3 T2VA recipe's rollout settings, at 256x384, 121 requested frames (the pipeline renders 124,
the next ``17 n + 5``), 10 inference steps and no CFG, with 8 cases (4 prompts x 2 seeds).
Two arms run the same pinned Diffusers pipeline:

* ``provider``: the pipeline as released (P);
* ``candidate``: the same pipeline with the transformer on the RL-Kernel strict graph
  (``full_model_execution`` with strict boundaries, which need SM90) and both schedulers on the
  scheduler rows (``h3_strict_scheduler``) (R). ``--provider-boundaries`` keeps the boundary
  modules on the provider, for a plumbing run off SM90; it is not final evidence, and the manifest
  says so.

``run`` writes one directory per case and repeat. Each holds:

* a sealed trajectory artifact, ``trajectory.json``, with per-step velocity and latent digests
  (``rl_engine.models.minimax_h3.trajectory``), and ``tensors.safetensors``;
* the decoded media as ``video.npy`` and ``audio.npy`` (float32), with their digests;
* ``manifest.json``: provenance, the operator trace (R), the scheduler-row counts, peak memory
  and the time split.

The time split covers text encoding, each transformer call, each scheduler step and media
decode. It is measured with CUDA synchronisation around each phase. Model offload is on. The
transformer's first call includes moving its weights to the GPU, so calls 2..N are reported
separately as the steady state. Text-encoder time excludes its weight move, which the pipeline
runs before calling ``text_encoder.model``. Communication is 0: one GPU.

``compare`` reads a provider and a candidate output and writes ``report.json``.

* **Gate:** every candidate repeat is bitwise equal to the case's first candidate run, at
  every step, in both modalities.
* **Reported, not gated:**
  - P against R: first drift, per-step max abs difference and the final latents;
  - provider repeats;
  - media PSNR and relative L2 between P and R;
  - the time split per arm.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

CONTRACT = "examples/models/minimax_h3/e2e-t2va-contract.json"
PHASES = ("text_encoder", "transformer", "scheduler", "decode")
REQUIRED = (
    "tokenizer",
    "text_encoder",
    "transformer",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)


def source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def contract_sha256(contract: dict[str, Any]) -> str:
    """The seal of a contract: its canonical JSON without the seal itself."""

    sealed = {k: v for k, v in contract.items() if k != "contract_sha256"}
    return hashlib.sha256(json.dumps(sealed, sort_keys=True).encode()).hexdigest()


def load_contract(root: Path) -> dict[str, Any]:
    contract = json.loads((root / CONTRACT).read_text())
    # the whole workload is sealed, not only the cases: sizes, steps, revisions and the query chunk
    if contract_sha256(contract) != contract["contract_sha256"]:
        raise ValueError("the contract changed")
    cases = contract["cases"]
    if [c["case"] for c in cases] != list(range(len(cases))):
        raise ValueError("contract cases must be numbered 0..N-1 in order")
    if contract["resolved_num_frames"] != resolved_frames(contract["num_frames"]):
        raise ValueError("resolved_num_frames is not the pipeline's rounding of num_frames")
    return contract


def resolved_frames(requested: int) -> int:
    """The frame count the pinned pipeline renders: the video VAE needs ``17 * n + 5``, and the
    pipeline rounds a request up to the next such count (the recipe's 121 becomes 124)."""

    return 5 + 17 * math.ceil(max(requested - 5, 0) / 17)


def select_cases(contract: dict[str, Any], spec: str | None) -> list[dict[str, Any]]:
    if not spec:
        return list(contract["cases"])
    wanted = [int(x) for x in spec.split(",")]
    by = {c["case"]: c for c in contract["cases"]}
    missing = [w for w in wanted if w not in by]
    if missing:
        raise ValueError(f"cases {missing} are not in the contract")
    return [by[w] for w in wanted]


# --- timing ----------------------------------------------------------------------------------


class PhaseTimer:
    """Seconds per phase call, with CUDA synchronisation around each call."""

    def __init__(self) -> None:
        self.calls: dict[str, list[float]] = {p: [] for p in PHASES}
        self._starts: dict[int, float] = {}

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def module(self, module: torch.nn.Module, phase: str):
        """Time every forward of ``module`` (hooks, so no instance ``forward`` is added)."""

        def pre(mod, args, kwargs):
            self._sync()
            self._starts[id(mod)] = time.perf_counter()

        def post(mod, args, kwargs, output):
            self._sync()
            self.calls[phase].append(time.perf_counter() - self._starts.pop(id(mod)))

        handles = [
            module.register_forward_pre_hook(pre, with_kwargs=True),
            module.register_forward_hook(post, with_kwargs=True),
        ]
        return lambda: [h.remove() for h in handles]

    @contextlib.contextmanager
    def method(self, obj: Any, name: str, phase: str):
        """Time ``obj.name(...)`` calls; the instance attribute is removed or restored on exit."""

        had = name in obj.__dict__
        previous = obj.__dict__.get(name)
        inner = getattr(obj, name)

        def timed(*args, **kwargs):
            self._sync()
            start = time.perf_counter()
            try:
                return inner(*args, **kwargs)
            finally:
                self._sync()
                self.calls[phase].append(time.perf_counter() - start)

        setattr(obj, name, timed)
        try:
            yield
        finally:
            if had:
                setattr(obj, name, previous)
            else:
                obj.__dict__.pop(name, None)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for phase, values in self.calls.items():
            out[phase] = {"calls": len(values), "seconds": sum(values), "per_call": values}
        model = self.calls["transformer"]
        out["transformer_steady"] = {
            "calls": max(len(model) - 1, 0),
            "mean_seconds": (sum(model[1:]) / (len(model) - 1)) if len(model) > 1 else None,
        }
        out["communication"] = {"calls": 0, "seconds": 0.0, "note": "one GPU"}
        return out


# --- trajectory capture -----------------------------------------------------------------------


class StepCapture:
    """Records each scheduler step's velocity, input and output latent, per modality."""

    def __init__(self) -> None:
        self.initial: dict[str, torch.Tensor] = {}
        self.velocity: dict[str, list[torch.Tensor]] = {"video": [], "audio": []}
        self.latent: dict[str, list[torch.Tensor]] = {"video": [], "audio": []}

    @contextlib.contextmanager
    def scheduler(self, scheduler: Any, modality: str):
        had = "step" in scheduler.__dict__
        previous = scheduler.__dict__.get("step")
        inner = scheduler.step

        def step(model_output, timestep, sample, *args, **kwargs):
            if modality not in self.initial:
                self.initial[modality] = sample.detach().cpu().clone()
            out = inner(model_output, timestep, sample, *args, **kwargs)
            prev = out[0] if isinstance(out, tuple) else out.prev_sample
            self.velocity[modality].append(model_output.detach().cpu().clone())
            self.latent[modality].append(prev.detach().cpu().clone())
            return out

        scheduler.step = step
        try:
            yield
        finally:
            if had:
                scheduler.step = previous
            else:
                scheduler.__dict__.pop("step", None)

    def seal(self, *, label, seed, grids, shifts, trace_sha256, metadata, keep_tensors, out_dir):
        from rl_engine.models.minimax_h3 import trajectory as tj

        steps = {len(v) for v in self.velocity.values()}
        if len(steps) != 1:
            raise ValueError(f"the modalities ran different step counts: {steps}")
        recorder = tj.TrajectoryRecorder(
            variant="FL2VA",
            label=label,
            seed=seed,
            sigma_grids=grids,
            initial_noise=self.initial,
            shifts=shifts,
            operator_trace_sha256=trace_sha256,
            metadata=metadata,
            keep_tensors=keep_tensors,
        )
        for i in range(steps.pop()):
            recorder.record_step(
                i,
                velocity={m: self.velocity[m][i] for m in ("video", "audio")},
                latent={m: self.latent[m][i] for m in ("video", "audio")},
            )
        artifact = recorder.seal()
        (out_dir / "trajectory.json").write_text(tj.dumps(artifact))
        if keep_tensors:
            recorder.save_tensors(out_dir / "tensors.safetensors")
        return artifact

    def finite(self) -> bool:
        return all(
            bool(torch.isfinite(t).all())
            for group in (self.velocity, self.latent)
            for values in group.values()
            for t in values
        )


# --- run -------------------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def provenance(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    import diffusers

    from rl_engine.backends import extension

    provider_root = Path(diffusers.__file__).resolve().parents[2]
    if _git("rev-parse", "HEAD", cwd=provider_root) != contract["provider_revision"]:
        raise ValueError("the Diffusers provider is not the pinned revision")
    if _git("status", "--porcelain", "--untracked-files=no", cwd=provider_root):
        raise ValueError("the Diffusers provider checkout is dirty")
    ext = getattr(extension, "_C", None)
    return {
        "source_commit": _git("rev-parse", "HEAD", cwd=root),
        "provider_revision": contract["provider_revision"],
        "model_revision": contract["model_revision"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "extension_in_source": bool(
            ext is not None
            and os.path.realpath(ext.__file__).startswith(os.path.realpath(root) + os.sep)
        ),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    }


def build_pipeline(checkpoint: Path, offload_margin: str = "20GB"):
    from diffusers import ComponentsManager, ModularPipeline
    from transformers import Qwen3VLProcessor

    # load the processor before the large models (as h3_full_generation does)
    processor = Qwen3VLProcessor.from_pretrained(
        str(checkpoint / "processor"), local_files_only=True
    )
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=offload_margin)
    pipe = ModularPipeline.from_pretrained(
        str(checkpoint), workflow="t2va", components_manager=manager
    )
    pipe.update_components(processor=processor)
    pipe.load_components(
        dtype=torch.bfloat16, pretrained_model_name_or_path=str(checkpoint), local_files_only=True
    )
    missing = [name for name in REQUIRED if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"required T2VA components failed to load: {missing}")
    return pipe


def candidate_bindings(contract: dict[str, Any]):
    from rl_engine.backends.cuda.flow.ode_step import H3OdeStepCUDAOp
    from rl_engine.backends.cuda.model_specific.minimax_h3.sigma_schedule import (
        H3SigmaScheduleCudaOp,
    )
    from rl_engine.validation.models.h3_block import CandidateOps
    from rl_engine.validation.models.h3_full_model import QueryChunkAttention

    ops = CandidateOps()
    ops.attention = QueryChunkAttention(ops.attention, contract["attention_query_chunk"])
    return ops, H3SigmaScheduleCudaOp(), H3OdeStepCUDAOp()


def media_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_case(
    pipe,
    case,
    repeat,
    *,
    arm,
    contract,
    ops,
    sched_ops,
    out_dir,
    keep_tensors,
    prov,
    strict_boundaries,
):
    from rl_engine.validation.models import h3_strict_scheduler as hss
    from rl_engine.validation.models.h3_full_model import FullModelTrace, full_model_execution

    out_dir.mkdir(parents=True, exist_ok=False)
    timer, capture = PhaseTimer(), StepCapture()
    trace = FullModelTrace()
    torch.cuda.reset_peak_memory_stats()
    removers = [
        # the T2VA encoder step calls ``text_encoder.model`` after running the offload hook itself
        timer.module(pipe.text_encoder.model, "text_encoder"),
        timer.module(pipe.transformer, "transformer"),
    ]
    calls: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        with contextlib.ExitStack() as stack:
            if arm == "candidate":
                sigma_op, step_op = sched_ops
                for name in ("scheduler", "audio_scheduler"):
                    calls[name] = stack.enter_context(
                        hss.bind(getattr(pipe, name), sigma_op=sigma_op, step_op=step_op)
                    )
            # timed inside the capture, so its copies to the CPU are not scheduler time
            for name in ("scheduler", "audio_scheduler"):
                stack.enter_context(timer.method(getattr(pipe, name), "step", "scheduler"))
            stack.enter_context(capture.scheduler(pipe.scheduler, "video"))
            stack.enter_context(capture.scheduler(pipe.audio_scheduler, "audio"))
            for name in ("vae", "audio_vae"):
                stack.enter_context(timer.method(getattr(pipe, name), "decode", "decode"))
            stack.enter_context(torch.inference_mode())
            stack.enter_context(
                full_model_execution(
                    pipe.transformer,
                    candidate=arm == "candidate",
                    trace=trace,
                    ops=ops,
                    strict_boundaries=strict_boundaries,
                )
            )
            result = pipe(
                prompt=case["prompt"],
                num_frames=contract["num_frames"],
                height=contract["height"],
                width=contract["width"],
                num_inference_steps=contract["num_inference_steps"],
                generator=torch.Generator().manual_seed(case["seed"]),
                output_type="np",
                output=["videos", "audio", "sampling_rate"],
            )
    finally:
        for remove in removers:
            remove()
    seconds = time.perf_counter() - started
    trace_summary = trace.validate()
    trace_sha256 = hashlib.sha256(json.dumps(trace_summary, sort_keys=True).encode()).hexdigest()
    video = np.asarray(result["videos"][0], dtype=np.float32)
    audio = np.asarray(result["audio"][0].detach().float().cpu().numpy(), dtype=np.float32)
    want = (contract["resolved_num_frames"], contract["height"], contract["width"], 3)
    if video.shape != want:
        raise ValueError(f"video shape {video.shape}, expected {want}")
    finite = capture.finite() and bool(np.isfinite(video).all()) and bool(np.isfinite(audio).all())
    grids = {
        "video": pipe.scheduler.sigmas.float().cpu().tolist(),
        "audio": pipe.audio_scheduler.sigmas.float().cpu().tolist(),
    }
    shifts = {"video": float(pipe.scheduler.shift), "audio": float(pipe.audio_scheduler.shift)}
    label = f"{arm}-case{case['case']}-r{repeat}"
    artifact = capture.seal(
        label=label,
        seed=case["seed"],
        grids=grids,
        shifts=shifts,
        trace_sha256=trace_sha256 if arm == "candidate" else "",
        metadata={"workflow": contract["workflow"], "case": case["case"], "arm": arm},
        keep_tensors=keep_tensors,
        out_dir=out_dir,
    )
    np.save(out_dir / "video.npy", video)  # float32: the media metrics are computed from it
    np.save(out_dir / "audio.npy", audio)
    manifest = {
        "label": label,
        "arm": arm,
        "case": case,
        "repeat": repeat,
        "provenance": prov,
        "strict_boundaries": strict_boundaries,
        "finite": finite,
        "seconds": seconds,
        "timing": timer.summary(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "scheduler_rows": {k: {kk: vv for kk, vv in v.items()} for k, v in calls.items()},
        "trace": trace_summary,
        "trace_sha256": trace_sha256,
        "artifact_sha256": artifact["artifact_sha256"],
        "final_latent_sha256": artifact["final_latent_sha256"],
        "video_float32_sha256": media_digest(video),
        "audio_float32_sha256": media_digest(audio),
        "sampling_rate": int(result["sampling_rate"]),
    }
    finish_run(out_dir, manifest)
    return manifest


def finish_run(out_dir: Path, manifest: dict[str, Any]) -> None:
    """Write the completion manifest of a finite run. A non-finite run writes ``failure.json``
    instead and raises, so the next ``run`` retries it rather than skipping it as done."""

    text = json.dumps(manifest, indent=1, default=str) + "\n"
    if not manifest["finite"]:
        (out_dir / "failure.json").write_text(text)
        raise ValueError(
            f"{manifest['label']}: non-finite tensors or media (outputs kept in {out_dir})"
        )
    (out_dir / "manifest.json").write_text(text)


def check_resumable(
    manifest: dict[str, Any],
    *,
    arm: str,
    case: dict[str, Any],
    strict_boundaries: bool,
    prov: dict[str, Any],
) -> None:
    """A finished run is skipped only if this invocation would have produced it: the same arm,
    case, boundaries, source commit and checkpoint check. Anything else raises."""

    want = {
        "arm": arm,
        "case": case,
        "strict_boundaries": strict_boundaries,
        "source_commit": prov["source_commit"],
        "checkpoint_hash_checked": prov["checkpoint_hash_checked"],
    }
    got = {
        "arm": manifest.get("arm"),
        "case": manifest.get("case"),
        "strict_boundaries": manifest.get("strict_boundaries"),
        "source_commit": manifest.get("provenance", {}).get("source_commit"),
        "checkpoint_hash_checked": manifest.get("provenance", {}).get("checkpoint_hash_checked"),
    }
    differ = sorted(k for k in want if want[k] != got[k])
    if differ:
        raise ValueError(
            f"{manifest.get('label')}: an existing run differs from this invocation in {differ}; "
            "use another --output, or move it aside"
        )


def cmd_run(args) -> None:
    root = source_root()
    contract = load_contract(root)
    cases = select_cases(contract, args.cases)
    if _git("status", "--porcelain", "--untracked-files=no", cwd=root):
        raise ValueError("an evidence run needs a clean source checkout")
    from tools.validation.models.h3_full_model import verify_checkpoint

    if not args.skip_checkpoint_hash:
        verify_checkpoint(
            args.checkpoint,
            root / "rl_engine/validation/models/h3_full_manifest.json",
            all_files=True,
        )
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("TF32 matmul must be off")
    prov = provenance(root, contract)
    prov["checkpoint_hash_checked"] = not args.skip_checkpoint_hash
    prov["offload_margin"] = args.offload_margin
    pipe = build_pipeline(args.checkpoint, args.offload_margin)
    ops, sched_ops = None, None
    if args.arm == "candidate":
        ops, sigma_op, step_op = candidate_bindings(contract)
        sched_ops = (sigma_op, step_op)
    args.output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        for repeat in range(args.repeats):
            directory = args.output / f"case{case['case']}" / f"r{repeat}"
            strict_boundaries = args.arm == "candidate" and not args.provider_boundaries
            if (directory / "manifest.json").exists():
                check_resumable(
                    json.loads((directory / "manifest.json").read_text()),
                    arm=args.arm,
                    case=case,
                    strict_boundaries=strict_boundaries,
                    prov=prov,
                )
                print(f"{directory} done; skipping", flush=True)
                continue
            if directory.exists():
                directory.rename(directory.with_name(f"{directory.name}-failed-{int(time.time())}"))
            m = run_case(
                pipe,
                case,
                repeat,
                arm=args.arm,
                contract=contract,
                ops=ops,
                sched_ops=sched_ops,
                out_dir=directory,
                keep_tensors=not args.no_tensors,
                prov=prov,
                strict_boundaries=strict_boundaries,
            )
            t, final = m["timing"], m["final_latent_sha256"]
            print(
                f"{m['label']}: {m['seconds']:.1f} s, transformer steady "
                f"{t['transformer_steady']['mean_seconds']}, final video "
                f"{final['video'][:12]} audio {final['audio'][:12]}",
                flush=True,
            )


# --- compare ---------------------------------------------------------------------------------


def _runs(root: Path, case: int) -> list[Path]:
    base = root / f"case{case}"
    return sorted(
        (
            p
            for p in base.glob("r*")
            if re.fullmatch(r"r\d+", p.name) and (p / "manifest.json").exists()
        ),
        key=lambda p: int(p.name[1:]),
    )


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(1.0 / mse)


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 of ``a`` against ``b``; infinite when ``b`` is silent and ``a`` is not."""

    den = float(np.linalg.norm(b.astype(np.float64)))
    num = float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))
    if den == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / den


def compare_case(
    provider: Path, candidate: Path, case: int, contract_case: dict[str, Any] | None = None
) -> dict[str, Any]:
    from rl_engine.models.minimax_h3 import trajectory as tj

    def load(p: Path):
        return tj.loads((p / "trajectory.json").read_text())

    def tensors(p: Path):
        path = p / "tensors.safetensors"
        return path if path.exists() else None

    p_runs, r_runs = _runs(provider, case), _runs(candidate, case)
    if not p_runs or not r_runs:
        raise ValueError(
            f"case {case}: missing runs (provider {len(p_runs)}, candidate {len(r_runs)})"
        )
    r0 = load(r_runs[0])
    repeats = [tj.drift_report(r0, load(r)) for r in r_runs[1:]]
    p0 = load(p_runs[0])
    provider_repeats = [tj.drift_report(p0, load(p)) for p in p_runs[1:]]
    pr = tj.drift_report(p0, r0, tensors_lhs=tensors(p_runs[0]), tensors_rhs=tensors(r_runs[0]))
    pv, rv = np.load(p_runs[0] / "video.npy"), np.load(r_runs[0] / "video.npy")
    pa, ra = np.load(p_runs[0] / "audio.npy"), np.load(r_runs[0] / "audio.npy")
    manifests = {
        "provider": [json.loads((p / "manifest.json").read_text()) for p in p_runs],
        "candidate": [json.loads((p / "manifest.json").read_text()) for p in r_runs],
    }
    # each folder must hold the arm it is compared as, and the contract's case
    for arm, ms in manifests.items():
        for m in ms:
            if m.get("arm") != arm:
                raise ValueError(
                    f"case {case}: {m.get('label')} is a {m.get('arm')} run, not {arm}"
                )
            if contract_case is not None and m.get("case") != contract_case:
                raise ValueError(
                    f"case {case}: {m.get('label')} ran another case than the contract's"
                )
    return {
        "case": case,
        "candidate_runs": len(r_runs),
        "candidate_repeats_bitwise": all(r["bitwise_equal"] for r in repeats),
        "candidate_repeats": [
            {k: r[k] for k in ("compared", "differing", "bitwise_equal", "first_drift")}
            for r in repeats
        ],
        "candidate_media_repeat_equal": len(
            {m["video_float32_sha256"] + m["audio_float32_sha256"] for m in manifests["candidate"]}
        )
        == 1,
        "provider_runs": len(p_runs),
        "provider_repeats_bitwise": all(r["bitwise_equal"] for r in provider_repeats),
        "provider_vs_candidate": {
            k: pr[k]
            for k in ("compared", "differing", "bitwise_equal", "first_drift", "final_latent_equal")
        }
        | {"entries": pr["entries"]},
        "media": {
            "video_psnr_db": _psnr(rv.astype(np.float32), pv.astype(np.float32)),
            "video_max_abs": float(np.max(np.abs(rv.astype(np.float32) - pv.astype(np.float32)))),
            "audio_rel_l2": _rel_l2(ra, pa),
        },
        "finite": all(m["finite"] for ms in manifests.values() for m in ms),
        # final evidence: a hashed checkpoint in both arms; in every candidate run strict
        # boundaries and both schedulers on their rows for at least one step
        "final_evidence": all(
            m["provenance"].get("checkpoint_hash_checked") for ms in manifests.values() for m in ms
        )
        and all(
            m.get("strict_boundaries")
            and all(
                m.get("scheduler_rows", {}).get(name, {}).get("row", 0) > 0
                for name in ("scheduler", "audio_scheduler")
            )
            for m in manifests["candidate"]
        ),
        "scheduler_rows": manifests["candidate"][0]["scheduler_rows"],
        "timing": {arm: [m["timing"] for m in ms] for arm, ms in manifests.items()},
        "seconds": {arm: [m["seconds"] for m in ms] for arm, ms in manifests.items()},
        "peak_allocated_bytes": {
            arm: max(m["peak_allocated_bytes"] for m in ms) for arm, ms in manifests.items()
        },
    }


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ("provider", "candidate"):
        first = [c["timing"][arm][0] for c in cases]
        arms[arm] = {phase: _mean([t[phase]["seconds"] for t in first]) for phase in PHASES} | {
            "transformer_steady_per_call": _mean(
                [t["transformer_steady"]["mean_seconds"] for t in first]
            ),
            "total": _mean([c["seconds"][arm][0] for c in cases]),
            "peak_allocated_gib": max(c["peak_allocated_bytes"][arm] for c in cases) / 2**30,
        }
    return {
        "gate_candidate_repeats_bitwise": all(c["candidate_repeats_bitwise"] for c in cases),
        "gate_every_case_has_a_repeat": all(c["candidate_runs"] >= 2 for c in cases),
        "all_finite": all(c["finite"] for c in cases),
        "final_evidence": all(c["final_evidence"] for c in cases),
        "provider_repeats_bitwise": all(c["provider_repeats_bitwise"] for c in cases),
        "provider_vs_candidate_bitwise_cases": sum(
            c["provider_vs_candidate"]["bitwise_equal"] for c in cases
        ),
        "first_drift": [c["provider_vs_candidate"]["first_drift"] for c in cases],
        "video_psnr_db": [c["media"]["video_psnr_db"] for c in cases],
        "audio_rel_l2": [c["media"]["audio_rel_l2"] for c in cases],
        "time_mean_first_run": arms,
    }


def cmd_compare(args) -> None:
    contract = load_contract(source_root())
    cases = [
        compare_case(args.provider, args.candidate, c["case"], c)
        for c in select_cases(contract, args.cases)
    ]
    report = {"row": contract["row"], "summary": summarize(cases), "cases": cases}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    s = report["summary"]
    print(
        json.dumps(
            {k: s[k] for k in s if k.startswith("gate") or k in ("all_finite", "final_evidence")}
        )
    )
    if not (
        s["gate_candidate_repeats_bitwise"]
        and s["gate_every_case_has_a_repeat"]
        and s["all_finite"]
        and (s["final_evidence"] or args.allow_non_final)
    ):
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run one arm over the contract's cases")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--arm", choices=("provider", "candidate"), required=True)
    r.add_argument("--cases", help="comma list of case numbers (default: all)")
    r.add_argument("--repeats", type=int, default=2, help="runs per case (default 2)")
    r.add_argument("--no-tensors", action="store_true", help="do not keep step tensors")
    r.add_argument(
        "--offload-margin",
        default="20GB",
        help="GPU memory model offload keeps free (default 20GB, as h3_full_generation)",
    )
    r.add_argument(
        "--provider-boundaries",
        action="store_true",
        help="candidate only: keep the boundary modules on the provider (plumbing runs off SM90)",
    )
    r.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="smoke runs only: skip hashing every checkpoint file (recorded in the manifest)",
    )
    c = sub.add_parser("compare", help="gate the candidate repeats, report P against R")
    c.add_argument("--provider", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--cases", help="comma list of case numbers (default: all)")
    c.add_argument(
        "--allow-non-final",
        action="store_true",
        help="exit 0 for a plumbing run that is not final evidence (the report still says so)",
    )
    args = p.parse_args(argv)
    if args.command == "run":
        if args.repeats < 1:
            p.error("--repeats must be at least 1")
        if args.provider_boundaries and args.arm != "candidate":
            p.error("--provider-boundaries applies to the candidate arm only")
        cmd_run(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
