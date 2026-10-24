# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Independent HF FP32/BF16 reference for pre-candidate Qwen3 tolerances.

Fixed teacher-forced tokens are calibration inputs, never sampler evidence.
This stage does not run Megatron, DeepSpeed or vLLM candidate code.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def report_json_default(value):
    """Transformers loading diagnostics can contain sets, including empty sets."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "model_revision": REVISION,
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    try:
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise ValueError("calibration requires a clean source commit")
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        manifest = json.loads(args.manifest.read_text())
        if manifest["revision"] != REVISION:
            raise ValueError("wrong model revision")
        for f in manifest["files"]:
            path = args.checkpoint / f["path"]
            if path.stat().st_size != f["size"] or sha(path) != f["sha256"]:
                raise ValueError(f"checkpoint mismatch: {path}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        texts = [
            "Explain why water freezes. Water freezes when its molecules lose enough "
            "thermal energy.",
            "Add one half and one third. A common denominator of six gives five sixths.",
        ]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        encoded["position_ids"] = (encoded.attention_mask.long().cumsum(1) - 1).clamp_min(0)
        valid = encoded.attention_mask[:, 1:].bool() & encoded.attention_mask[:, :-1].bool()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        rows = {}
        for name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            model, loading = AutoModelForCausalLM.from_pretrained(
                args.checkpoint,
                local_files_only=True,
                dtype=dtype,
                attn_implementation="eager",
                output_loading_info=True,
            )
            report[name + "_checkpoint_loading"] = loading
            if any(
                loading.get(key)
                for key in (
                    "missing_keys",
                    "unexpected_keys",
                    "mismatched_keys",
                    "error_msgs",
                )
            ):
                raise ValueError("Qwen3 reference checkpoint has missing or unmatched tensors")
            parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
            if parameter_dtypes != {dtype}:
                raise ValueError(f"expected reference dtype {dtype}, got {parameter_dtypes}")
            report[name + "_parameter_dtype"] = str(dtype)
            model = model.cuda().eval()
            model.gradient_checkpointing_enable()
            logits = model(**encoded, use_cache=False).logits[:, :-1].float()
            logps = logits.log_softmax(-1).gather(-1, encoded.input_ids[:, 1:, None]).squeeze(-1)
            loss = -logps[valid].mean()
            loss.backward()
            norm_sq = torch.zeros((), device="cuda", dtype=torch.float64)
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError("missing or nonfinite independent reference gradient")
                norm_sq += parameter.grad.double().square().sum()
            row = dict(
                logps=logps.detach().cpu(),
                loss=loss.detach().cpu(),
                grad_norm=norm_sq.sqrt().cpu(),
                input_ids=encoded.input_ids.cpu(),
                attention_mask=encoded.attention_mask.cpu(),
                valid=valid.cpu(),
            )
            torch.save(row, args.output / (name + ".pt"))
            rows[name] = row
            report[name] = dict(
                seconds=time.perf_counter() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            del model, logits, logps, loss, parameter, norm_sq
            gc.collect()
            torch.cuda.empty_cache()
        # Rule is committed before any candidate run. Never infer bounds from a
        # Megatron/DeepSpeed/vLLM candidate result or widen a failed bound.
        error = (
            (rows["fp32"]["logps"] - rows["bf16"]["logps"])[rows["fp32"]["valid"]]
            .abs()
            .max()
            .item()
        )
        thresholds = dict(
            model_revision=REVISION,
            fp32_reference_sha256=sha(args.output / "fp32.pt"),
            bf16_control_sha256=sha(args.output / "bf16.pt"),
            derivation="4 * independent HF BF16-vs-FP32 max error + fixed floor",
            behavior_logprob=dict(atol=4 * error + 1e-5, rtol=0.0),
            loss=dict(
                atol=4 * abs(rows["fp32"]["loss"].item() - rows["bf16"]["loss"].item()) + 1e-6,
                rtol=0.0,
            ),
            grad_norm=dict(
                atol=4 * abs(rows["fp32"]["grad_norm"].item() - rows["bf16"]["grad_norm"].item())
                + 1e-6,
                rtol=0.0,
            ),
        )
        (args.output / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
        report.update(
            status="reference_complete_candidate_not_run",
            torch=torch.__version__,
            device=torch.cuda.get_device_name(),
        )
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, default=report_json_default) + "\n"
        )


if __name__ == "__main__":
    main()
