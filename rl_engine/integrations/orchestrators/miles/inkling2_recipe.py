# SPDX-License-Identifier: Apache-2.0
"""INK-75: pinned, preconverted checkpoint GRPO launch planning."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

MILES_REVISION = "43ee2489a76a059801f60961c59310178b1ce74d"
# Do not invoke the upstream prepare() path: it downloads an unpinned revision.
NATIVE_ENTRY = (
    "import sys; from scripts.run_inkling2_mini import ScriptArgs, execute; "
    "execute(ScriptArgs(num_nodes=1, num_gpus_per_node=int(sys.argv[1]), "
    "num_rollout=int(sys.argv[2]), model_dir=sys.argv[3], data_dir=sys.argv[4], "
    "output_dir=sys.argv[5]))"
)


def plan(miles_root, model_dir, data_dir, output_dir, *, platform, mode, steps, gpus=8,
         qualified_entry=None):
    """Return argv, never silently fall back to the native CUDA recipe.

    A separately qualified worker launcher is mandatory for strict mode and
    ROCm because the pinned upstream recipe hardcodes FA4 and lacks a strict
    startup binding. Such launchers consume this same explicit manifest.
    """
    if platform not in ("cuda", "rocm") or mode not in ("native", "strict"):
        raise ValueError("unknown platform or execution mode")
    if steps not in (1, 10, 100, 200) or gpus not in (2, 4, 8):
        raise ValueError("unsupported rollout count or GPU count")
    root = Path(miles_root).resolve()
    model_dir, data_dir, output_dir = map(lambda p: Path(p).resolve(),
                                          (model_dir, data_dir, output_dir))
    native = platform == "cuda" and mode == "native"
    if native and gpus != 8:
        raise ValueError("pinned native reference uses eight GPUs")
    if not native and qualified_entry is None:
        raise RuntimeError("a qualified ROCm/strict worker entry is required")
    spec = dict(platform=platform, mode=mode, steps=steps, gpus=gpus,
                checkpoint=str(model_dir / "Inkling-2-Mini-bf16"),
                dataset=str(data_dir / "dapo-math-17k/dapo-math-17k.jsonl"),
                output_dir=str(output_dir), miles_revision=MILES_REVISION,
                precision="bf16-dequant", algorithm="grpo", ep=gpus, rollout_tp=1,
                prompts=32, samples=8, response_length=4096, sequence_length=6144,
                temperature=1.0, optimizer="adam", lr=1e-6, global_batch=256,
                micro_batch=1, kl_coef=0.0, entropy_coef=0.0,
                eps_clip=0.2, eps_clip_high=0.28, weight_decay=0.1,
                adam_beta1=0.9, adam_beta2=0.98)
    if native:
        argv = [sys.executable, "-c", NATIVE_ENTRY, str(gpus), str(steps),
                str(model_dir), str(data_dir), str(output_dir)]
    else:
        entry = Path(qualified_entry).resolve()
        argv = [sys.executable, str(entry), "--manifest", str(output_dir / "run.json")]
    return {"cwd": str(root), "argv": argv, "manifest": spec}


def launch(run):
    root = run["cwd"]
    revision = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != MILES_REVISION:
        raise RuntimeError("Miles revision mismatch")
    if subprocess.check_output(["git", "-C", root, "status", "--porcelain", "-uno"],
                               text=True).strip():
        raise RuntimeError("Miles checkout has tracked modifications")
    spec = run["manifest"]
    import torch

    actual = "rocm" if torch.version.hip is not None else "cuda"
    if not torch.cuda.is_available() or actual != spec["platform"]:
        raise RuntimeError("actual GPU platform mismatch")
    if torch.cuda.device_count() < spec["gpus"]:
        raise RuntimeError("insufficient visible GPUs")
    for rank in range(spec["gpus"]):
        props = torch.cuda.get_device_properties(rank)
        if actual == "rocm" and props.gcnArchName.split(":")[0] != "gfx950":
            raise RuntimeError("initial ROCm recipe requires gfx950")
        if actual == "cuda" and spec["mode"] == "native" and "B200" not in props.name:
            raise RuntimeError("native reference requires B200")
    checkpoint = Path(spec["checkpoint"])
    for name in ("model.safetensors.index.json", "config.json", "hf_quant_config.json"):
        if not (checkpoint / name).is_file():
            raise RuntimeError(f"missing preconverted checkpoint file: {name}")
    quant = json.loads((checkpoint / "hf_quant_config.json").read_text())
    if any(g["compute_dtype"] in ("nvfp4", "mxfp8")
           for g in quant["model"]["parameter_groups"]):
        raise RuntimeError("expected BF16-dequant sidecar")
    if not Path(spec["dataset"]).is_file():
        raise RuntimeError("missing prepared dataset")
    out = Path(spec["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(spec, indent=2) + "\n")
    subprocess.run(run["argv"], cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("miles-root", "model-dir", "data-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--platform", choices=("cuda", "rocm"), required=True)
    parser.add_argument("--mode", choices=("native", "strict"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--qualified-entry")
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args())
    execute = args.pop("execute")
    run = plan(**args)
    if execute:
        launch(run)
    else:
        print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
