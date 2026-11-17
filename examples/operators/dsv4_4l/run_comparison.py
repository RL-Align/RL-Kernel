# SPDX-License-Identifier: Apache-2.0
"""Run one fresh native or RLK job with selected-device guards and VRAM observations."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--group", choices=["P/P", "R/R"], required=True)
parser.add_argument("--image-manifest", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--fixed-cases", type=Path, required=True)
opts = parser.parse_args()
R = opts.output_root.resolve()
for d in ["artifacts", "logs"]:
    (R / d).mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[3]
group = opts.group
name = "native-PP" if group == "P/P" else "rlk-RR-rocm_triton"
if (R / "artifacts" / f"job-{name}.json").exists():
    raise SystemExit("This group already has a recorded job; choose a new output root")
fixed = R / "artifacts/predeclared-cases.json"
if fixed.exists():
    assert fixed.read_bytes() == opts.fixed_cases.read_bytes(), "predeclared fixed inputs changed"
else:
    fixed.write_bytes(opts.fixed_cases.read_bytes())
lock = (R / "artifacts/four-gpu.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
im = json.loads(opts.image_manifest.read_text())
name = "native-PP" if group == "P/P" else "rlk-RR-rocm_triton"
out = (
    R / "logs" / f'{name}-{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
)
assert (
    subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", im["tag"]], text=True
    ).strip()
    == im["id"]
)
args = [
    "bash",
    str(repo / "examples/operators/dsv4_4l/run_pipeline_cycle.sh"),
    im["tag"],
    "4,5",
    "6,7",
    str(opts.checkpoint.resolve()),
    str(out),
    str(opts.cache.resolve()),
]
env = dict(
    os.environ,
    DSV4_EXPERIMENT_GROUP=group,
    DSV4_TRAIN_STEPS="3",
    DSV4_TRACE_FORMAT="cas-v1",
    DSV4_MEASURE="1",
    DSV4_MHC_BACKEND="rocm_triton",
    DSV4_FIXED_CASES_HOST=str(R / "artifacts/predeclared-cases.json"),
)
rec = dict(
    group=group,
    output=str(out),
    image=im,
    argv=args,
    environment={k: v for k, v in env.items() if k.startswith("DSV4_")},
    start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    fixed_cases_sha256=hashlib.sha256(
        (R / "artifacts/predeclared-cases.json").read_bytes()
    ).hexdigest(),
)
(R / "artifacts/current-run.json").write_text(json.dumps(rec, indent=2) + "\n")
print("START", group, str(out), flush=True)
with (R / "logs" / f"launcher-{name}.log").open("w") as log:
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    dockername = f"rlk-dsv4-pipeline-{proc.pid}"
    start = time.perf_counter()
    captured = False
    top = False
    with (R / "logs" / f"gpu-memory-{name}.jsonl").open("w") as mem:
        while proc.poll() is None:
            sample = subprocess.run(
                ["rocm-smi", "-d", "4", "5", "6", "7", "--showmeminfo", "vram", "--json"],
                text=True,
                capture_output=True,
            )
            if sample.returncode == 0:
                mem.write(
                    json.dumps(
                        dict(
                            elapsed_seconds=time.perf_counter() - start,
                            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            devices=json.loads(sample.stdout),
                        )
                    )
                    + "\n"
                )
                mem.flush()
            if not captured:
                q = subprocess.run(
                    ["docker", "inspect", dockername], capture_output=True, text=True
                )
                if q.returncode == 0:
                    c = json.loads(q.stdout)[0]
                    assert c["Image"] == im["id"]
                    assert any(
                        m["Destination"] == "/outputs" and m["Source"] == str(out)
                        for m in c["Mounts"]
                    )
                    (out / "container-inspect.json").write_text(json.dumps(c, indent=2) + "\n")
                    captured = True
                    code = (
                        "import importlib.metadata as m,json; "
                        'print(json.dumps({d.metadata["Name"]:d.version '
                        'for d in m.distributions() if "rocm" in '
                        '(d.metadata.get("Name") or "").lower()}))'
                    )
                    q = subprocess.run(
                        ["docker", "exec", dockername, "/opt/dsv4-venv/bin/python", "-c", code],
                        capture_output=True,
                        text=True,
                    )
                    if q.returncode == 0:
                        (out / "runtime-rocm-sdk-distributions.json").write_text(q.stdout)
            if (out / "actor-topology.json").exists() and not top:
                q = subprocess.run(
                    ["docker", "top", dockername, "-eo", "pid,ppid,comm,args"],
                    capture_output=True,
                    text=True,
                )
                if q.returncode == 0:
                    (out / "worker-host-processes.txt").write_text(q.stdout)
                    pids = [
                        line.split()[0]
                        for line in q.stdout.splitlines()[1:]
                        if "distributed_actor_service" in line or "Worker_PP" in line
                    ]
                    if pids:
                        gpu = subprocess.run(
                            ["rocm-smi", "--showpidgpus", *pids], capture_output=True, text=True
                        )
                        (out / "own-worker-physical-gpu-processes.txt").write_text(
                            gpu.stdout + gpu.stderr
                        )
                    top = True
            time.sleep(1)
rec.update(
    exit_code=proc.returncode,
    elapsed_seconds=time.perf_counter() - start,
    stop_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    memory_samples=str(R / "logs" / f"gpu-memory-{name}.jsonl"),
)
if proc.returncode == 0 and group == "R/R":
    # A separate CPU-only saved-artifact audit; no source mount or model execution.
    audit_cmd = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "-i",
        "-v",
        str(out) + ":/outputs:ro",
        "-v",
        str(out) + ":/audit",
        "-v",
        str(opts.checkpoint.resolve()) + ":/models/4l:ro",
        im["tag"],
        "/opt/dsv4-venv/bin/python",
        "-",
    ]
    with (out / "actual-state-audit.log").open("w") as audit_log:
        subprocess.run(
            audit_cmd,
            input=(repo / "examples/operators/dsv4_4l/audit_continuous_state.py").read_text(),
            text=True,
            stdout=audit_log,
            stderr=subprocess.STDOUT,
            check=True,
        )
status = out / ("native-status.json" if group == "P/P" else "continuous-acceptance.json")
if status.exists():
    rec["result"] = json.loads(status.read_text())
(R / "artifacts" / f"job-{name}.json").write_text(json.dumps(rec, indent=2) + "\n")
print("FINISH", json.dumps(rec)[-2500:], flush=True)
if proc.returncode:
    print((out / "container.log").read_text()[-9000:], flush=True)
raise SystemExit(proc.returncode)
