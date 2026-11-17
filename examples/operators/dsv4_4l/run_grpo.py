# SPDX-License-Identifier: Apache-2.0
"""Run only the selected physical GPUs in an isolated four-device Ray container."""
from pathlib import Path
import json
import subprocess
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("checkpoint", type=Path)
parser.add_argument("data", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("cache", type=Path)
parser.add_argument("--trace-first", action="store_true")
parser.add_argument("--audit-response-length", type=int, default=3)
parser.add_argument("--audit-prompts-file", type=Path)
parser.add_argument("--stress-final-repeats", type=int, default=0)
args = parser.parse_args()
if not 1 <= args.audit_response_length <= 16:
    parser.error("audit response length must be between 1 and 16 for this four-layer recipe")
audit_overrides = []
if not 0 <= args.stress_final_repeats <= 10000:
    parser.error("stress repeats must be between zero and 10000")
if args.stress_final_repeats:
    audit_overrides.append(f"+rlk.stress_final_repeats={args.stress_final_repeats}")
if args.audit_prompts_file is not None:
    path = args.audit_prompts_file.resolve()
    if not path.is_relative_to(args.data.resolve()):
        parser.error("audit prompts must be a data artifact inside the mounted data directory")
    prompts = json.loads(path.read_text())
    if (
        not isinstance(prompts, list)
        or len(prompts) != 3
        or not all(isinstance(p, str) and p for p in prompts)
    ):
        parser.error("audit prompts must contain three nonempty strings")
    audit_overrides.append("rlk.audit_prompts=" + json.dumps(prompts))
repo = Path(__file__).resolve().parents[3]
for gpu in range(4, 8):
    subprocess.run(
        ["python3", str(repo / "examples/operators/dsv4_4l/guard_gpu.py"), str(gpu)], check=True
    )
a = json.loads(
    subprocess.check_output(
        ["python3", str(repo / "examples/operators/dsv4_4l/device_allocation.py"), "4,5", "6,7"],
        text=True,
    )
)
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
(out / "device-allocation.json").write_text(json.dumps(a, indent=2))
for command, label in [
    (["quota", "-s"], "quota"),
    (["df", "-h", str(out)], "disk"),
    (["docker", "info", "--format", "{{.DockerRootDir}}"], "docker-root"),
]:
    check = subprocess.run(command, capture_output=True, text=True)
    (out / (label + ".txt")).write_text(check.stdout + check.stderr)

name = "rlk-grpo-" + out.name
image = args.image
cmd = ["docker", "run", "--rm", "--init", "--name", name, "--device=/dev/kfd"]
for d in a:
    cmd += ["--device=" + d["render_node"]]
cmd += [
    "--group-add",
    str(os.stat("/dev/kfd").st_gid),
    "--group-add",
    str(os.stat(a[0]["render_node"]).st_gid),
    "--shm-size=16g",
    "--ipc=private",
]
env = dict(
    LD_PRELOAD="/opt/dsv4/lib/rlk-isolated-kfd.so",
    RLK_ALLOWED_KFD_GPU_IDS=",".join(str(d["kfd_gpu_id"]) for d in a),
    HIP_VISIBLE_DEVICES="0,1,2,3",
    RLK_TRACE_FIRST_AUDIT=("1" if args.trace_first else "0"),
    RLK_REPLAY_FIXED_CASES="/data/replay-fixed-sequences.json",
    DSV4_TRACE_FORMAT="cas-v1",
    RAY_ADDRESS="local",
    RAY_TMPDIR="/outputs/ray",
    TMPDIR="/outputs/tmp",
    OMP_NUM_THREADS="4",
    TOKENIZERS_PARALLELISM="false",
    VLLM_ROCM_USE_AITER="1",
    VLLM_PP_LAYER_PARTITION="3,1",
    VLLM_ENABLE_V1_MULTIPROCESSING="1",
    HF_HOME="/cache/huggingface",
    TRITON_CACHE_DIR="/cache/triton",
    XDG_CACHE_HOME="/cache/xdg",
    TORCH_EXTENSIONS_DIR="/cache/torch_extensions",
    HYDRA_FULL_ERROR="1",
)
if os.environ.get("RLK_TRACE_AUDIT_PATTERN"):
    env["RLK_TRACE_AUDIT_PATTERN"] = os.environ["RLK_TRACE_AUDIT_PATTERN"]
for k, v in env.items():
    cmd += ["-e", k + "=" + v]
for src, dst, mode in [
    (out, "/outputs", "rw"),
    (args.data.resolve(), "/data", "ro"),
    (args.checkpoint.resolve(), "/models/4l", "ro"),
    (args.cache.resolve(), "/cache", "rw"),
]:
    cmd += ["-v", str(src) + ":" + dst + ":" + mode]
args = [
    "actor_rollout_ref.model.path=/models/4l",
    "data.train_files=/data/train.parquet",
    "data.val_files=/data/val.parquet",
    "data.train_batch_size=2",
    "actor_rollout_ref.actor.ppo_mini_batch_size=2",
    "actor_rollout_ref.rollout.n=2",
    "trainer.total_training_steps=3",
    f"rlk.audit_response_length={args.audit_response_length}",
    "trainer.default_local_dir=/outputs/checkpoints",
    "rlk.sync_dir=/outputs/policy-sync",
    "data.max_prompt_length=128",
    "data.max_response_length=16",
    "data.dataloader_num_workers=0",
    "reward.num_workers=1",
    "actor_rollout_ref.rollout.agent.num_workers=1",
    "reward.custom_reward_function.path=/opt/dsv4/RL-Kernel/examples/operators/dsv4_4l/copy_task.py",
    "reward.custom_reward_function.name=compute_score",
    "actor_rollout_ref.rollout.max_model_len=256",
    "actor_rollout_ref.rollout.max_num_batched_tokens=512",
    "actor_rollout_ref.rollout.max_num_seqs=8",
    "actor_rollout_ref.rollout.enable_chunked_prefill=false",
    "ray_kwargs.ray_init.num_cpus=24",
    "+ray_kwargs.ray_init.num_gpus=4",
    "+ray_kwargs.ray_init.address=local",
    "+ray_kwargs.ray_init._temp_dir=/outputs/ray",
    "hydra.run.dir=/outputs/hydra",
    "trainer.save_freq=-1",
    "trainer.total_epochs=1",
    "data.shuffle=false",
]
args += audit_overrides
cmd += [
    image,
    "bash",
    "-c",
    (
        "mkdir -p /outputs/tmp /outputs/ray; cd /opt/dsv4/verl; "
        'exec /opt/dsv4-venv/bin/python -m verl.experimental.rl_kernel.main_ppo "$@"'
    ),
    "grpo",
    *args,
]
rec = dict(
    container=name,
    output=str(out),
    image=image,
    image_id=subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", image], text=True
    ).strip(),
    command=cmd,
    environment=env,
)
(out / "launch.json").write_text(json.dumps(rec, indent=2))
print(str(out), flush=True)
with (out / "driver.log").open("w") as f:
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    captured = False
    last_top = 0
    start = time.perf_counter()
    with (out / "gpu-memory.jsonl").open("w") as samples:
        while proc.poll() is None:
            sample = subprocess.run(
                ["rocm-smi", "-d", "4", "5", "6", "7", "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
            )
            if sample.returncode == 0:
                samples.write(
                    json.dumps(
                        dict(elapsed=time.perf_counter() - start, memory=json.loads(sample.stdout))
                    )
                    + "\n"
                )
                samples.flush()
            if not captured:
                inspect = subprocess.run(
                    ["docker", "inspect", name], capture_output=True, text=True
                )
                if inspect.returncode == 0:
                    (out / "container-inspect.json").write_text(inspect.stdout)
                    captured = True
            if time.perf_counter() - last_top > 20:
                top = subprocess.run(
                    ["docker", "top", name, "-eo", "pid,ppid,comm,args"],
                    capture_output=True,
                    text=True,
                )
                if top.returncode == 0:
                    (out / "own-processes.txt").write_text(top.stdout)
                    pids = [
                        line.split()[0]
                        for line in top.stdout.splitlines()[1:]
                        if any(
                            n in line
                            for n in (
                                "WorkerDict",
                                "Worker_PP",
                                "RLKTrainerVLLMServer",
                                "RLKTaskRunner",
                            )
                        )
                    ]
                    if pids:
                        physical = subprocess.run(
                            ["rocm-smi", "--showpidgpus", *pids], capture_output=True, text=True
                        )
                        (out / "physical-gpu-processes.txt").write_text(
                            physical.stdout + physical.stderr
                        )
                last_top = time.perf_counter()
            time.sleep(1)
    result = proc
    rec.update(exit_code=proc.returncode, wall_seconds=time.perf_counter() - start)
    (out / "job-result.json").write_text(json.dumps(rec, indent=2))
(out / "exit-code.json").write_text(json.dumps(dict(exit_code=result.returncode)))
print((out / "driver.log").read_text()[-12000:], flush=True)
raise SystemExit(result.returncode)
