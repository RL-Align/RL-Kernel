# SPDX-License-Identifier: Apache-2.0
"""Attempt pristine framework startup; never enable RLK numerical/import adapters.

The direct-vLLM check is a diagnostic, not a completed native verl P/P job.
"""

import argparse
import asyncio
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/outputs")


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def actor():
    import torch
    import torch.distributed as dist
    from verl.trainer.config import CheckpointConfig
    from verl.utils.profiler.config import ProfilerConfig
    from verl.workers.config import (
        HFModelConfig,
        McoreEngineConfig,
        McoreOptimizerConfig,
        TrainingWorkerConfig,
    )
    from verl.workers.engine_workers import TrainingWorker

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    cfg = TrainingWorkerConfig(
        model_type="language_model",
        model_config=HFModelConfig(path="/models/4l", use_remove_padding=False),
        engine_config=McoreEngineConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=False,
            param_offload=False,
            optimizer_offload=False,
            forward_only=False,
            use_distributed_optimizer=False,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1,
            infer_micro_batch_size_per_gpu=1,
            override_transformer_config=dict(
                gradient_accumulation_fusion=False,
                recompute_granularity=None,
                apply_rope_fusion=False,
                moe_permute_fusion=False,
                moe_shared_expert_overlap=False,
            ),
        ),
        optimizer_config=McoreOptimizerConfig(
            lr=1e-4, total_training_steps=4, weight_decay=0.0, clip_grad=1.0
        ),
        checkpoint_config=CheckpointConfig(),
        profiler_config=ProfilerConfig(tool=None, save_path="/outputs", tool_config={}),
    )
    print("NATIVE_ACTOR_REAL_CALL", TrainingWorker, cfg, flush=True)
    worker = TrainingWorker(cfg)
    worker.reset()
    save(
        f"native-actor-loaded-rank{dist.get_rank()}.json",
        dict(
            engine=str(type(worker.engine)),
            source="pristine TrainingWorker.reset",
            completed_steps=0,
            status="LOADED_REQUIRES_TRAINING_HARNESS",
        ),
    )
    dist.destroy_process_group()


async def rollout():
    from verl.workers.config import HFModelConfig, RolloutConfig
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    cfg = RolloutConfig(
        name="vllm",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        data_parallel_size=1,
        expert_parallel_size=1,
        seed=42,
        prompt_length=129,
        response_length=3,
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=512,
        enforce_eager=True,
        enable_sleep_mode=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        load_format="safetensors",
        gpu_memory_utilization=0.30,
        logprobs_mode="raw_logprobs",
        calculate_log_probs=True,
        engine_kwargs={
            "vllm": {
                "distributed_executor_backend": "mp",
                "compilation_config": {"cudagraph_mode": "NONE"},
            }
        },
    )
    server = vLLMHttpServer(
        cfg,
        HFModelConfig(path="/models/4l", trust_remote_code=False, use_remove_padding=True),
        RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=2,
        nnodes=1,
        cuda_visible_devices="0,1",
    )
    await server.launch_server()
    save("native-rollout-loaded.json", dict(status="LOADED_REQUIRES_TRAINING_HARNESS"))


async def direct_vllm():
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    args = AsyncEngineArgs(
        model="/models/4l",
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        dtype="bfloat16",
        seed=42,
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=512,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.30,
        load_format="safetensors",
        distributed_executor_backend="mp",
    )
    engine = AsyncLLM.from_engine_args(args)
    cases = json.loads((ROOT / "predeclared-cases.json").read_text())
    for c in cases:
        async for output in engine.generate(
            TokensPrompt(prompt_token_ids=c["prompt_token_ids"]),
            SamplingParams(max_tokens=len(c["response_token_ids"]), temperature=0, ignore_eos=True),
            "native-capability-" + c["id"],
        ):
            save(
                "direct-native-vllm-last-output.json",
                dict(case_id=c["id"], value=str(output)),
            )
    engine.shutdown()


def child(mode):
    start = time.perf_counter()
    try:
        if mode == "actor":
            actor()
        else:
            asyncio.run(rollout() if mode == "rollout" else direct_vllm())
    except Exception as exc:
        tb = traceback.format_exc()
        loaded = [k for k in sys.modules if k.startswith("rl_engine")]
        save(
            f'native-{mode}-rank{os.environ.get("RANK", "driver")}-error.json',
            dict(
                status="BLOCKED",
                mode=mode,
                exception=type(exc).__name__,
                message=str(exc),
                traceback=tb,
                seconds=time.perf_counter() - start,
                completed_steps=0,
                rlk_modules_loaded=loaded,
                numerical_adapters_enabled=False,
                pid=os.getpid(),
                local_rank=os.environ.get("LOCAL_RANK"),
            ),
        )
        print(tb, flush=True)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=["actor", "rollout", "direct-vllm"])
    args = parser.parse_args()
    if args.child:
        raise SystemExit(child(args.child))
    plugins = importlib.metadata.entry_points(group="vllm.general_plugins")
    retained = [p.name for p in plugins if p.name != "rl_kernel"]
    save(
        "native-plugin-policy.json",
        dict(
            disabled=["rl_kernel"],
            retained=retained,
            reason="Exclude only RLK general-plugin registration; preserve other native plugins",
        ),
    )
    outcomes = []
    for mode in ["actor", "rollout", "direct-vllm"]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("RL_KERNEL_")}
        env["VLLM_PLUGINS"] = ",".join(retained)
        env.pop("DSV4_FIXED_SHARED_DOWN", None)
        env.pop("DSV4_MHC_BACKEND", None)
        env.pop("DSV4_MEASURE", None)
        if mode == "actor":
            env.pop("ROCR_VISIBLE_DEVICES", None)
            env["HIP_VISIBLE_DEVICES"] = "0,1"
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "--module",
                __spec__.name,
                "--child",
                mode,
            ]
        else:
            cmd = [sys.executable, "-m", __spec__.name, "--child", mode]
        start = time.perf_counter()
        with (ROOT / f"native-{mode}.log").open("w") as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        outcomes.append(
            dict(
                mode=mode,
                exit_code=proc.returncode,
                elapsed_seconds=time.perf_counter() - start,
                log=str(ROOT / f"native-{mode}.log"),
            )
        )
    # Isolated provenance subprocess: runtime_versions installs import ABI hooks;
    # it MUST NOT run in any pristine native model process above.
    cmd = [
        sys.executable,
        "-c",
        "import json; from examples.operators.dsv4_4l.audit_dependencies "
        "import audit,runtime_versions; "
        "print(json.dumps(dict(audit=audit(),versions=runtime_versions())))",
    ]
    p = subprocess.run(cmd, text=True, capture_output=True, check=True)
    (ROOT / "provenance-subprocess.log").write_text(p.stdout + p.stderr)
    save("native-provenance.json", json.loads(p.stdout.splitlines()[-1]))
    blocked = [x for x in outcomes if x["mode"] in ("actor", "rollout") and x["exit_code"]]
    save(
        "native-status.json",
        dict(
            group="P/P",
            status="BLOCKED" if blocked else "STARTUP_ONLY",
            completed_steps=0,
            versions_completed=[],
            outcomes=outcomes,
            numerical_substitutions=False,
            compatibility=["GPU visibility/KFD aperture isolation only; no model arithmetic"],
            mismatch_count=None,
            throughput=None,
            reason=(
                "Actual pristine verl startup failed; see per-rank exception records"
                if blocked
                else "Startup succeeded; training comparison must still be executed "
                "before any acceptance"
            ),
        ),
    )
    if not blocked:
        raise RuntimeError(
            "Native startup succeeded: implement/run native continuous scoring before reporting P/P"
        )


if __name__ == "__main__":
    main()
