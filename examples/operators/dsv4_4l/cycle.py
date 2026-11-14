# SPDX-License-Identifier: Apache-2.0
"""Real verl actor/rollout cycle, retaining the SAME live rollout through update."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/outputs")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


async def wait_file(path, process, timeout=600):
    end = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"actual actor exited {process.returncode}; see actor-service.log")
        if (ROOT / "actor-error.json").exists():
            raise RuntimeError((ROOT / "actor-error.json").read_text())
        if time.monotonic() > end:
            raise TimeoutError(str(path))
        await asyncio.sleep(0.2)
    return json.loads(path.read_text())


async def command(process, name, action, cases=()):
    temp = ROOT / "actor-commands" / f".{name}.tmp"
    save_json(temp, {"id": name, "action": action, "cases": cases})
    temp.rename(temp.with_name(f"{name}.json"))
    if action == "quit":
        return
    return await wait_file(ROOT / f"actor-{name}.json", process)


def compare_values(name, cases, rollout_scores, actor_scores, version, kind="actor_vs_rollout"):
    rows = []
    for position, (case, x, y) in enumerate(zip(cases, rollout_scores, actor_scores, strict=True)):
        expected = len(case["response_token_ids"])
        assert expected > 0 and x.numel() == y.numel() == expected
        assert x.dtype == y.dtype == torch.float32
        assert bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        xb = x.reshape(-1).view(torch.int32)
        yb = y.reshape(-1).view(torch.int32)
        changed = xb != yb
        row = dict(
            comparison_kind=kind,
            case_id=case["id"],
            configuration=name,
            weight_version=version,
            batch_size=len(cases),
            batch_position=position,
            expected_tokens=expected,
            compared_tokens=expected,
            mismatch_count=int(changed.sum()),
            dtype=str(x.dtype),
            all_finite=True,
        )
        if changed.any():
            i = int(changed.nonzero()[0])
            row["first_mismatch"] = dict(
                response_offset=i,
                token_id=case["response_token_ids"][i],
                target_position=len(case["prompt_token_ids"]) + i,
                causal_input_position=len(case["prompt_token_ids"]) + i - 1,
                rollout=x[i].item(),
                actor=y[i].item(),
                rollout_bits=f"{int(xb[i]) & 0xFFFFFFFF:08x}",
                actor_bits=f"{int(yb[i]) & 0xFFFFFFFF:08x}",
            )
        rows.append(row)
    save_json(ROOT / f"compare-{name}.json", rows)
    print("STRICT_COMPARISON", json.dumps(rows), flush=True)
    return rows


def compare(name, cases, rollout_scores, actor_artifact, version):
    actor = torch.load(actor_artifact, weights_only=True, map_location="cpu")
    assert actor["weight_version"] == version and actor["cases"] == cases
    return compare_values(name, cases, rollout_scores, actor["selected"], version)


def generated_scores(name, case, version):
    files = sorted((ROOT / "raw" / name).glob("*.pt"))
    records = [torch.load(p, weights_only=True, map_location="cpu") for p in files]
    assert len(records) == len(case["response_token_ids"])
    values = []
    for token, record in zip(case["response_token_ids"], records, strict=True):
        assert record["weight_version"] == version and record["mode"] == "raw_logprobs"
        assert int(record["token_ids"][0, 0]) == token
        values.append(record["raw_logprobs"][0, 0])
    return torch.stack(values)


def teacher_scores(name, cases, version):
    records = [
        torch.load(p, weights_only=True, map_location="cpu")
        for p in sorted((ROOT / "raw" / name).glob("*.pt"))
    ]
    rows = []
    for record in records:
        assert record["weight_version"] == version and record["mode"] == "raw_logprobs"
        ids = record["token_ids"][:, 0]
        for case in cases:
            target = torch.tensor(
                case["prompt_token_ids"] + case["response_token_ids"], dtype=torch.int64
            )[1:]
            for start in range(max(0, ids.numel() - target.numel() + 1)):
                if torch.equal(ids[start : start + target.numel()], target):
                    lo = start + len(case["prompt_token_ids"]) - 1
                    selected = record["raw_logprobs"][
                        lo : lo + len(case["response_token_ids"]), 0
                    ].clone()
                    rows.append((case["id"], selected))
    result = []
    for case in cases:
        matches = [v for key, v in rows if key == case["id"]]
        assert len(matches) == 1, (case["id"], len(matches), "unaligned prompt-score capture")
        result.append(matches[0])
    return result


async def main():
    from examples.operators.dsv4_4l.audit_dependencies import audit

    save_json(ROOT / "dependency-audit.json", audit())
    import ray
    from verl.workers.config import HFModelConfig, RolloutConfig
    from verl.workers.rollout.replica import RolloutMode
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    ray.init(
        num_cpus=8,
        num_gpus=1,
        include_dashboard=False,
        object_store_memory=256 * 1024**2,
        _memory=4 * 1024**3,
        _temp_dir="/cache/ray-parity",
        log_to_driver=True,
    )
    cfg = RolloutConfig(
        name="vllm",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
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
                "worker_cls": "rl_engine.integrations.engines.rollout.vllm.dsv4_worker.RLKDeepSeekV4Worker",
                "distributed_executor_backend": "uni",
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
        gpus_per_node=1,
        nnodes=1,
        cuda_visible_devices="0",
    )
    process = None
    results = []
    before_passed = False
    update_attempted = False
    update_executed = False
    sync_confirmed = False
    try:
        await server.launch_server()
        first_ack = await server.collective_rpc("set_case", args=("generate-A-v0",))
        save_json(ROOT / "rollout-start-ack.json", first_ack)
        prompt = [0] + [100] * 128
        output = await server.generate(
            prompt,
            {"max_tokens": 2, "temperature": 0.0, "logprobs": True, "ignore_eos": True},
            request_id="generated-A-v0",
        )
        actual = output.model_dump()
        case = dict(id="A", prompt_token_ids=prompt, response_token_ids=actual["token_ids"])
        assert len(case["response_token_ids"]) == 2
        save_json(ROOT / "cases-v0.json", [case])
        scores = generated_scores("generate-A-v0", case, 0)
        env = dict(os.environ)
        env.pop("ROCR_VISIBLE_DEVICES", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["HIP_VISIBLE_DEVICES"] = os.environ["DSV4_PHYSICAL_GPU"]
        (ROOT / "actor-commands").mkdir()
        process = subprocess.Popen(
            [sys.executable, "-m", "examples.operators.dsv4_4l.actor_service"],
            env=env,
            stdout=(ROOT / "actor-service.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        await wait_file(ROOT / "actor-ready.json", process)
        await command(process, "v0-A", "infer", [case])
        initial = compare("v0-A", [case], [scores], ROOT / "actor-v0-A.pt", 0)
        results += initial
        if any(r["mismatch_count"] for r in initial):
            save_json(
                ROOT / "results.json",
                dict(
                    before_passed=False,
                    after_passed=False,
                    pure_rlk_passed=False,
                    update_executed=False,
                    rows=results,
                    stop_reason="first same-weight case mismatched; original traces retained",
                ),
            )
            return

        references = {}

        async def score_fixed(name, cases, version):
            await server.collective_rpc("set_case", args=(name,))

            async def run(case):
                tokens = case["prompt_token_ids"] + case["response_token_ids"]
                sampling = SamplingParams(
                    max_tokens=1, temperature=0, prompt_logprobs=1, ignore_eos=True
                )
                async for item in server.engine.generate(
                    TokensPrompt(prompt_token_ids=tokens), sampling, f"{name}-{case['id']}"
                ):
                    pass

            await asyncio.gather(*(run(c) for c in cases))
            metadata = await server.collective_rpc("case_batches")
            save_json(ROOT / f"rollout-batches-{name}.json", metadata)
            if len(cases) > 1:
                assert any(
                    b["num_requests"] == len(cases)
                    for worker in metadata
                    for b in worker["batches"]
                ), "scheduler did not execute the requested real batch"
            if len(cases) > 1:
                expected_inputs = [c["prompt_token_ids"] + c["response_token_ids"] for c in cases]
                assert any(
                    b["input_token_rows"] == expected_inputs
                    for worker in metadata
                    for b in worker["batches"]
                ), "actual rollout batch order differs from requested case order"
            values = teacher_scores(name, cases, version)
            await command(process, name, "infer", cases)
            rows = compare(name, cases, values, ROOT / f"actor-{name}.pt", version)
            results.extend(rows)
            actor = torch.load(ROOT / f"actor-{name}.pt", weights_only=True, map_location="cpu")
            for i, c in enumerate(cases):
                key = (version, c["id"])
                if key in references:
                    old_rollout, old_actor = references[key]
                    for kind, current, previous in (
                        ("rollout_repeat_or_batch_position", values[i], old_rollout),
                        ("actor_repeat_or_batch_position", actor["selected"][i], old_actor),
                    ):
                        extra = compare_values(
                            name + "-" + c["id"] + "-" + kind,
                            [c],
                            [current],
                            [previous],
                            version,
                            kind,
                        )
                        extra[0].update(batch_size=len(cases), batch_position=i)
                        save_json(ROOT / f"compare-{name}-{c['id']}-{kind}.json", extra)
                        results.extend(extra)
                        rows.extend(extra)
                else:
                    references[key] = (values[i].clone(), actor["selected"][i].clone())
            return not any(r["mismatch_count"] for r in rows)

        async def generate_case(case_id, prompt, count, version):
            name = f"generate-{case_id}-v{version}"
            await server.collective_rpc("set_case", args=(name,))
            output = await server.generate(
                prompt,
                {"max_tokens": count, "temperature": 0.0, "logprobs": True, "ignore_eos": True},
                request_id=name,
            )
            case = dict(
                id=case_id,
                prompt_token_ids=prompt,
                response_token_ids=output.model_dump()["token_ids"],
            )
            assert len(case["response_token_ids"]) == count
            await command(process, name, "infer", [case])
            rows = compare(
                name,
                [case],
                [generated_scores(name, case, version)],
                ROOT / f"actor-{name}.pt",
                version,
            )
            results.extend(rows)
            if any(r["mismatch_count"] for r in rows):
                raise RuntimeError(f"{name} generated token comparison mismatched")
            return case

        async def matrix(version, cases):
            a, b, c = cases
            ok = True
            for label, group in (
                ("A", [a]),
                ("B", [b]),
                ("C", [c]),
                ("AB", [a, b]),
                ("BA", [b, a]),
                ("A-repeat", [a]),
                ("AB-repeat", [a, b]),
            ):
                passed = await score_fixed(f"v{version}-{label}-matrix", group, version)
                ok = ok and passed
            return ok

        # First prove a fixed-sequence independent replay before updating anything.
        if not await score_fixed("v0-A-repeat", [case], 0):
            raise RuntimeError("pre-update fixed-sequence replay mismatched")
        b = await generate_case("B", [0] + [101] * 127, 3, 0)
        c = await generate_case("C", [0, 104, 105, 106, 107, 108, 109, 110, 111], 1, 0)
        cases = [case, b, c]
        save_json(ROOT / "cases-v0.json", cases)
        before_passed = await matrix(0, cases)
        if not before_passed:
            raise RuntimeError("expanded pre-update comparison mismatched; update not attempted")
        update_attempted = True
        update = await command(process, "step", "step", [case])
        update_executed = True
        assert update["weight_version"] == 1 and update["before"] != update["after"]
        ack = await server.collective_rpc("load_actor_update", args=("/outputs/actor-update.pt",))
        save_json(ROOT / "rollout-sync-ack.json", ack)
        after_ack = await server.collective_rpc("set_case", args=("v1-A-fixed",))
        save_json(ROOT / "rollout-after-ack.json", after_ack)
        assert [r["pid"] for r in first_ack] == [r["pid"] for r in after_ack]
        assert all(r["weight_version"] == 1 for r in after_ack)
        sync_confirmed = True
        post = await score_fixed("v1-A-fixed", [case], 1)
        post = await matrix(1, cases) and post
        new_cases = []
        for old in cases:
            new_cases.append(
                await generate_case(
                    old["id"] + "-new", old["prompt_token_ids"], len(old["response_token_ids"]), 1
                )
            )
        save_json(ROOT / "cases-v1-generated.json", new_cases)
        save_json(
            ROOT / "results.json",
            dict(
                before_passed=True,
                after_passed=post,
                pure_rlk_passed=post,
                update_executed=True,
                same_rollout_process=True,
                rows=results,
                expanded_coverage_passed=post,
                limitation="independent container repeat must be evaluated separately",
            ),
        )
    except Exception as error:
        step_status = ROOT / "actor-step-status.json"
        if step_status.exists():
            status = json.loads(step_status.read_text())
            update_executed = bool(status["optimizer_returned"] and status["parameter_changed"])
        save_json(
            ROOT / "results.json",
            dict(
                before_passed=before_passed,
                after_passed=False,
                pure_rlk_passed=False,
                update_attempted=update_attempted,
                update_executed=update_executed,
                sync_confirmed=sync_confirmed,
                rows=results,
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                await command(process, "zz-quit", "quit")
                try:
                    await asyncio.wait_for(asyncio.to_thread(process.wait), 30)
                except asyncio.TimeoutError:
                    process.terminate()
        if hasattr(server, "engine"):
            server.engine.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
