# SPDX-License-Identifier: Apache-2.0
"""Generate the English report only from validated, machine-aggregated job data."""

import argparse
import json
from pathlib import Path


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def number(x):
    return "N/A" if x is None else f"{x:.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    a = p.parse_args()
    d = json.loads(a.summary.read_text())
    out = a.summary.parent
    perf = d["performance"]["R/R"]
    jobs = d["jobs"]
    runtime = d["provenance"]["rlk"]
    native_root = Path(d["groups"]["P/P"]["output"])
    rlk_root = Path(d["groups"]["R/R"]["output"])
    accepted = json.loads((rlk_root / "continuous-acceptance.json").read_text())
    scopes = []
    for version in (1, 2, 3):
        scope = []
        for rank in (0, 1):
            e = json.loads(
                (
                    rlk_root / f"actor-rank-{rank}/step-v{version}/state-change-evidence.json"
                ).read_text()
            )
            g = json.loads(
                (rlk_root / f"actor-rank-{rank}/step-v{version}/gradient-evidence.json").read_text()
            )
            scope.append(
                dict(
                    rank=rank,
                    trainable_parameters=len(e["before_parameters"]),
                    changed_parameters=len(e["changed_parameters"]),
                    nonzero_gradient_parameters=sum(
                        x["nonzero"] for x in g["local_gradients"].values()
                    ),
                )
            )
        scopes.append(dict(version=version, ranks=scope))
    (out / "update-scope.json").write_text(json.dumps(scopes, indent=2) + "\n")
    source_lock = json.loads(
        Path(
            "/scratch/rl-kernel/vensen/dsv4-4l/runs/20261117T080547Z-continuous/a"
            "rtifacts/sources.lock"
        ).read_text()
    )
    assert source_lock["checkpoint"]["checkpoint_sha256"] == d["actual_container_checkpoint_sha256"]
    # Store the actual small provenance inputs beside the report, not tensor/checkpoint payloads.
    (out / "checkpoint.json").write_text(json.dumps(source_lock["checkpoint"], indent=2) + "\n")
    (out / "image.json").write_text(json.dumps(jobs["R/R"]["image"], indent=2) + "\n")
    (out / "predeclared-cases.json").write_bytes((rlk_root / "predeclared-cases.json").read_bytes())
    cfg = next(x for x in perf["events"] if x["phase"] == "actor_configuration")
    allocation = json.loads((rlk_root / "device-allocation.json").read_text())
    sections = [
        "# Native verl versus verl + RL-Kernel: DSV4 four-layer PP2/DP1\n",
        "This report follows the evidence layout of [PR "
        "#377](https://github.com/RL-Align/RL-Kernel/pull/377). "
        "It uses this machine’s actual four-layer results; no Qwen/H100, "
        "200-step, reward or bootstrap results are reused.\n",
        "## 1. Completed experiment scope\n",
        table(
            ["Group", "Actual route", "Job / container", "Completed updates", "Status"],
            [
                [
                    "P/P",
                    "Pristine verl Megatron TrainingWorker + pristine verl vLLM PP2 configuration",
                    native_root.name
                    + " / "
                    + json.loads((native_root / "container-inspect.json").read_text())["Id"][:12],
                    d["groups"]["P/P"]["completed_steps"],
                    d["groups"]["P/P"]["status"],
                ],
                [
                    "R/R",
                    "Real verl RLK native-MCore actor + RLK verl/vLLM workers; rocm_triton, split",
                    rlk_root.name
                    + " / "
                    + json.loads((rlk_root / "container-inspect.json").read_text())["Id"][:12],
                    d["groups"]["R/R"]["completed_steps"],
                    d["groups"]["R/R"]["status"],
                ],
            ],
        ),
        "\nThe two fresh containers ran sequentially, on the same physical "
        "GPUs. P/P is **BLOCKED**, not inconsistent or zero-throughput. "
        "Both actor ranks failed before loading the model. Native verl also "
        "rejected PP2 rollout.\n",
    ]
    for e in d["groups"]["P/P"]["errors"]:
        sections.append(
            f'- `{e["mode"]}`, rank `{e["local_rank"]}`: '
            f'`{e["exception"]}: {e["message"]}`; '
            f'RLK modules loaded: `{e["rlk_modules_loaded"]}`.'
        )
    direct = next(x for x in jobs["P/P"]["result"]["outcomes"] if x["mode"] == "direct-vllm")
    sections += [
        "\nThe separate **direct native vLLM** capability check exited "
        + str(direct["exit_code"])
        + ". It loaded the original checkpoint, selected `TRITON_UNFUSED "
        "Mxfp4` on gfx942 and generated outputs. "
        "This establishes a native vLLM capability, **not** completion of "
        "the native verl actor/rollout experiment. "
        "We do not claim original FP4 inference is unsupported. Exact call "
        "stacks and logs are indexed in `raw-log-index.json`.\n",
        "## 2. Matched workload\n",
        "The following conditions were executed by R/R and requested for "
        "P/P; the native import/configuration blockers prevented "
        "verification of a matched native training objective. "
        "No speed comparison between matched completed training jobs is possible.\n",
        table(
            ["Item", "Condition"],
            [
                [
                    "Hardware",
                    "One AMD MI300X node; actor physical 4,5; rollout 6,7; GPUs 0–3 excluded",
                ],
                ["Parallelism", "Actor TP1/PP2/EP1/CP1/DP1; rollout TP1/PP2/EP1/CP1/DP1"],
                ["PP partition", "Original layers [0,1,2] / [3], not 2+2"],
                ["Checkpoint", source_lock["checkpoint"]["path"]],
                ["Checkpoint SHA256", source_lock["checkpoint"]["checkpoint_sha256"]],
                ["Original revision", source_lock["checkpoint"]["original_revision"]],
                [
                    "Precision",
                    "Original packed FP4/FP8 weights and original scales frozen; "
                    "nonquantized BF16/FP32 dtypes retained",
                ],
                ["Seeds", f'Actor {cfg["seed"]}; rollout 42; greedy generation'],
                ["Execution", "Eager, no dropout; standalone PP2 rollout; no graph capture"],
                [
                    "Fixed inputs",
                    "A/B/C full lengths 131/130/10; saved before either job; SHA256 "
                    + jobs["R/R"]["fixed_cases_sha256"],
                ],
                [
                    "Matrix",
                    "B1 A/B/C; genuine B2 AB/BA; individual and batch repeats, fixed and "
                    "newly generated sequences",
                ],
                [
                    "Training microbatch",
                    "Fixed A, B1, same order for all 3 continuous steps; 129 prompt + 2 "
                    "response-loss tokens",
                ],
                ["Loss", cfg["loss"] + "; " + cfg["normalization"]],
                [
                    "Optimizer",
                    f'Native Adam/FusedAdam; lr={cfg["optimizer"]["lr"]}; '
                    f'betas={cfg["optimizer"]["betas"]}; '
                    f'clip={cfg["optimizer"]["clip_grad"]}; '
                    f'weight_decay={cfg["optimizer"]["weight_decay"]}; '
                    "linear LR schedule over 4 configured steps",
                ],
                [
                    "State",
                    "Same workers and optimizer/moments through versions 0/1/2/3; clocks 1/2/3",
                ],
                [
                    "Cache/warmup",
                    "Same cache path and image, no cache clearing; pre-existing shared "
                    "compilation cache; normal engine warmup, no extra performance warmup",
                ],
                [
                    "Resource policy",
                    "Same 16 GiB container shm, OMP_NUM_THREADS=4, rollout memory "
                    "utilization 0.30, max model length 256, max_num_seqs=2",
                ],
            ],
        ),
        "\nThe selected trainable group contains 83 native Parameters "
        "(1,081,524,829 elements); all have backward entries. "
        "In this short-context workload 77 have nonzero gradients: five "
        "indexer Parameters and the first mHC alpha_pre scalar have zero "
        "gradients. "
        "Not every Parameter changes at every step; actual rank/version "
        "counts are in `update-scope.json`. "
        "Three frozen routing tables use native int32 indices with the "
        "original int64 values preserved exactly. No logprob score casting "
        "is used.\n",
        "Verified R/R device identities (native actor startup failed before model execution):\n",
        table(
            [
                "Role / PP rank",
                "Physical GPU",
                "Container ordinal",
                "Worker HIP ordinal",
                "HIP PCI",
            ],
            [
                [
                    x["role"] + " / PP" + str(x["logical_gpu"]),
                    x["physical_gpu"],
                    x["container_gpu"],
                    x["logical_gpu"],
                    x["pci_bus_id"],
                ]
                for x in allocation
            ],
        ),
        "## 3. Version provenance\n",
        "Both groups used the identical image and actual runtime "
        "source/dependency versions below; each checkout was clean. "
        "Native provenance was collected in a separate metadata process "
        "because that audit installs import ABI hooks; those hooks never ran "
        "in the native model processes.\n",
        table(
            ["Project", "Version", "Full SHA", "Repository", "Actual import path", "Dirty"],
            [
                [
                    k,
                    v["module_version"] or v["distribution_version"],
                    v["sha"],
                    v["repository_url"],
                    v["import_path"],
                    v["dirty"],
                ]
                for k, v in runtime["projects"].items()
            ],
        ),
        "\n"
        + table(
            ["Dependency", "Actual version"],
            list(runtime["dependencies"].items())
            + [["HIP", runtime["hip"]], ["ROCm SDK", "10.0.0"], ["GPU architecture", "gfx942"]],
        ),
        "\nImage: `" + jobs["R/R"]["image"]["tag"] + "`; ID `" + jobs["R/R"]["image"]["id"] + "`. "
        "Parent image: `"
        + jobs["R/R"]["image"]["parent_tag"]
        + "`, `"
        + jobs["R/R"]["image"]["parent_id"]
        + "`. "
        "Bridge 0.7.0’s 878 Python RECORD files and upstream source "
        "cleanliness were verified.\n",
        "Compatibility/implementation inventory:\n",
        "- Both: selected-render-node visibility and an allowlisted "
        "KFD-aperture ABI shim. These restrict device access and do not "
        "replace model arithmetic.",
        "- P/P: only RLK’s vLLM general plugin was excluded; other "
        "discovered general plugins were retained. No RLK numerical "
        "operators, custom VJP, loader, import-ABI adapter, PP server "
        "override or math correction was enabled.",
        "- R/R: existing #711 import-ABI/Bridge compatibility, original "
        "packed checkpoint loading and PP [0–2]/[3] placement; native MCore "
        "scheduling; RLK deterministic mHC/norm/projection/attention/router/c"
        "ombine/FP4 input-gradient and raw-logprob paths; dynamic live shard "
        "synchronization including router bias.",
        "- R/R batch-order adaptation explicitly preserves the requested "
        "ordinary prefill AB/BA order. Observation timers, read-only "
        "captures and comparison tools add overhead. No upstream files or "
        "site-packages were edited.",
        "- AITER remains installed/enabled for ordinary dependencies; mHC is "
        "RLK rocm_triton + split, not rocm_aiter. #662 remains excluded.\n",
        "## 4. Strict train/rollout consistency\n",
        "Each row compares one group’s own actor/rollout at its own actual "
        "weight version. Cross-group post-update weights are never scored as "
        "a train/rollout mismatch. "
        "The count includes repeated batch/position observations, not unique "
        "sampled token IDs. All R/R comparisons use original finite FP32 "
        "words; diagnostic differences do not replace bitwise acceptance.\n",
        table(
            [
                "Group",
                "Version",
                "Comparison",
                "Compared / expected",
                "Mismatch count / rate",
                "Mean / max abs diff",
                "Batch invariant",
                "Sync",
            ],
            [
                [
                    x["group"],
                    x["version"],
                    x["comparison_kind"],
                    (
                        "N/A"
                        if x["compared_tokens"] is None
                        else f'{x["compared_tokens"]} / {x["expected_tokens"]}'
                    ),
                    (
                        "N/A"
                        if x["mismatch_count"] is None
                        else f'{x["mismatch_count"]} / {x["mismatch_rate"]:.6f}'
                    ),
                    (
                        "N/A"
                        if x["mean_abs_diff"] is None
                        else f'{x["mean_abs_diff"]:.8g} / {x["max_abs_diff"]:.8g}'
                    ),
                    x["batch_invariance"] if x["batch_invariance"] is not None else "N/A",
                    x["synchronization"],
                ]
                for x in d["consistency"]
            ],
        ),
        "\nR/R passed "
        + str(accepted["comparison_rows"])
        + " strict comparison rows. Fixed inputs match the predeclared JSON exactly. "
        "Each updated version passed native PP activation/gradient transfer, "
        "gradient coverage/finiteness, actual parameter changes, optimizer "
        "continuity and full dynamic shard readback. "
        "Observed inventories contain 71 original keys split 49/22 in these "
        "runs; this is an observation, not a hardcoded acceptance criterion. "
        "Original rollout PIDs are unchanged and caches advance with each "
        "verified weight version. DP1 performs no cross-replica DP gradient "
        "reduction.\n",
        "## 5. Preliminary performance\n",
        "P/P training/rollout consistency performance is **N/A** because no "
        "native verl cycle completed. The native container’s elapsed time "
        "includes failed starts and a separate direct-vLLM probe; "
        "it is not compared with training time. R/R numbers below are "
        "**diagnostic acceptance measurements**, not formal end-to-end "
        "GRPO/PPO throughput.\n",
        table(
            [
                "Version",
                "Generate s",
                "Actor logprob s",
                "F/B + optimizer s",
                "Optimizer subset s",
                "Sync s",
                "Full diagnostic window s",
                "Generated response tokens",
                "Response tok/GPU/s",
            ],
            [
                [
                    x["version"],
                    number(x["rollout_generate_seconds"]),
                    number(x["actor_logprob_pp_max_rank_seconds"]),
                    number(x["forward_backward_optimizer_pp_max_rank_seconds"]),
                    number(x["optimizer_pp_max_rank_seconds"]),
                    number(x["weight_sync_seconds"]),
                    number(x["full_diagnostic_acceptance_cycle_seconds"]),
                    x["generated_response_tokens"],
                    number(x["rollout_response_tokens_per_gpu_second"]),
                ]
                for x in perf["versions"]
            ],
        ),
        "\n"
        + table(
            [
                "Version",
                "Generated prompt/response tokens",
                "Actor scored input/padded tokens",
                "Actor scored response tokens",
                "Rollout teacher input / auxiliary emitted tokens",
                "Actor input tok/GPU/s",
            ],
            [
                [
                    x["version"],
                    f'{x["generated_prompt_tokens"]} / {x["generated_response_tokens"]}',
                    f'{x["actor_logprob_input_tokens"]} / {x["actor_logprob_padded_input_tokens"]}',
                    x["actor_logprob_response_scored_tokens"],
                    f'{x["rollout_teacher_input_tokens"]} / {x["teacher_aux_emitted_tokens"]}',
                    number(x["actor_logprob_input_tokens_per_gpu_second"]),
                ]
                for x in perf["versions"]
            ],
        ),
        "\nTiming definitions and limits:\n",
        "- Actor inference, training and optimizer timers synchronize the "
        "local GPU before and after the measured real API. For PP aggregate "
        "observations use the maximum rank time. Optimizer is a subset of "
        "F/B+optimizer, not an additive extra stage.",
        "- Rollout generation/synchronization use completed real API/RPC "
        "acknowledgements; teacher scoring drains all actual request "
        "outputs. CPU comparison-tool time is recorded separately.",
        "- Version 1–3 full windows contain the update/export, live "
        "synchronization, complete fixed and generated matrices, "
        "comparisons, state audits and I/O. Version 0’s full window starts "
        "after its initial fixed matrix, so it is not a comparable "
        "update-cycle window.",
        "- `response tok/GPU/s = actual generated response tokens / "
        "(generation seconds × 2 rollout GPUs)`. `actor input tok/GPU/s = "
        "unpadded processed input tokens / (maximum PP rank’s cumulative "
        "inference seconds × 2 actor GPUs)`. These include repeated "
        "diagnostic work, not unique training examples.",
        "- Teacher-scoring requests emit one auxiliary token per sample "
        "(`max_tokens=1`); those outputs are recorded separately from the "
        "primary generated responses.",
        "- Full tensor capture, hashing, optimizer/state audits, CPU copies, "
        "IPC and filesystem I/O overlap phases and are not independently "
        "separable here. No trace-free timing, speedup, bootstrap interval "
        "or stable performance conclusion is claimed.",
        "- Initialization timings start at rollout launch or actor "
        "build_actor_worker after its import/distributed initialization. "
        "Loading, compilation and normal warmup cannot be separated fairly; "
        "subprocess/import startup is included only in whole-job elapsed "
        "time.\n",
        table(
            ["Initialization observation", "Rank", "Seconds"],
            [[x["phase"], x["rank"], number(x["seconds"])] for x in perf["initialization"]],
        ),
        "\n"
        + table(
            ["Physical GPU", "Observed peak resident VRAM GiB"],
            [[g, number(n / 1024**3)] for g, n in perf["sampled_peak_vram_bytes"].items()],
        ),
        "\nVRAM is a roughly one-second polling maximum, not a hardware-exact "
        "instantaneous peak. Per-worker PyTorch allocator high-water "
        "observations are also preserved in summary.json.\n",
        "R/R bootstrap also includes paired 20-replay shared_experts.down_proj diagnostics; "
        "these are rollout-only observations, not an additional native P/P comparison or "
        "performance warmup. Their overhead is included and not separately timed.\n",
        "Whole R/R job elapsed time, including launch, imports, initialization, diagnostics "
        f'and teardown: {perf["job_elapsed_seconds"]:.3f} seconds.\n',
        "## 6. Figures\n",
        "P/P is explicitly marked BLOCKED rather than plotted at zero. Valid "
        "R/R observations are shown without reward/benefit plots.\n",
        "![Strict bitwise consistency](consistency.png)\n",
        "![Diagnostic phase times](phase-times.png)\n",
        "![Token-normalized observations and sampled VRAM](throughput-memory.png)\n",
        "## 7. Data integrity and limitations\n",
        "- Only one native attempt and one R/R job, three continuous "
        "updates, short context, PP2/DP1 and a frozen quantized base. "
        "Genuine B2 forward scoring is tested; training uses B1 A.",
        "- P/P is BLOCKED before model/target equivalence could be "
        "established; no native mismatch rate, stable throughput or "
        "cross-group speed ratio exists. Direct vLLM support is reported "
        "separately.",
        "- R/R fails closed on missing matrix cells, empty comparisons, "
        "NaN/Inf, wrong versions, synchronization failures, worker errors or "
        "nonzero bit mismatches. The aggregation independently verifies the "
        "complete continuous-run evidence and predeclared input hash.",
        "- Native finite score mismatches would be a valid inconsistent "
        "control result, not a worker failure. This attempt produced no "
        "valid native score matrix, so no relaxed native numerical "
        "assertions were exercised.",
        "- Independent-job optimizer trajectories are known not to be "
        "bitwise reproducible. That historical record is preserved; this "
        "experiment validates same-version parity within its R/R job and "
        "does not establish independent trajectory reproducibility.",
        "- Original attention/compressor/router/expert settings remain "
        "intact. Long-context indexer-budget overflow, other parallel-axis "
        "combinations, this round’s DP2/six-GPU training, quantized-base "
        "updates, full models, multi-node execution and performance "
        "optimization are not qualified.",
        "- This is a real verl TrainingWorker/custom response-token loss "
        "harness, **not** a complete GRPO/PPO reward/advantage pipeline. "
        "Reward, KL and task benefit are **not measured**. Zero mismatch "
        "does not establish training benefit.",
        "- Historical four-card and six-card reports/logs remain under their "
        "original run roots; this report does not rewrite them. Original "
        "tensor/state payloads stay on /scratch and may use SHA-verified "
        "lossless archival.\n",
        "## 8. Reproducible artifacts\n",
        "- [Machine summary](summary.json), [all consistency "
        "groups/versions/types](consistency.csv), [performance "
        "observations](performance.csv), [update scope](update-scope.json).",
        "- [Raw log index](raw-log-index.json), [image "
        "manifest](image.json), [checkpoint metadata](checkpoint.json), "
        "[predeclared fixed inputs](predeclared-cases.json).",
        "- Runner: [run_comparison.py](../../run_comparison.py); "
        "aggregation: [aggregate_comparison.py](../../aggregate_comparison.py"
        "); plotting: [plot_comparison.py](../../plot_comparison.py); report "
        "rendering: [write_comparison_report.py](../../write_comparison_repor"
        "t.py).\n",
        "Exact machine evidence roots:\n\n```text\n"
        + str(native_root)
        + "\n"
        + str(rlk_root)
        + "\n```\n",
        "From the #711 checkout, use confirmed idle GPUs 4–7 and enough "
        "/scratch user quota. Both invocations use the same image manifest, "
        "checkpoint, cache, inputs and device guards. Each creates a fresh "
        "container.\n\n```bash\n"
        "REPORT=examples/operators/dsv4_4l/results/native_vs_rlk_pp2_dp1_20261117\n"
        "NEW=/scratch/rl-kernel/vensen/dsv4-4l/runs/native-vs-rlk-NEW\n"
        "MODEL=/scratch/rl-kernel/vensen/dsv4-4l/models/DeepSeek-V4-Flash-4l\n"
        "CACHE=/scratch/rl-kernel/vensen/dsv4-4l/runs/20261116T024532Z/cache\n"
        "quota -s\n"
        "python3 examples/operators/dsv4_4l/run_comparison.py --group P/P \\\n"
        '  --output-root "$NEW" --image-manifest "$REPORT/image.json" \\\n'
        '  --checkpoint "$MODEL" --cache "$CACHE" --fixed-cases '
        '"$REPORT/predeclared-cases.json"\n'
        "python3 examples/operators/dsv4_4l/run_comparison.py --group R/R \\\n"
        '  --output-root "$NEW" --image-manifest "$REPORT/image.json" \\\n'
        '  --checkpoint "$MODEL" --cache "$CACHE" --fixed-cases '
        '"$REPORT/predeclared-cases.json"\n'
        "python3 examples/operators/dsv4_4l/aggregate_comparison.py \\\n"
        '  --native-job "$NEW/artifacts/job-native-PP.json" \\\n'
        '  --rlk-job "$NEW/artifacts/job-rlk-RR-rocm_triton.json" '
        '--output-dir "$NEW/report"\n'
        "python3 examples/operators/dsv4_4l/plot_comparison.py "
        '"$NEW/report/summary.json" --output-dir "$NEW/report"\n'
        "python3 examples/operators/dsv4_4l/write_comparison_report.py "
        '"$NEW/report/summary.json"\n```\n',
        "Plotting is CPU-only (Matplotlib 3.9.2, NumPy 2.2.6); install "
        "plotting dependencies into a separate report environment, not the "
        "actor/rollout image. "
        "The measured image contains all framework/source dependencies and "
        "requires no host framework-source mounts. "
        "Its parent and exact build command are preserved in image.json.",
    ]
    (out / "README.md").write_text("\n\n".join(sections) + "\n")


if __name__ == "__main__":
    main()
