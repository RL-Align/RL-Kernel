# CLAUDE.md — DSv4 CUDA work on `test-dsv4-cuda` (PR #635)

Notes for an assistant (or developer) continuing the DSv4 SM90 batch-invariant work. The
design of record is `docs/architecture/dsv4-design.md`: goal, environment, how to build
and test, module map, decisions, the reduction order of every module, status and open
work. Read it first and keep its status section current. Last updated 2026-11-20.

## Working rules

- Commits: conventional style (`feat(dsv4): ...`, `fix(dsv4): ...`), `git commit -s` (DCO),
  end with the Co-Authored-By line. Split pure formatting from logic changes. Lines <= 100
  characters.
- Commit verified work without waiting for approval (owner, 2026-11-10). **Push only when
  asked.** If a design decision belongs to the owner, take the recommended default,
  document it, and list it in the report so the owner can revisit it.
- Reply to the owner in Traditional Chinese.
- Every forward change must keep the DSv4 regression green (design doc 2.3) and the layerwise
  gate at zero (`scripts/check_dsv4_layerwise_gate.py`). New kernels get a PyTorch golden
  defined on CUDA, a byte-equality test and a batch-invariance test.
- A kernel that cannot hold an existing numeric profile declares a new one.
- Performance gate: 1.3x of the non-invariant path, measured with CUDA-graph replay.

## Owner decisions not to re-open

The technical ones are in the design doc (4.2). In short:

- **Target: LoRA training.** Base weights (norms, mHC, router, experts, embedding) are
  frozen, so their backward is dX only. Adapters on attention `wq_a` / `wkv` / `wq_b` /
  `wo_b` (Megatron: `linear_q_down_proj` / `linear_kv_proj` / `linear_q_up_proj` /
  `linear_proj`), routed `fc1` / `fc2` (one adapter shared by all experts), shared expert.
  Unmerged: `h = BF16(x A^T)`, `d = h B^T` in FP32, `y = BF16(FP32(y_base) + scale d)`; the
  rollout runs the same forward.
- **Forward bitwise batch-invariant; backward need not be deterministic** (cuDNN's atomic
  dKV stays; deterministic dKV is deprioritized).
- **mHC:** `rsqrt(mean(x^2) + eps)` (DeepSeek / vLLM), its own profile, the controller GEMM
  on DeepGEMM `tf32_hc_prenorm_gemm` with a fixed 16-way split.
- **RoPE:** plain on window-only layers, YaRN on compressed layers (vLLM uses YaRN on all;
  the rollout plugin passes our tables).
- **Per-expert LoRA deferred** until verl needs it (then grouped BF16 GEMM, delta rounded
  to BF16, profile `dsv4-lora-expert-v1`). Chunked training not planned.
- **verl integration (2026-11-15, "option A"):** the actor is a verl engine backend
  `rlkernel` on `Dsv4Model`, not Megatron, so verl, vLLM and Megatron stay unmodified
  (Megatron would need nearly every operator swapped plus verl's own logprob code).
  Everything lives in `rl_engine/integrations/dsv4/` as plugins. Re-checked 2026-11-20
  (design doc 10.2): a Megatron actor works only by swapping whole decoder layers, the
  weight import, LoRA and log-probs; not now.
- **Deliverable:** RL-Kernel + verl (Hopper, BI, LoRA) against verl #7483 (Hopper, not
  BI, LoRA), with an ablation that turns BI off in our stack to separate engine and kernel
  effects on speed.

## verl integration: facts that are not obvious from the code

Design doc 10.1 has the pieces. Things that cost time:

- The LoRA rank goes in `actor.rlkernel.lora_rank`. verl's `model.lora_rank` must stay 0,
  or verl turns on vLLM's own LoRA in the rollout. Since verl then sees no LoRA:
  - the reference policy (only with a KL term) is a separate `rlkernel` engine;
  - verl asks for sleep level 2, which would drop the rollout's base weights, so
    `RlkWorker.sleep` keeps level 1.
- Only BF16 adapters are synced. The rollout loads the base weights from the checkpoint
  (`load_format=auto`; the plugin refuses `dummy`, verl's default).
- vLLM pickles its config into the processes it spawns. Unpickling the DSv4 quantization
  config imports vLLM's DSv4 package, and with it vLLM's FlashMLA build, before any plugin
  runs. The fix is the `RlkDeepseekV4FP8Config` subclass registered by `vllm_model.py`.
- The vLLM plugin needs RL-Kernel's `vllm.general_plugins` entry point installed
  (`pip install -e .`).
  The dev env here uses an entry-point-only package instead (`~/dsv4-work/rlk-plugin`).
- **Verdict metric:** verl's own `training/rollout_logprobs_mismatch_count` (FP32 `!=`).
  - verl's `probs_diff` metrics miss 1-ulp differences.
  - On verl #7483's commit, `actor/old_logprobs_mismatch_count` covers the update forward.
- Log-prob targets are `roll(ids, -1)`. A packed sequence's last position is meaningless and
  verl never reads it; exclude it from byte comparisons.
- `persist_w8` keeps E4M3 expert weights expanded: about 6.4 GB per layer extra, in both
  the actor and the rollout (the rollout's copy is outside vLLM's memory budget). The full
  43 layers at EP 8 need it off: `RL_KERNEL_DSV4_PERSIST_W8=0` (same bytes).
- The attention backward is not deterministic, so grad norms differ slightly run to run.
- `VLLM_BATCH_INVARIANT=1` sets NCCL to one channel / Simple / tree in every vLLM worker.
  The plugin undoes it (`_keep_nccl_bandwidth`): the rollout's NCCL only moves bytes.

## Status (2026-11-20) and next

- Done:
  - The kernel set (forward BI, LoRA training end to end; EP = 2 bitwise to EP = 1).
  - The vLLM rollout plugin, bitwise to `Dsv4Model` with LoRA (W1, W2).
  - The `rlkernel` verl engine (W3) and adapter sync (W4).
  - Toy GRPO on 2 x H200 (W5): mismatch 0 at every step; the negative control is nonzero
    from step 2.
  - The rollout's BI-off switch (`RL_KERNEL_DSV4_NON_BI=1`, launcher `BI=0`).
    - On the toy run (1 GPU), `timing_s/gen` is 0.79x of the BI run.
    - It changes the rollout only: the actor's kernels have no non-invariant variant.
- W6 gate (`run.sh gate`) passed on the real checkpoint's first 4 layers (2 x H100,
  2026-11-17, cluster u4478697; the rlkernel / control / ablation / baseline runs are next).
- Full-model comparison (`scripts/verl_e2e_lora/experiments.sh`, README section 8; 8 x H200,
  43 layers, 2026-11-20 / 21): E3 (rlkernel, 32 x 16, response <= 4096) mismatch 0 at all
  100 steps; BL2 (verl #7483, 50 steps) runs; A3 (rollout BI off, 30 steps) 156k to 437k
  mismatched tokens per step. Our actor side is faster, our decode about 2.5x slower per
  step (84 against 34 ms; A3 56 ms, so BI is about 30 ms of the gap). The rollout profile
  (design doc 10.3): graphs work (one replay per step). vLLM's BI mode had capped NCCL at
  one channel, slowing our EP exchange 8x; fixed in the plugin (BI now 12-14 % over BI
  off in the server alone). The rest of the gap to vLLM's TP 8: the EP exchange (vLLM
  needs none), the unmerged LoRA's small kernels, the expert GEMM at decode sizes; those
  are the next performance items. E3 predates the fix. The launcher and the synthetic
  reward moved from `scripts/verl_e2e/` into `scripts/verl_e2e_lora/`, and the old
  verl #7688 guide in `scripts/verl_e2e/` was removed (2026-11-21).
- EP and memory (2026-11-17, real checkpoint's first 4 layers, H100):
  - `rlkernel` engine at EP = 2 / 4 and the rollout plugin at DP = EP = 2 / 4: bitwise to
    EP = 1; EP = 8 too (8 x H200, engine 2026-11-18, rollout 2026-11-19).
  - Activation recompute (`rlkernel.activation_recompute`, `RECOMPUTE=1`): same bytes, held
    activations 1.53 to 0.18 GiB per layer at 4096 tokens.
- Open, roughly in order:
  - verl GRPO at EP 2, 4 and 8 is bitwise (3 steps). 8 GPUs pass the per-user thread
    limit of Slurm job steps (2048) here, so they run over ssh in an salloc (design doc
    section 12). 8, 16 and all 43 layers (8 x H200, EP 8, recompute, persist_w8 off)
    are bitwise too (3 steps, 2026-11-19). The full model with the DAPO math reward
    (32 x 8, 1024 + 2048 tokens) is bitwise at all of 100 steps (2026-11-20); the same
    run continues to 200;
  - CUDA graphs in the rollout: done (2026-11-20, launcher `GRAPH=1`; bytes unchanged,
    `gen` about 2.4x faster on the full model). Under graphs the MoE uses the static EP
    exchange and the adapters must be reserved at load (`RL_KERNEL_DSV4_LORA=rank,alpha`,
    which the launcher sets): a sync copies into them, a different layout is refused;
  - the layerwise gate on the real checkpoint beyond the first 4 layers.

## DSv4.1 (branch `test-dsv41-cuda`, from 2026-11-25)

DeepSeek-V4.1-Flash on the same kernel set. Design of record: `docs/architecture/dsv41-design.md`
(delta against the DSv4 doc; its section 12 has the progress table). Code in
`rl_engine/models/deepseek_v41/` (config on the release `config.json`, loader, attention with
KV / index sharing, Engram, model) plus the `dsv41_*` kernels (mHC carried pre, compressor
ratios 1 / 2, Engram gate, q-RoPE without q-norm). Facts that cost time:

- The release's `rms_norm_eps` is **1e-20** (every RMSNorm, the mHC rms, Engram); `hc_eps`
  stays 1e-6. DSv4 golden constants are 1e-6: DSv4.1 code passes eps explicitly.
- No learned `hc_head`: the output is `norm(collapse(streams, pre))` with the last FFN's pre.
- DeepSeek's reference code ships with the release (`inference/`); release metadata, tokenizer
  and safetensors headers are in `/work/u4478697/dsv4-lora-e2e/models/DeepSeek-V4.1-Flash-meta`
  on cluster u4478697 (`DSV41_META`).
- Forward, training (LoRA, the KV group's cross-layer gradient), recompute done. The
  layerwise gate passes on the release, all 40 layers (2026-11-26, 4 x H200 on `normal2`,
  EP 4 via torchrun, `--engram-tables mmap`).
- Loading the release: keep the Engram tables (94 GiB each) as a read-only shared mmap of
  the shards (a copy-on-write map fails with ENOMEM) and unmap shards after each layer (all
  shards mapped per rank hit ENOMEM too, with `ulimit -v` unlimited).
- vLLM rollout plugin (`rl_engine/integrations/dsv41/`, `RL_KERNEL_DSV41_ROLLOUT=1`) on
  vLLM main `3f86db3` in env `rlk41` (torch 2.14): vLLM keeps scheduling, caches and
  metadata; vLLM's attention layers are built on meta (caches only), `Dsv41Model` computes.
  Toy gate (`scripts/check_dsv41_vllm_gate.py`) PASS, 2026-11-27. Eager only, no prefix
  caching.
- verl GRPO on DSv4.1 (2026-11-27): the `rlkernel` engine loads DSv4.1 by architecture,
  `dsv41/vllm_worker.py` syncs adapters, launchers take `MODEL_FAMILY=dsv41`. Mismatch 0 on
  the toy (2 GPUs) and the release (40 layers, 8 x H200, EP 8, `GPU_UTIL=0.55`,
  `RL_KERNEL_DSV4_PERSIST_W8=0`). 8-GPU Ray runs here go over ssh to the node
  (`jobs/exp8.sbatch`, `jobs/dsv41_grpo.sbatch SSH=1`). An idle `sleep infinity` hold job
  was SIGKILLed after ~30 min: submit real work instead.
- Comparison plan (owner, 2026-11-27): native verl DSv4.1 (NeMo Automodel engine; Megatron
  DSv4.1 support is not public; layer-cut model, it cannot fit 40 layers on 8 x H200), ours
  DSv4.1 (40 layers, `experiments.sh D41`), then native DSv4 (BL2) and ours DSv4 (E3), 100
  steps each, measuring the mismatch count and the runtime. verl PR 8403 (Megatron Lite,
  bitwise) needs SM100 (MegaMoE, UMMA backward): not runnable on H200.

## This cluster (pinghsieh-rsv1)

The design doc's environment section is generic. On this account:

- **Slurm:** partition `h200q-rsv` (3 nodes x 8 H200).
  - GPUs as `--gres=gpu:gpu:N` (plain `gpu:N` is rejected).
  - Account `at0816` is shared with other users and has a group limit of **2 GPUs, 48 CPUs
    and 500 GB of memory across all its running jobs** (`sacctmgr show assoc`). A 2-GPU
    job waits while anything else on the account holds a GPU.
  - 8-GPU work cannot run here.
  - `sinfo` is not allowed; use `squeue` / `scontrol`.
- **Conda envs** under `~/.conda/envs/`:
  - `dsv4` (torch 2.9 / CUDA 12.8, old);
  - `dsv4-t213` (torch 2.13 / CUDA 13, kernel tests and the vLLM gate);
  - `dsv4-verl`: a clone of `dsv4-t213` plus verl (`~/dsv4-work/ref/verl` at `4b39670`,
    editable), the vLLM fork (`~/dsv4-work/vllm-src` at `1a6542c`, editable), Megatron and
    TE, and the DeepGEMM paged-MQA patch (`~/dsv4-work/patches/`).
- **CUDA_HOME facades** (no CUDA modules on this cluster): `~/dsv4-work/cuda130-home` for
  `dsv4-t213`, `~/dsv4-work/cuda130-verl-home` for `dsv4-verl`.
- **Job scripts** in `~/dsv4-work/jobs/`; logs go to `~/dsv4-work/logs/`.
  - `gpu_job_t213.sh '<cmd>'`: rsyncs the checkout into `~/dsv4-work/wt213`, builds `_C`,
    then runs the command. Pass the build switches as
    `BUILD_FLAGS="KERNEL_ALIGN_MOE_SM90=1"`, or `SKIP_BUILD=1` to skip the build.
  - `gpu_job_verl.sh '<cmd>'`: the `dsv4-verl` env, with `wt213` on `PYTHONPATH`. Rsync
    the checkout into `wt213` first.
  - Toy GRPO: `sbatch --gres=gpu:gpu:2 --mem=400G gpu_job_verl.sh
    "MODEL_PATH=$WORK/dsv4-toy256 DATA=$WORK/data/gsm8k bash
    $WORK/wt213/scripts/verl_e2e_lora/run_dsv4_rlkernel_grpo.sh"`.
- **Checkpoints:**
  - `~/dsv4-work/dsv4-toy256`: the toy checkpoint, real dimensions, 4 layers, 256 experts,
    random weights.
  - `~/dsv4-work/dsv4-meta`: the real checkpoint's config, headers and tokenizer, with no
    weights.
  - `~/dsv4-work/ref/dsv4-inference`: DeepSeek's reference code.
  - The router kernel needs E = 256 or 384, so the old 16-expert `dsv4-toy` cannot run.
- Triton-backed `det_gemm` tests skip in jobs ("Triton is unavailable"); this predates the
  DSv4 work.

## Numerics lessons

See the design doc, section 13.
