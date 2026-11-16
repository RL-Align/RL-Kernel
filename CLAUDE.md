# CLAUDE.md — DSv4 CUDA work on `test-dsv4-cuda` (PR #635)

Notes for an assistant (or developer) continuing the DSv4 SM90 batch-invariant work. The
design of record is `docs/architecture/dsv4-design.md`: goal, environment, how to build
and test, module map, decisions, the reduction order of every module, status and open
work. Read it first and keep its status section current. Last updated 2026-11-16.

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
  Everything lives in `rl_engine/integrations/dsv4/` as plugins.
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
  the actor and the rollout. With a data-parallel-only engine this caps the layer count per
  GPU (4 layers on H200).
- The attention backward is not deterministic, so grad norms differ slightly run to run.

## Status (2026-11-16) and next

- Done:
  - The kernel set (forward BI, LoRA training end to end; EP = 2 bitwise to EP = 1).
  - The vLLM rollout plugin, bitwise to `Dsv4Model` with LoRA (W1, W2).
  - The `rlkernel` verl engine (W3) and adapter sync (W4).
  - Toy GRPO on 2 x H200 (W5): mismatch 0 at every step; the negative control is nonzero
    from step 2.
- Ready, not run: `scripts/verl_e2e_lora/` (W6), the 8-GPU guide against verl #7483 on the
  real checkpoint (first `LAYERS` layers). It needs a machine with 8 GPUs per job.
- Open, roughly in order:
  - the BI-off switch for the ablation: the plugin refuses `VLLM_BATCH_INVARIANT=0` and
    the paged FP8 decode today;
  - `rlkernel` engine at EP > 1 and the rollout plugin at EP > 1, so the full 43-layer
    model can run;
  - CUDA graphs in the rollout (eager today);
  - the layerwise gate on the real checkpoint;
  - the older `scripts/verl_e2e/README.md` guide (verl #7688 baseline, full-parameter).

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
    $WORK/wt213/scripts/verl_e2e/run_dsv4_rlkernel_grpo.sh"`.
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
