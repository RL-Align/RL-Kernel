# DSV4 ROCm: AMD reproduction and multi-node handoff

## Status at the PR #711 handoff

**The measured baseline is single-node, four-layer GRPO. Multi-node model
execution and full-model acceptance are not implemented by this handoff.**
This document adds preparation commands, a communication smoke test and the
implementation/acceptance gates needed before AMD can run those extensions.
The smoke test has not been executed on AMD hardware in this documentation update.

| Scope | Evidence / status |
| --- | --- |
| Single-node, four original checkpoint layers | Completed jobs show actor/rollout raw FP32 parity through three real GRPO updates; see the report below |
| Overall four-layer acceptance | Still pending: one unexplained historical v3 failure and two final-image GPU regression setup errors |
| Multi-node communication smoke | New preparation command below; hardware validation pending |
| Actor and rollout on separate nodes | Requires placement, shared policy storage and evidence changes; not yet a supported recipe |
| Actor or rollout model parallelism across nodes | Requires the multi-node integration changes below; not enabled by changing `nnodes` alone |
| Full model, EP/TP/CP expansion, performance | Separate, unvalidated extensions; no pass or speedup claim |

The [measured report](../../examples/operators/dsv4_4l/results/native_grpo_pp2_dp1_20261117/README.md)
is the authority for results and remaining failures. It used one node and four
MI300X GPUs: actor physical GPUs 4–5 and rollout 6–7, each PP2/DP1 with
TP1/EP1/CP1. Original layer ownership is 0–2 / 3; execution is eager,
`rocm_triton/split`. Packed FP4/FP8 weights and scales stay frozen. This recipe
updates selected nonquantized parameters; it is **not a LoRA training result**.
Do not generalize it to MI355X, long-context training or arbitrary parallel layouts.

## 1. Freeze code, image and input artifacts

After #711 is merged, use `dsv4-rocm` and record the resulting commit. Before
merge, the review branch is `vensen/dsv4-4l-pp2-dp2-20261116`.

```bash
git clone --branch dsv4-rocm https://github.com/RL-Align/RL-Kernel.git
git -C RL-Kernel rev-parse HEAD
git clone https://github.com/RL-Align/verl.git
git -C verl checkout --detach 784346ad07a55b83e6cf77cb937fdadd1c731b45
git -C verl rev-parse HEAD
```

The community [verl PR #1](https://github.com/RL-Align/verl/pull/1) is merged.
The published pin above has the same source tree as the measured
`8196af485ae50e09aa42c1fc6e399f266fb166d3`. Future multi-node integration needs
a new companion verl commit; this pin intentionally still rejects unsupported layouts.

The measured RLK execution pin is `478caf1a787f6c9d9d3747a45ad79756b6c2c616`.
Keep measured results associated with that pin, even when testing a newer branch.
Record actual imported source paths/SHAs inside every container, not just host checkouts.
Use [sources.lock](../../examples/operators/dsv4_4l/results/native_grpo_pp2_dp1_20261117/sources.lock)
and [image.json](../../examples/operators/dsv4_4l/results/native_grpo_pp2_dp1_20261117/image.json)
for the measured dependency versions and image identity.

The tag `rlk/dsv4-4l-rocm:grpo-stress` is a local measured image, **not a published
registry download**. Before scheduling reproduction, obtain from the baseline owner:

- The image archive or an accessible registry digest; verify its image ID against
  the report. The committed Dockerfile inherits a local dependency image and is
  not a clean-room build recipe by itself.
- The four-layer checkpoint, tokenizer and checkpoint inventory; validate the
  checkpoint hashes against the recorded inventory.
- The `final-data` artifacts: train/validation parquet, fixed replay sequences,
  extended audit prompts and inventory. Their checksums are in the report's data manifest.
- Adequate disk space for retained updated policies, raw comparisons and traces.

Mount the same model/data paths on every participating node. For a future
file-based multi-node policy sync, all processes must see the **same filesystem**
at the sync path; identical directory names on different local disks are insufficient.
Record GPUs, hostnames, NIC/RDMA devices, container digest and allocated device IDs.

## 2. Reproduce the measured single-node baseline

On the original machine, use the recorded
[reproduce.sh](../../examples/operators/dsv4_4l/results/native_grpo_pp2_dp1_20261117/reproduce.sh).
It contains machine-specific paths and exact image/device guards.

On another MI300X node with the transferred artifacts, the equivalent launcher is:

```bash
python3 examples/operators/dsv4_4l/run_grpo.py \
  "$GRPO_IMAGE" "$GRPO_CHECKPOINT" "$GRPO_DATA" "$GRPO_OUTPUT" "$GRPO_CACHE" \
  --audit-response-length 16 \
  --audit-prompts-file "$GRPO_DATA/extended-audit-prompts.json" \
  --stress-final-repeats 1000
```

Set the five variables to the image and absolute artifact directories; output must
be a new directory. This launcher **still requires available physical GPUs 4–7**,
Docker, ROCm device isolation and the recorded image layout. It starts a local Ray
runtime. Do not use it on a four-GPU host or repurpose it as a multi-node launcher.
Porting device selection is a separate launcher change.

Then verify the retained raw evidence using the same runtime image:

```bash
docker run --rm --network=none --cpus=4 -e OMP_NUM_THREADS=4 \
  -v "$GRPO_OUTPUT:/outputs" -v "$GRPO_DATA:/data:ro" \
  "$GRPO_IMAGE" /opt/dsv4-venv/bin/python \
  /opt/dsv4/RL-Kernel/examples/operators/dsv4_4l/verify_grpo.py \
  /outputs/policy-sync --inventory /data/checkpoint-inventory.json
```

Retain driver exit records, verifier output and raw policy-sync artifacts. A run
without a successful process exit or complete evidence is not a completed pass.
The verifier is currently specific to four layers, two PP stages and v0–v3.

## 3. AMD can run this now: multi-node communication preparation

Use identical ROCm/PyTorch environments on allocated nodes, with reachable node IPs,
the required container network/RDMA access and no conflicting GPU jobs. Run the
following **inside the runtime environment on every node**. This uses torchrun,
not Ray, and does not start training or rollout.

```bash
# Example: two nodes, two allocated GPUs per node. Adjust to the actual allocation.
export NNODES=2
export GPUS_PER_NODE=2
export NODE_RANK=0                 # set 1 on the second node
export MASTER_ADDR=10.0.0.10       # replace with the reachable first-node IP
export MASTER_PORT=29571           # same unused port on every node
export RLK_NODE_ID="$(hostname)"  # use a unique physical-node ID per node
export COMM_OUTPUT=/outputs/dsv4-comm-001  # fresh run directory, local or shared
export NCCL_DEBUG=INFO
# If needed, set NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME to the actual cluster NIC.
# Configure RDMA according to the cluster; do not copy interface names from another host.
mkdir -p "$COMM_OUTPUT"
set -o pipefail
python -m torch.distributed.run \
  --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  examples/operators/dsv4_4l/multinode_comm_smoke.py \
  --expected-nodes="$NNODES" --output="$COMM_OUTPUT" \
  2>&1 | tee "$COMM_OUTPUT/node-$NODE_RANK.log"
```

PyTorch uses the `nccl` backend name for RCCL on ROCm. Each rank must exit zero
and produce `comm-rank-<rank>.json` with PASS. Collect logs from every node.
This tests integer all-reduce and forward/backward rank-chain P2P payloads.
It does **not** test EP all-to-all, bandwidth, floating-point reduction order,
Ray placement, weight synchronization or model logprob parity.

## 4. Required code changes before multi-node model execution

These are implementation gates, not settings AMD should bypass. Do not remove
validation checks merely to make a larger configuration start.

| Location | Current limitation | Required change and evidence |
| --- | --- | --- |
| RLK `examples/operators/dsv4_4l/run_grpo.py` | Physical GPUs 4–7, `address=local`, one container | Separate cluster bootstrap/job submission; explicit allocated GPUs and per-node environment; retain every node's logs and exits |
| verl `experimental/rl_kernel/trainer_contract.py` and `config/dsv4_grpo.yaml` | Actor and rollout each fixed to one node/two GPUs | Add explicit supported topologies, resource accounting and negative config tests; preserve rejection of unimplemented layouts |
| RLK `rl_engine/integrations/orchestrators/verl/pipeline_rollout.py` | Requires one node, two local GPUs and vLLM `mp` | For a replica spanning nodes, implement a distributed executor/placement path with correct PP worker ownership; local `mp` cannot span hosts |
| verl `experimental/rl_kernel/trainer_rollout.py` | Two reservations and hard-coded original layer partition `3,1` | Configure and verify node/rank placement and layer ownership; do not carry the four-layer partition into a full model |
| verl `experimental/rl_kernel/trainer_sync.py`, actor export and rollout readback | Exactly two stage exports; policy files read by path | Validate shared storage visibility/atomic publication or implement explicit transfer; collect complete shard acknowledgments and preserve stale/corrupt failure behavior |
| verl acceptance plus RLK `verify_grpo.py` / evidence writers | Four-layer PP2 identities, file names and stage counts | Generalize identities to node/rank/replica, prevent filename collisions, verify complete expected token/shard coverage and all process exits |

Keep checkpoint layout and native Megatron layer mapping aligned when expanding
to the full model. EP/DP expansion additionally needs routing/gradient/sync
ownership checks; this PR's native GRPO evidence has EP1/DP1 only.

## 5. Multi-node end-to-end execution order after those gates land

1. Publish the new RLK and companion verl SHAs, image digest, launch script and
   resolved configuration. Specify actor and rollout node/GPU ownership separately.
   Joining Ray nodes alone is not proof that model ranks span hosts.
2. Start with four layers and TP1/EP1/CP1. First validate actor/rollout placement
   on separate hosts, then explicitly test model PP across hosts. Record actual
   host/rank/device placement for both cases; they exercise different communication.
3. Run the real GRPO path: generation → reward → advantages → native PPO
   backward/optimizer → policy export/load/readback → next rollout. Audit v0 and
   each of three updates on the same live workers, with fixed/fresh B1/B2 cases,
   request-order swaps and repeats. Retain failed policies and first-difference traces.
4. Require original finite FP32 token-score bit equality, zero mismatches and
   `expected_tokens == compared_tokens > 0`, matching tokens/positions/masks,
   current policy versions and complete shard readback. Record all node exits.
5. Only after multi-node four-layer acceptance, switch to the full checkpoint
   with a declared layer partition and feasible GPU/memory plan. Expand EP/DP and
   sequence lengths in named cases; TP/CP remain 1 unless explicitly in scope.
   Do not treat a four-layer result as coverage of all compression branches.
6. Collect performance separately after correctness passes. Report rollout,
   update and synchronization time, token counts, total GPU count and peak memory;
   instrumented correctness timing is not a speedup result.

There is deliberately **no multi-node GRPO/full-model launch command yet**:
the current pinned integration rejects it. The follow-up integration must supply
that executable command before asking AMD to run steps 2–6. AMD can meanwhile
prepare artifacts, reproduce the supported baseline and run the communication smoke.

## Result handoff

For every case, return source/image pins, resolved configuration, model/data hashes,
actual node/GPU/rank map, driver and per-rank logs, exit codes, raw score comparisons,
policy manifests/readbacks and the verifier output. State PASS, FAIL or BLOCKED
with the exact tested scope. Preserve the historical failures in the baseline report.
