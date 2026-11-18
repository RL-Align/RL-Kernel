# DSV4 ROCm: AMD reproduction and multi-node handoff

## Status at the PR #711 handoff

**Measured evidence remains single-node, four-layer GRPO.** This update adds an
experimental multi-node PP2 implementation and executable launch commands. It has
CPU contract coverage only; AMD multi-node GPU acceptance is still pending.

| Scope | Status |
| --- | --- |
| Single-node, four original layers | Existing measured jobs; historical failure and two GPU setup errors remain open |
| Actor and rollout on separate hosts, each local PP2 | Implemented via explicit Ray placement and checked shared-file policy synchronization; GPU validation pending |
| Actor and rollout PP2 each spanning two hosts | Implemented via native Megatron PP and the pinned vLLM multi-node `mp`/headless path; GPU validation pending |
| Full model, EP/DP/TP/CP expansion | Not enabled by this change; four layers, PP2/DP1, TP1/EP1/CP1 remain mandatory |

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
`8196af485ae50e09aa42c1fc6e399f266fb166d3`. For multi-node runs, use the companion `dsv4-rocm-multinode` verl PR branch
instead of this historical pin, and record its exact SHA. Both repositories must
include the multi-node changes; the historical pin still rejects those layouts.

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

## 4. Run the experimental multi-node four-layer GRPO path

Use RLK #711 together with the community verl `dsv4-rocm-multinode` PR. Build
both exact commits into the **same image on every node** using the measured
dependency environment. Record the new image digest; it is not the previously
measured image. Do not merely update a host checkout while running an old image.
The new modules must be importable in the image's Python environment.

This recipe still uses four GPUs total. On two nodes, allocate two GPUs per node.
Supported placements include:

| Case | Actor | Rollout | What crosses nodes |
| --- | --- | --- | --- |
| Separate engines | Node A: two PP ranks | Node B: two PP ranks | Control flow and shared-file policy sync |
| Cross-node PP | One actor PP rank on A and B | One rollout PP rank on A and B | Actor forward/backward PP, rollout PP, control and policy sync |

The two cases use the same GPU allocation but must run separately. Four-layer
ownership stays `0–2 / 3`; no extra DP replicas are created. Do not substitute
PP4/EP4/full-model settings into these commands.

### Start the allocated Ray cluster

Run in identical runtime containers, using the image's Python/Ray executable.
Expose only allocated GPU devices. Preserve each node's own local visibility;
never copy GPU ordinals from one host to the other. The examples assume the image
is installed at `/opt/dsv4` and a shared mount is visible at `/shared/dsv4`.
Use host networking or an equivalent cluster-approved network setup with Ray,
RCCL and vLLM rendezvous ports reachable. Mount required RDMA devices if applicable.
`NODE_A` and `NODE_B` must be actual Ray node IP addresses, not hostnames.

On node A:

```bash
export PATH=/opt/dsv4-venv/bin:$PATH
export NODE_A=10.0.0.10  # replace with the actual address
export VLLM_HOST_IP="$NODE_A"
export NCCL_DEBUG=INFO
ray start --head --node-ip-address="$NODE_A" --port=6379 \
  --num-gpus=2 --num-cpus=16 --disable-usage-stats
```

On node B:

```bash
export PATH=/opt/dsv4-venv/bin:$PATH
export NODE_A=10.0.0.10
export NODE_B=10.0.0.11
export VLLM_HOST_IP="$NODE_B"
export NCCL_DEBUG=INFO
ray start --address="$NODE_A:6379" --node-ip-address="$NODE_B" \
  --num-gpus=2 --num-cpus=16 --disable-usage-stats
```

These commands start a dedicated allocated cluster; do not stop or replace a
shared Ray service. Apply the actual NIC/RDMA settings on **both nodes before
starting Ray**. On the head, `ray status` must show both allocated nodes and four
GPUs. The launcher attaches to this cluster; it never creates a local replacement.

### Submit one job from node A

All paths below must exist at the same absolute location on every node. The
output mount must be a real shared filesystem with coherent reads and atomic
rename. Preflight writes a unique challenge, checks peer writes, hashes checkpoint
weights/tokenizer/data, and compares source/dependency identity on every node.
Node binding has a bounded placement timeout. No model workers are allocated
until preflight succeeds.

```bash
cd /opt/dsv4/RL-Kernel
export NODE_A=10.0.0.10
export NODE_B=10.0.0.11
export GRPO_CHECKPOINT=/shared/dsv4/models/DeepSeek-V4-Flash-4l
export GRPO_DATA=/shared/dsv4/final-data
export GRPO_OUTPUT=/shared/dsv4/runs/cross-node-pp-001  # must not exist

python examples/operators/dsv4_4l/run_cluster_grpo.py \
  --ray-address "$NODE_A:6379" \
  --actor-nodes "$NODE_A,$NODE_B" \
  --rollout-nodes "$NODE_A,$NODE_B" \
  --checkpoint "$GRPO_CHECKPOINT" --data "$GRPO_DATA" \
  --output "$GRPO_OUTPUT" --stress-final-repeats 1000

python examples/operators/dsv4_4l/verify_grpo.py \
  "$GRPO_OUTPUT/policy-sync" --inventory "$GRPO_DATA/checkpoint-inventory.json"
```

Add `--dry-run` to inspect the exact Hydra command without submitting a job.
For the separate-engines case, use `--actor-nodes "$NODE_A"` and
`--rollout-nodes "$NODE_B"` with a **new** output directory. Both cases keep
three actual GRPO/PPO updates and audit v0–v3 on the same live workers.

The launcher writes `launch.json`, `driver.log`, the process exit code and wall
time. `policy-sync/cluster-preflight.json` records node/source/input checks;
`cluster-placement.json` records actual node/rank/PCI ownership. Placement is
checked before policy publication: every declared rank must be present and no
actor/rollout GPU can overlap. Worker identity includes node, PID and process
instance, so equal PIDs on different hosts are valid and restarted workers fail.

Weight synchronization uses the existing complete actor export and native
stage-owned rollout load/readback over the checked shared mount. It retains
atomic rename, version ordering, admission pause and full tensor acknowledgments.
There is no new RDMA weight transport. Node failure or missing shared files must
fail the job, not fall back to stale weights.

Collect each node's Ray worker logs in addition to driver and policy evidence.
The cluster daemons remain running after the job; a daemon being alive is not a
training pass. The verifier checks driver success, actual worker placement,
complete score matrices, state readback and unchanged live-worker identity.

## 5. Acceptance and remaining scope

Run the separate-engines case, then cross-node PP. Require original finite FP32
logprob bit equality, zero mismatches and `expected_tokens == compared_tokens > 0`
for fixed/fresh B1/B2, swapped orders and repeats at v0–v3. Preserve failures,
updated policies and traces. Report CPU checks and GPU results separately.

Full-model execution remains blocked by the explicit four-layer recipe and
stage partition. A follow-up must generalize native model binding, checkpoint
ownership and the acceptance matrix before changing the model path. EP/DP/TP/CP
expansion also needs its own numerical and synchronization tests. Neither a
successful communication smoke nor this code submission is multi-node GPU evidence.

## Result handoff

For every case, return source/image pins, resolved configuration, model/data hashes,
actual node/GPU/rank map, driver and per-rank logs, exit codes, raw score comparisons,
policy manifests/readbacks and the verifier output. State PASS, FAIL or BLOCKED
with the exact tested scope. Preserve the historical failures in the baseline report.
