# Reproduce the native four-layer actor/rollout cycle

This opt-in harness runs the actual verl TrainingWorker, native NVIDIA MCore
HybridModel and AMD vLLM rollout worker. It retains the original layers 0–3,
BF16 embedding/head/norm/mHC, FP8 dense tensors/scales and E2M1/E8M0 experts.
It does not train the frozen quantized bases. Nonquantized trainable Parameters
retain their identities and receive a real optimizer update. No upstream source
or Bridge Python changes are needed; `audit_dependencies.py` checks the three
pinned upstream HEADs and all 878 Bridge Python wheel RECORD entries.

The harness requires the companion native integration/operator Draft changes:
#684, #685, #688, #690–#696, plus the existing ROCm norm, deterministic reduction
and shared-down fixes. The versioned output boundary is separate from mHC and
keeps the final head → final norm → original lm_head → raw logprob order.
The native Megatron-Lite integration remains available.

The initial MI300X eager TP=PP=EP=CP=1 cycle compares both original response
tokens exactly in FP32, then independently replays them, performs actual
TrainingWorker backward/optimizer.step and synchronizes the same live rollout
PID. The final norm digest changes and agrees after synchronization. Both
updated fixed response scores also match byte-for-byte. This initial probe is
not sufficient for the expanded acceptance gate.

The extended harness saves multiple generated prompts with response lengths
1, 2 and 3; fixed-sequence comparisons at batch 1, AB, BA and repeated AB; and
separate within-engine repeat/position comparisons. A/B have equal complete
length 131 to avoid padding changes. The short case has complete length 10.
Actual rollout attention metadata records the request count and complete input
rows; the harness rejects a requested batch that did not really execute in the
specified order. It preserves all valid response tokens, checks finite FP32,
and compares the original 32-bit words. Signed zero differs, zero-token and
nonfinite comparisons are rejected. Generation uses raw full-vocabulary scores,
not transformed sampling scores. Updated fixed old sequences and newly generated
updated sequences are both checked. Independent container repeat remains a
separate run with separate output directory.

The current inputs exercise ordinary sliding-window attention in layers 0/1,
C4 compression/indexer in layer 2, C128 in layer 3, hash routing in layers 0–2
and learned routing in layer 3. They do not exceed the indexer TopK budget;
this is not coverage of long-context indexer selection, distributed training,
graph capture, full-model parity or training packed base weights.

Run from the task's assembled image, without host framework/source mounts:

```bash
bash examples/operators/dsv4_4l/run_cycle.sh \
  "$IMAGE" "$PHYSICAL_GPU" "$FOUR_LAYER_CHECKPOINT" \
  "$NEW_OUTPUT_DIRECTORY" "$DATA_DISK_CACHE_DIRECTORY"
```

The launcher rejects an occupied GPU, requires a new output directory, records
the image inspect result, mounts the model read-only and leaves all previous
outputs intact. The image must already include the pinned dependencies and
assembled RLK changes. Use another new output directory for the independent
repeat. `results.json` is authoritative per configuration; a passing operator
regression alone does not count as actor/rollout acceptance. Original GPU
captures and checkpoint weights stay on the data disk and are not committed.

`decode0-shared-down-paired.json` deliberately reports a separate rollout
self-repeat test: actual identical inputs trigger differences in the original
shared-down kernel, while the fixed production kernel repeats exactly. It is
not an actor-versus-rollout score result.
