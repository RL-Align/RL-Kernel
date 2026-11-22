# ROCm companion sources

The ROCm integration requires these VIME, Megatron and vLLM changes alongside
RL-Kernel. They include the inherited ROCm environment adaptations and complete
top-p support transport used in the MI300X validation. These are source patches,
not claims that the corresponding upstream projects have merged the changes.

`sources.json` records each base, tested companion commit, final tree and patch
SHA256. Each patch was applied to an isolated Git index at its base and verified
to reproduce the final committed tree exactly. The recorded commits are source
identifiers, not a guarantee that they are available from upstream. In
particular, the vLLM base is a local build commit; use the
[public-upstream reconstruction below](#reconstruct-the-vllm-base-from-public-upstream)
for a fresh vLLM checkout.

If a separate clean checkout already contains its recorded base, apply the
corresponding companion patch:

```bash
git switch -c fix/rocm-canonical-sampling <base-from-sources.json>
git apply --check /absolute/path/to/<repository>.patch
git apply --index /absolute/path/to/<repository>.patch
```

Use all three patched companion checkouts with this RL-Kernel branch. Configure
their paths in the
[MI300X profile](../../../../../../configs/integrations/orchestrators/vime/rocm_attention/mi300x-qwen3-8b.json)
before running `rlk-repro`.
ROCm libraries, compiled extensions, model weights and datasets are not bundled.

## Reconstruct the vLLM base from public upstream

The historical vLLM base `11d76fc7259b3c1d70e2043b7513226aec9a665a` is a local
ROCm build commit with committer `docker-build <build@localhost>`, not an upstream
release tag. Its checkout used `vllm-project/vllm` as `origin`, but the commit
itself is not available there. Its direct parent is the public upstream commit
[`b22afe45ac797ae58e67a7a3ad79ee5714024420`](https://github.com/vllm-project/vllm/commit/b22afe45ac797ae58e67a7a3ad79ee5714024420).
The original local commit was recovered and its diff is bundled as
[`vllm-base.patch`](vllm-base.patch), so reconstruction does not depend on access
to the original machine or a private fork:

```text
b22afe45ac797ae58e67a7a3ad79ee5714024420  public upstream
  + vllm-base.patch
11d76fc7259b3c1d70e2043b7513226aec9a665a  historical local base
  + vllm.patch
1d813359776761b7631e192e2240e4012879a559  tested companion commit
```

The base patch is `[ROCm] Grant host access to VMM allocations in sleep-mode
pool`. It changes the allocator, its ROCm compatibility definitions and a
regression test to grant host access to VMM allocations. It is **not a KV-cache
tensor-layout or stride fix**. The existing `vllm.patch` remains the separate
companion patch for complete nucleus support and ROCm IPC compatibility.

Use a new checkout, rather than applying these steps to an existing vLLM
installation. Replace both absolute paths below. The two `git write-tree`
checks verify the staged source trees without needing the original local
commit objects:

```bash
set -euo pipefail
RLK_ROOT=/absolute/path/to/RL-Kernel
VLLM_ROOT=/absolute/path/to/vllm-rocm-repro
PATCH_DIR="$RLK_ROOT/rl_engine/integrations/orchestrators/vime/patches/rocm_attention"
UPSTREAM_BASE=b22afe45ac797ae58e67a7a3ad79ee5714024420

git clone --filter=blob:none --no-checkout \
  https://github.com/vllm-project/vllm.git "$VLLM_ROOT"
cd "$VLLM_ROOT"
git fetch origin "$UPSTREAM_BASE"
git switch -c fix/rocm-canonical-sampling "$UPSTREAM_BASE"

# First reconstruct the historical local base's source tree.
git apply --check "$PATCH_DIR/vllm-base.patch"
git apply --index "$PATCH_DIR/vllm-base.patch"
test "$(git write-tree)" = bea570245d926776ac49d2749f64881e7d209698

# Then apply the existing ROCm companion patch exactly once.
git apply --check "$PATCH_DIR/vllm.patch"
git apply --index "$PATCH_DIR/vllm.patch"
test "$(git write-tree)" = c0bd64859e1ee1e1268732fef0f0541da5c80f13
```

These commands leave both patches staged and `HEAD` at the public upstream
parent. They reproduce the historical **source trees**, not the original
commit SHAs. A new local commit may have a different SHA because its metadata
differs. Do not use `git rev-parse HEAD` as the tree-equality check, or reapply
`vllm.patch` after completing these steps. Keep the patched sources when
building/installing vLLM and point the profile's `vllm_root` at this checkout.
The base patch changes C++ code, so changing only the Python import path does
not update an already-built vLLM extension.

The vLLM entry in `sources.json` preserves the original `base`, `commit`,
`tree` and companion `patch_sha256`. Its additional `upstream_repository`,
`upstream_base`, `base_patch`, `base_patch_sha256` and `base_tree` fields record
how to reconstruct that base from public sources.

### Other vLLM versions and MI355

Applying `vllm.patch` successfully to a release such as `v0.27.1` only checks
patch-context compatibility; it does not establish source-tree equivalence or
train/rollout logprob consistency. If a different version needs a KV-cache
layout fix, record that fix and validate the exact resulting configuration with
rollout-logprob reuse disabled. A passing run is evidence for that tested
configuration, not blanket support for the release or other configurations.

The historical GPU evidence below covers MI300X/gfx942, not MI355. The source
reconstruction checks do not add MI355 runtime validation. Use the exact
reconstructed trees for a source-level reproduction of the documented stack;
validate a port to another GPU or dependency stack separately.

## Companion behavior and validation scope

The current VIME patch packs complete retained top-p supports into compact
token-ID rows and selects `processed_logits` for the strict sparse route.
RL-Kernel supplies the HIP scorer and fused monitoring entropy; rebuild its
extension after updating. The vLLM patch still transports complete support,
including nuclei larger than 64 or 128 tokens. Use the
[ROCm performance command](../../../../../../docs/usage/rocm-sparse-performance.md)
with rollout-logprob reuse disabled. The historical 4.44% throughput difference
is not a measurement of this updated command.

Previous companion validation: 8 MI300X/gfx942 GPUs, PyTorch 2.12.0+rocm7.14.0a20260608, HIP
7.14.60850. Nine complete one-round cases passed, each with eight samples and
maximum response length 7168, with rollout-logprob reuse disabled and reference
KL coefficient 0.001. In total 292,249 selected train/rollout logprobs matched
byte for byte. This covers training TP/CP=(1,8),(2,4),(4,2),(8,1) against rollout
TP4; training TP4/CP2 against rollout TP1/2/8; and temperature/top-p pairs
0.7/0.95, 1/1, 1.3/0.8. It is not an exhaustive Cartesian matrix or a claim of
identical gradients or trajectories between topologies.

Targeted RL-Kernel pytest: 90 passed. Companion vLLM pre-commit checks and
Python 3.10/3.12 type checks passed. The one-round runs include cold execution
effects and do not establish steady-state performance overhead. Runs were made
during development with frozen before/after source fingerprints, then the final
sources were committed; earlier topology passes preceded the final TP1 fix.

AI assistance was used to prepare these integration changes and validation.
