# ROCm companion sources

The ROCm integration requires these VIME, Megatron and vLLM changes alongside
RL-Kernel. They include the inherited ROCm environment adaptations and complete
top-p support transport used in the MI300X validation. These are source patches,
not claims that the corresponding upstream projects have merged the changes.

`sources.json` records each base, tested companion commit, final tree and patch
SHA256. Each patch was applied to an isolated Git index at its base and verified
to reproduce the final committed tree exactly. Use a separate clean checkout:

```bash
git switch -c fix/rocm-canonical-sampling <base-from-sources.json>
git apply --check /absolute/path/to/<repository>.patch
git apply --index /absolute/path/to/<repository>.patch
```

Use all three patched companion checkouts with this RL-Kernel branch. Configure
their paths in `../profiles/mi300x-qwen3-8b.json` before running `rlk-repro`.
ROCm libraries, compiled extensions, model weights and datasets are not bundled.

Validation: 8 MI300X/gfx942 GPUs, PyTorch 2.12.0+rocm7.14.0a20260608, HIP
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
