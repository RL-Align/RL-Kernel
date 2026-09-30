# CUDA and ROCm consistency audit — 2026-09-19

Both platforms have evidence for **within-run training/rollout selected logprob
byte equality** with independently chosen training and rollout TP and configurable
temperature/top-p. This does not establish arbitrary topology support, an exhaustive
sampling matrix, identical optimizer trajectories across topologies, or byte equality
between CUDA and ROCm hardware. No training or GPU test was rerun for this audit.
The integrated source has CPU regression checks; the GPU evidence below predates
the merge and must not be presented as a GPU validation of the merged revision.

Subsequent GPU work is recorded separately in the
[H100 CP backward follow-up](h100-cp-gradient-validation.md): the original
TP2/CP4 versus TP4/CP2 pair now has zero observed update/norm differences,
and the [subsequent rollout CP2 implementation](h100-pcp-validation.md) now
passes two updates with zero token/norm/parameter differences against CP1.
The later [H100 topology matrix](h100-matrix-validation.md) records full-model
CP2/4/8 coverage, the CP8 gradient fix and measured costs.

## Common command

Configure `.rlk-profile.json` once in each checkout, including its runtime Python,
framework paths, checkpoints and backend (`requirements.backend`). CUDA and ROCm
then share this command:

```bash
./rlk run --tp 2 --rollout-tp 4 --temperature 0.7 --top-p 0.95 --steps 200
```

Use `./rlk plan` with the same arguments to inspect the expanded command. On the
single eight-GPU Qwen3-8B profile, omitted CP is `8 / TP`. Changing `--tp` alone
overrides any stale profile CP; explicit `--cp` still wins. Training TP can be
1/2/4/8, and rollout TP is independently 1/2/4/8. Training PP is 1; rollout CP
was 1 in this original audit; the later H100 matrix adds measured CP2/4/8
support. A syntactically valid configuration is not a memory or runtime guarantee.

The installed `rlk-repro`, checkout `bin/rlk-repro`, and module entry use the same
parser; `./rlk` additionally selects the profile's Python. Common options include
`--tp`, `--cp`, `--rollout-tp`, `--temperature`, `--top-p`, `--steps`, `--lr`,
`--weight-decay`, `--kl-coef`, `--max-response-len`, `--max-tokens-per-gpu`,
`--mode`, and `--run-id`. Long spellings remain compatibility aliases.
Both run commands wait for validation and disable rollout-logprob reuse.
ROCm now forwards optimizer/KL options into both the manifest and shell command;
they are no longer silently ignored. Bundled run defaults are 200 steps,
temperature 1, top-p 1, top-k -1, LR 5e-7, weight decay 0.1 and KL coefficient 0;
machine profiles may deliberately override them. Backend-specific memory budgets,
Ray setup and framework versions remain different.

| Capability | CUDA H100 | ROCm MI300X |
|---|---|---|
| Positive finite temperature; top-p in `(0,1]` | Configurable; complete support recomputed | Configurable; complete retained IDs replayed |
| Top-k | `-1` or positive; tested `128` | Only `-1`; filtering rejected |
| Greedy temperature 0 | Implemented; sampling-kernel evidence | Rejected |
| Mixed per-request temperatures | Rollout adapter handles row values; full-model tests use one value per run | Strict scoring uses the configured run temperature; heterogeneous requests not certified |
| Training TP/CP | `(1,8),(2,4),(4,2),(8,1)` | Same factorizations |
| Independent rollout TP | 1/2/4/8 | 1/2/4/8 |
| Rollout CP > 1 | See the subsequent H100 full-model topology matrix; shared-IPC PCP and CP2/4/8 attention checks are covered | Shared adapter is present, but ROCm PCP compatibility and GPU validation remain unverified |
| `verify` with real weight-update acceptance | Supported, two steps by default | Not implemented; explicit error |
| `run` and `plan` | Shared interface | Shared interface |
| `--detach`, `--allow-dirty`, standalone prepare/doctor/validate/report | CUDA path | Not shared; explicit errors for unsupported options/commands |

## Existing numerical evidence

CUDA: five two-step complete-model runs, 8,192 active tokens each, 40,960 total,
zero byte mismatches, real Adam updates and changed audited weight hashes. Training
TP/CP → rollout TP: `1/8 → 1`, `2/4 → 4`, `4/2 → 4`, `4/2 → 8`, `8/1 → 2`.
The first four use temperature/top-p/top-k `0.7/0.95/-1`; the last uses
`1.3/0.99/128`. Separate prior sampling tests cover 16 TP/parameter cases,
including greedy and a large nucleus. The two-step model tests capped responses
at 512 and all responses were truncated; their nonzero gradients came from KL,
not nonzero reward advantages. These are not new 200-step results.

ROCm: the original MI300X `single-arm-summary.json` and `validation.json` files
were read directly for this audit. Nine successful one-step cases, eight samples
each, 292,249 compared logprobs, zero byte mismatches. They cover all four training
TP/CP factorizations against rollout TP4, TP4/CP2 against rollout TP1/2/8, and
temperature/top-p `0.7/0.95`, `1/1`, `1.3/0.8`. Top-k was disabled, response cap
7168, reference KL coefficient 0.001. Earlier failed cases are retained in the
evidence JSON. Runs used evolving development revisions with frozen before/after
fingerprints, so passes do not all refer to one identical final source tree.

See [CUDA records](evidence/pr432-20260919/cuda-existing.json),
[ROCm records](evidence/pr432-20260919/rocm-existing.json), and the
[cross-configuration comparison](evidence/pr432-20260919/cross-configuration.json).
The selected logprobs are compared as uint8 bytes, including the distinction
between positive and negative zero. Neither route copies rollout logprob values
into the training score; ROCm replays support membership, then recomputes scores.

## Remaining gaps

1. **General cross-configuration backward/optimizer coverage remains incomplete.**
   On H100, TP2/CP4 and TP4/CP2 had the same initial audited weights, tokens,
   masks and rewards, but gradient norms differed by 0.021213% and the first
   updated audited weight hashes differed. The follow-up fixes and verifies
   this pair over two updates, comparing every exported parameter. The subsequent
   H100 topology matrix expands that coverage. It does not
   certify all topologies, microbatch packings or sampling parameters. ROCm has
   no corresponding optimizer trajectory proof for this change.
2. No exhaustive Cartesian topology × temperature × top-p matrix and no new
   200-step validation of the configurable implementation or merged revision.
   ROCm top-k, greedy and the update-verification contract remain unsupported.
3. No guarantee of equal bytes between CUDA and ROCm hardware. The existing
   checks compare training and rollout within each backend.
4. Performance is not uniformly close: H100 strict TP2/CP4 measured 30.726 s per
   step versus native 30.099 s (+2.1%), but strict TP1/CP8 → rollout TP1 took
   69.022 s versus strict TP4/CP2 → rollout TP4 at 29.960 s (+130.4%). ROCm's
   one-round measurements include cold-start overhead and do not establish
   steady-state overhead. No acceptance threshold has been selected.

## Companion source setup

CUDA requires the [CUDA VIME patch](../../examples/vime_qwen3_8b_tp4_cp2_200/companion_patches/README.md);
ROCm requires its own [VIME, Megatron and vLLM patches](../../examples/vime_rocm_attention_ablation/companion_patches/README.md).
They use different framework revisions. The CUDA patch is based on publicly
available VIME commit `c80200e7aef08edc918e50a3998ea981bd689934`, includes inherited
H100 integration changes, and was checked in an isolated Git index to reproduce
the tested VIME tree exactly. After applying with `git apply --index`, commit the
companion source changes locally before running clean-source validation. Do not
mix the two platform patch bundles. This is one-time environment setup, not a
per-run command requirement.
