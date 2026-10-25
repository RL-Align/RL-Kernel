# F2 final local review and submission preparation

Suggested title: `feat(rope): add Muse Glimmer F2 CUDA/Triton norm-scale-RoPE`
Synced target: `origin/test-museglimmer` at `16b8117`. Working branch:
`feat/museglimmer-f2`. No conflicts; tracked F2 changes and all untracked files retained.
Post-sync local tests: 32 passed, 87 GPU tests skipped. Recorded F2 source hashes
match H100 validation; no post-sync GPU rerun. Details: `sync.json`.
Draft body: `pr-body.md`. No commit, push, PR creation or external comment was made.

Reviewed the public dispatch/autograd, table lifecycle, CUDA/Triton arithmetic,
build integration and current regression/benchmark evidence. No additional blocking
correctness defect was identified in this pass; this is not exhaustive proof.

- P1 r-cubed underflow: fixed by VJP v2, independent FP64 counterexamples included.
- P2 table identity: public entry enforces registered exact pairs and version checks;
  loaders verify against trusted expected metadata/content. Raw native/private
  backend entry points are internal and do not claim standalone provenance checks.
- Native baseline gap: now measured against the pinned native eager function bodies,
  with generation excluded on both paths and public-call overhead included. There
  is no full-model or optimized-compiler baseline claim.
- Build scope: dropping forced C++17 affects the extension build generally; justified
  by installed PyTorch requiring C++20. A100 full extension build passed. Older
  toolchains/platforms are not newly qualified by that result.

A100–H100 VJP v2 parity is now validated for the archived inputs: 24 cases /
144 tensor comparisons, zero bit differences; H100 complete targeted suite 121 passed.
See `../f2-h100-2026-10-24/README.md`. H100 used an isolated F2 build.

Remaining qualification: broader cross-architecture/input coverage, periodic ROCm validation,
upstream agreement on strict arithmetic/thresholds, and runtime integration distributing
trusted table identities. Do not describe those as completed in the PR or résumé.
A reviewable draft can be prepared with these limits visible.

`submission-files.txt` is a proposed allowlist, not staged files. It includes source,
tests, benchmark and compact historical/current evidence so existing contract links
remain meaningful. Large tensor dumps, profiler traces, compiled binaries and tarballs
remain local and are excluded. Some historical reports describe artifacts that are
intentionally not included in the PR; they are retained locally. Review the allowlist
before staging rather than using `git add .`.

Current A100 work is complete for this pass: full regression and native eager baseline
are saved locally. The instance has not been stopped or destroyed.
