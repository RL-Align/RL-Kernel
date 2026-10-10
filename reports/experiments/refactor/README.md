# Local refactor validation

Baseline: upstream `bea224b`; candidate: this refactor. See
`local-validation.json` for counts and failure identities.

The existing CPU-capable suite produced the same **1832 passed, 1920 skipped,
5 failed** before and after relocation. The five failures are pre-existing:
missing native `_C`, a strict ROCm capability expectation on CPU, missing
`get_loss_op`, a GRPO negative control on this PyTorch build, and a macOS Gloo
CLI timeout. Four modules requiring unavailable Triton and one module importing
an already-missing AITER contract symbol were excluded at collection in both
runs. This is a differential result, not an all-green full-suite claim.

Tests ran with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, using identical Python/PyTorch
and normal local shared-memory/loopback access. Refactor-only layout/build checks
then passed 28 tests with one native-extension skip; the focused follow-up passed
232 tests with 23 accelerator skips. No tests were weakened or newly skipped to
obtain this comparison.

The CI-like MyPy environment without optional accelerator packages passes. With
PyTorch's types installed, baseline and candidate both have the same 14 existing
type errors. Native source comparison excludes relocated includes/comments;
Python operator comparison excludes imports, docstrings and source-root offsets.
The numerical bodies match. Archived files are checked against original Git
blobs, including line endings.

Source and wheel builds, wheel imports/resources/entry points from outside the
checkout, shell syntax and strict documentation build were checked. Docker
images and native CUDA/ROCm extensions were not built on this macOS CPU host.
GPU numerical acceptance and performance measurements remain pending; use
`docs/validation/refactor-acceptance.md`. Performance differences have no automatic
percentage threshold.
