# CLAUDE.md — DSv4 CUDA work on `test-dsv4-cuda` (PR #635)

Notes for an assistant (or developer) continuing the DSv4 SM90 batch-invariant work. The
design of record is `docs/architecture/dsv4-design.md`: goal, environment, how to build
and test, module map, decisions, the reduction order of every module, status and open
work. Read it first and keep its status section current. Last updated 2026-11-11.

## Working rules

- Commits: conventional style (`feat(dsv4): ...`, `fix(dsv4): ...`), `git commit -s` (DCO),
  end with the Co-Authored-By line. Split pure formatting from logic changes.
- Push only when asked.
- Every forward change must keep the DSv4 regression green (design doc 2.3) and the layerwise
  gate at zero (`scripts/check_dsv4_layerwise_gate.py`). New kernels get a PyTorch golden
  defined on CUDA, a byte-equality test and a batch-invariance test.
- A kernel that cannot hold an existing numeric profile declares a new one.
- Performance gate: 1.3x of the non-invariant path, measured with CUDA-graph replay.

## Environment

Requirements, build flags and the test / benchmark commands are in the design doc,
section 2. Multi-GPU work (EP 4/8, DeepEP, verl end to end) needs a machine with more than
two GPUs per job.

## Numerics lessons

See the design doc, section 13.
