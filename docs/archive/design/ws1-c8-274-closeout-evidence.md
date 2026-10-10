# WS1 C8 (#274) closeout evidence

**Parent:** #266 · **Depends on:** C3 / C4 · **Not a substitute for #150 / C10**

## Execute result

Checked-in matrix: `docs/archive/design/ws1-c8-execute.json` (`schema_version: ws1-c8-execute-v2`)

```bash
python tools/validation/operators/sweep_ws1_four_judgments.py --execute --json
```

| Field | Value |
| --- | --- |
| Source commit | `fdf5bcc5165820abb506291a29370225306514ca` |
| Branch | `feat/ws1-c6-c11-closeout-266` |
| GPU | NVIDIA H20, CC 9.0 |
| Driver | 580.76.05 |
| CUDA / PyTorch / Triton | 12.8 / 2.8.0+cu128 / 3.4.0 |
| Workload | `ws1-qwen3-8b-dense-primary-v6` (`ws1-c2-v7`) |
| Result | `green=176`, `N/A=16` (`pack`), **red=0**, exit 0 |

The execute JSON was produced on that source commit. The follow-up commit that adds the JSON is evidence-only.

Both `cuda_bf16` and `triton_cuda_bf16` run the same C1 contract and C2 logical workload. Invariance cells are the C3/C4 bitwise gates (`atol=0`, `rtol=0`). Accuracy cells are the C2 `case_id` runner with BF16 candidate vs FP32 reference.

`pack` remains N/A with the C2/C4 layout-helper reason. It is still a first-class
gtest op: `check_operator.py --op pack` and
`tests/validation/operators/test_forward_invariance.py` / `tests/validation/operators/test_gradient_invariance.py` run the
Native pack adapter on C2 pad/pack/chunk layouts. It is **not** a CUDA or Triton
candidate.

`linear_logp` is **not** a WS1 required single-op. C2 marks it
`optional_fused_path`; C4 `optional_fused`; C8 does not include it in
`C8_REQUIRED_OPS`.

Every other required row has short + primary `case_id`s and four green judgments.
Invariance cells record the C3/C4 observed `actual_backend_id` and
`actual_kernel_config_id` (the loaded candidate class path).

## Close criteria map

| #274 AC | Status |
| --- | --- |
| Required rows have reference/candidate or C2 boundary | Pass |
| Separate complete CUDA and Triton matrices, same C1/C2 | Pass |
| Applicable rows run BF16 + FP32 reference | Pass |
| Short + representative full-model tiers on C2 `case_id`s | Pass |
| expected/actual backend + kernel path recorded by the case runner | Pass |
| Invariance cells record observed actual backend + kernel | Pass |
| Every cell green/red/N/A | Pass (execute artifact; classify-only still paints unrun cells red) |
| Applicable + required four judgments green | Pass |
| Batch/Chunk invariance is the C1 bitwise gate | Pass |
| N/A has C2/C4 reason | Pass (`pack`) |
| No Native/Triton/reference masquerade | Pass (Triton attention has its own VJP; SM90 ops fail closed) |
| Zero red | Pass |

C8 all-green is not WS1 EXIT. This artifact was refreshed during the C10/C11 closeout; parent EXIT still requires the final-commit required GPU CI run and issue bookkeeping.
