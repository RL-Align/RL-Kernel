# Batch-Invariant Elementwise and RoPE Audit

Issue #149 audits the forward-path operations that should be pass-through with
respect to batch configuration. The goal is to prove that a row's output is
unchanged when unrelated rows are added, the row moves to a different batch
position, or padding/packed-sequence position ids are present.

## Scope

This audit covers Transformer pointwise operations and RoPE only. RMSNorm,
matmul, attention, selected logprob, and FP8 remain out of scope because they
have separate roadmap issues and reduction contracts.

## Tested Contract

The shared sweep helper lives in `rl_engine.testing.forward_invariance`:

- `BatchInvariantConfig`
- `DEFAULT_BATCH_INVARIANT_SWEEP`
- `assert_batch_invariant_across_configs`

The executable contract lives in `tests/test_forward_invariance.py` and calls
that helper for every audited elementwise operation and for fixed-position RoPE.

Each audited operation is evaluated on a target row in isolation, then on the
same target row embedded into larger batches with unrelated noise rows. The
target output must be bitwise identical with `torch.equal` unless a tight
operator-specific tolerance is documented below.

| Sweep case | Batch size | Target row |
| --- | ---: | ---: |
| `batch1` | 1 | 0 |
| `batch2-first` | 2 | 0 |
| `batch4-middle` | 4 | 2 |
| `batch9-last` | 9 | 8 |

| Operation | Verdict | Coverage |
| --- | --- | --- |
| SiLU activation | Pass | Pointwise; no batch-dependent reduction. |
| GELU activation | Pass | Pointwise; uses `atol=1e-6, rtol=1e-6` for CPU libm/vector paths. |
| Residual add | Pass | Depends only on the matching residual row. |
| Scalar scaling | Pass | Scalar multiply has no accumulation. |
| Bias add | Pass | Broadcast bias is independent of batch position. |
| Mask fill | Pass | Uses only the value/mask pair for each element. |
| Explicit dtype cast | Pass | fp32/fp16/bf16 casts are batch-invariant. |

CUDA runs cover fp32, fp16, and bf16 when the device supports bf16. CPU CI covers
fp32 for the arithmetic elementwise cases and fp32/fp16/bf16 for explicit dtype
casts.

## RoPE Contract

`rl_engine.testing.build_rope_cache` builds a deterministic table-lookup cache:

1. Compute fp32 inverse frequencies from `base` and `head_dim`.
2. Build `[max_position, head_dim]` cos/sin tables.
3. `apply_rope_reference` gathers rows with explicit `position_ids`.
4. Gathered cos/sin values are cast to the Q/K dtype before the multiply/add.

The reference path has no batch-dependent reduction, no launch-shape-dependent
accumulation, and no inline recomputation that can vary by batch shape.

| RoPE case | Verdict | Coverage |
| --- | --- | --- |
| Fixed position | Pass | Same Q/K token and position id stays bitwise identical. |
| Batch position changes | Pass | Target row can move without drift. |
| Unrelated noise rows | Pass | Noise Q/K rows and position ids do not affect target output. |
| Padding | Pass | Valid tokens keep the same output when padding is inserted. |
| Packed-sequence reset | Pass | Local ids `[0, 1, 2]` match the standalone segment. |
| Position-id validation | Pass | Invalid position ids and cache shapes are rejected. |

## Verification

The CPU CI unit-test job runs `tests/test_forward_invariance.py` directly. The
standard `tests/` suite run by GPU CI also includes it when GPU CI is requested.

Run the focused contract:

```bash
python -m pytest tests/test_forward_invariance.py -q
```

Run the helper regression suite:

```bash
python -m pytest tests/test_forward_invariance.py tests/test_reference_ops.py -q
```

Build the documentation after changing this page:

```bash
mkdocs build --strict -f mkdocs.yaml
```

## Known Boundaries

This audit adds the shared sweep helper used by the #149 tests, a PyTorch RoPE
reference contract, and documentation. It does not add a production RoPE kernel,
change runtime dispatch, or close issues assigned to RMSNorm, matmul, attention,
logprob, or FP8.

If #108 later lands a broader model-level harness, these tests should bridge to
that harness without changing the pass/fail expectations above.
