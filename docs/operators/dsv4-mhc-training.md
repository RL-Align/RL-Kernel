# DSV4 mHC training (task 6)

Use `mhc_pre_norm` before a sublayer, `mhc_post` after it, and `mhc_head`
after the final layer. The first call returns post and combination coefficients,
the normalized sublayer input and the separate, unnormalized task-5 row.
Post consumes the original four streams; do not add the raw row again.

The opt-in `rocm` backend runs RLK's row arithmetic on gfx942 or gfx950. It
uses ordinary on-device tensor operations for correctness bring-up, not a
throughput-optimized mHC kernel. The explicitly selected `reference` path
requires CPU. `rocm_aiter` is a forward-only comparison against #662 on
gfx942, hidden size 4096. It runs each row at M=1, so neither
the selected configuration nor the compiled M changes with batch packing.
Use `torch.no_grad()` or `torch.inference_mode()` to compare with trainable
inputs or weights. With gradients enabled, an input that requires grad is
rejected before dispatch. The comparison requires #662 and its pinned aiter
build; it never falls back to RLK. All modes use task 5 for sublayer
RMSNorm. Use `backend="rocm"` for both training and rollout in a bitwise run.

The first-order VJP uses rsqrt(mean(square) + 1e-6), the FP32 projection of
DeepSeek-V4's `inference/model.py`, and asymmetric Sinkhorn. The owned forward
uses explicit adjacent-pair sums, which can round differently from AITER's
fused instructions. Its saved state and VJP are used only for the owned and
reference paths, never for AITER outputs.

Save mode records the per-token gates and Sinkhorn normalization results;
recompute mode rebuilds them from version-checked inputs. Frozen parameters
receive no gradient. Parameter gradients accumulate FP32 row contributions in
ascending local token order; padding must carry zero loss gradient. This does
not define a distributed or cross-microbatch optimizer accumulation policy.

Run task-6 CPU and device tests with
`python -m pytest -q tests/ops/mhc/test_p1_training.py`. Set
`RLK_REQUIRE_ROCM=1` for a hardware acceptance run. Run the aiter case only
after installing #662. Record the GPU, torch, Triton, aiter revision, backend
and shape. These are local operator checks; full-model consistency, performance
and Graph replay are separate measurements.
