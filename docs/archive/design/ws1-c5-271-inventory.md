# WS1 C5 (#271) elementwise / RoPE inventory

**Parent:** #266 · **Depends on:** C2 / C3 / C4 · **Does not wait for C8 close**

C5 is a written inventory. Differentiable on-chain items reuse C3/C4. CUDA
RoPE is the declared `cuda-sm90` candidate; H20 C3/C4/C8 are green. No
sm86-reproducible elementwise or RoPE defect remains open.

## Inventory

| Item | CUDA | Triton | Evidence |
| --- | --- | --- | --- |
| `rope` | pass (`cuda-sm90` on H20) | pass | C3/C4 + C8; `[S]`/`[B,S]` + packed-position tests |
| `silu` | pass | pass | C3 + C4 green both profiles |
| `swiglu` | pass | pass | C3 + C4 green both profiles |
| `residual_add` | pass | pass | `torch.add`; no cross-batch reduction |
| `scale` | pass | pass | `1/sqrt(head_dim)` broadcast |
| `bias` | pass | pass | official fingerprint `attention_bias=false` |
| `mask_fill` | pass | pass | Triton valid KV interval is rebased to logical reduction lanes |
| `dtype_cast` | pass | pass | C1 policy; provenance rejects drift |

Source of truth: `rl_engine/validation/operators/elementwise_inventory.py`.

## Hopper evidence

H20 C8 execute recorded CUDA `rope` four-judgment green. Re-check with:

```bash
python tools/validation/operators/check_forward_invariance.py --op rope --candidate cuda-sm90 --backend-profile cuda_bf16
python tools/validation/operators/check_gradient_invariance.py --op rope --candidate cuda-sm90 --backend-profile cuda_bf16
```
