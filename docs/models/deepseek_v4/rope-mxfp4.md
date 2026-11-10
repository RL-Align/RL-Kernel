# Partial RoPE and Indexer Q preprocessing (items 11, 15)

Two operators, no new GPU kernel. Runtime wiring is a separate PR.

```text
item 11: x[T,H,D] BF16, pos[T] int64 -> y[T,H,D] BF16      D in {128, 512}, rope_dim 64
item 15: q[T,H,128] BF16, pos[T]     -> values[T,H,128] BF16, packed[T,H,64] uint8,
                                        scales[T,H,4] uint8 (E8M0, not shuffled)
```

Item 15 follows DeepSeek `model.py` lines 413-416: `apply_rotary_emb(q[..., -64:])`,
`rotate_activation(q)`, `fp4_act_quant(q, 32)`. `values` is the BF16 tensor before
quantization; the ICV path consumes `packed`/`scales`.

## Item 11: `rope_gptj_interleaved_partial`

`PartialRoPE(PartialRoPEContract(head_dim, rope_dim=64))`. Only the last 64 dims
rotate, as GPT-J interleaved pairs `(2i, 2i+1)`; `y[..., :D-64]` is a byte copy.
`D=128` is the Indexer branch `[64:128]`, `D=512` is the main branch `[448:512]`.

|Step|Precision|
|---|---|
|cos/sin table|BF16 or FP32 `[positions, 32]`, read-only|
|lookup|`index_select` by global position, then cast to FP32 `[T,1,1,32]`|
|rotation|aiter `rope_cached_fwd` (`rotate_style=1, reuse_freqs_front_part, nope_first`), FP32 math, one BF16 rounding|
|backward|aiter `rope_cached_bwd` on BF16 dY: the transpose of the rotation|

Positions are global token positions; `-1` marks padding, whose rows output 0
and receive 0 gradient. A CP shard passes its own global positions, so its rows
equal the same rows of the unsharded call. These are the calls #631
`query_transform` and #627 `inverse_rope` already make; the forward of item 11
is byte-equal to #627's `inverse_rope_gradient`, and its backward to
`inverse_rope`.

## Item 15: `indexer_q_rope_hadamard_mxfp4`

`IndexerQPreprocess(IndexerQPreprocessContract(num_heads))` composes item 11
(`D=128`), #624's `_rotate` (aiter `rotate_activation`, rows padded to an 8-row
wave on gfx942) and #624's `quantize` (`per_1x32_f4_quant_triton`, `shuffle=False`).

**Scale rule: even** (the #624 rule already used for Indexer K):

```text
amax  = max |x| over 32 values (FP32)
E8M0  = floor(log2(round_mantissa_up(amax))) - 2, clamped to [-127, 127]; byte = E8M0 + 127
code  = E2M1(x / 2^E8M0), round to nearest even, saturate at 6
byte  = code[2i] | code[2i+1] << 4
```

`round_mantissa_up` adds `0x200000` to the FP32 bits and clears the mantissa.
DeepSeek, vLLM `fused_indexer_q.py` and the aiter HIP kernel use
`ceil(log2(amax/6))`. The two rules differ in about 12% of Hadamard-rotated
random groups. Example: amax = 6.5 gives even byte 127 (6.5 saturates to 6),
ceil byte 128 (6.5 encodes as 3 x 2). Rollout and training must both use this
operator to agree byte for byte; vLLM's own Q kernel will not.

## Gradient rule

Identity straight-through on the decoded value, the same rule as #631
(`dequantized_operand_identity_ste.v1`):

```text
dq = RoPE^T( H^T( d_values ) )      H symmetric, so H^T = H
```

`packed` and `scales` are not differentiable. The VJP runs the same native
Hadamard, then `rope_cached_bwd`. Upstream (`x`, `qr`) is detached in DSA
training, so this gradient only reaches Indexer weights.

## Use

```python
from rl_engine.integrations.engines.rollout.vllm.indexer_q_preprocess import (
    contracts_from_hf_config,
)
from rl_engine.models.deepseek_v4.indexer_q_preprocess import IndexerQPreprocess
from rl_engine.models.deepseek_v4.partial_rope import PartialRoPE

rope_c, q_c = contracts_from_hf_config(hf_config)
q_rope = PartialRoPE(rope_c)(q_main, positions, cos_table, sin_table)    # [T,H,512]
out = IndexerQPreprocess(q_c)(q_index, positions, cos_table, sin_table)  # out.packed / out.scales
```

Both accept `expected_implementation` to reject a different contract, backend
source or plan. Inputs must be contiguous BF16 on the backend device, outside
CUDA Graph capture. Only gfx942 is admitted.

## Verification

```bash
python -m pytest tests/models/deepseek_v4/test_rope_mxfp4.py \
  tests/models/deepseek_v4/test_rope_mxfp4_rocm.py -q
python benchmarks/benchmark_rope_mxfp4.py --repeats 30 --output benchmark.json
```

Accuracy against FP64 uses `assert_close`. Byte equality is checked for NoPE
dims, #627 inverse round trip, #631 `transform_queries` + `encode` (values,
codes, scales and dQ), a torch MXFP4 oracle, row splits, shuffles, CP 2/4
shards and two fresh processes. Results: `reports/rope-mxfp4-mi300x-20261110.md`.

Not covered: the aiter fused `rope_rotate_activation_fp4quant_inplace` (gfx950
only), the ceil rule, runtime wiring of main-branch Q/K RoPE, CUDA Graph and
multi-GPU CP.

References: [#624](https://github.com/RL-Align/RL-Kernel/pull/624),
[#627](https://github.com/RL-Align/RL-Kernel/pull/627),
[#631](https://github.com/RL-Align/RL-Kernel/pull/631),
[#634](https://github.com/RL-Align/RL-Kernel/pull/634),
[DeepSeek inference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py).
