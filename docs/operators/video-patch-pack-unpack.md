# MiniMax-H3 video patch pack/unpack

## Summary

Rearrange video latents from `(B,24,T,H,W)` into Transformer patch tokens
`(B,S,96)`, where `S=T*(H/2)*(W/2)`. `unpack` restores the original layout
exactly. Each token contains the 24 channels of one `2x2` region in a single
frame: `24*1*2*2=96` values.

The operator copies values without arithmetic. Backward applies the inverse
permutation. It supports MiniMax-H3 video layout conversion; the caller owns
video projections, multimodal sequence assembly, and full model execution.

The public name is `video_patch_pack_unpack`. Its current contract fixes H3's
24 channels and `(1,2,2)` patch; other model configurations are unsupported.

## Entry Point

Run this example from the repository root using the CPU PyTorch backend:

```python
import torch

from rl_engine.runtime.registry import KernelRegistry

x = torch.arange(24 * 2 * 4 * 6, dtype=torch.float32).reshape(1, 24, 2, 4, 6)
op = KernelRegistry().get_op("video_patch_pack_unpack", device=x.device)
tokens = op.pack(x)
recovered = op.unpack(tokens, tuple(x.shape))

assert tokens.shape == (1, 12, 96)
assert torch.equal(recovered, x)
print(op.backend_id, tuple(tokens.shape))
```

For CUDA inputs, the same entry point executes the reused functions through
PyTorch's CUDA tensor operations. `unpack` requires the original latent shape
because the token shape alone does not uniquely determine `T,H,W`.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch on CPU / NVIDIA CUDA | `rl_engine.reference.packing.video_patch_pack_unpack.NativeVideoPatchPackUnpackOp` | Vendored vLLM-Omni video functions; PyTorch tensor operations | Strict CPU and single-GPU CUDA capability; no custom compiled extension |
| PyTorch on ROCm / NPU | Same wrapper | Platform tensor operations | Ordinary dispatch only; target-hardware validation pending |

### Existing Implementations and Reuse

The runtime directly reuses `minimax_h3_patchify_video_latent` and
`minimax_h3_unpatchify_video_tokens`, including their `_rank` and `_int_tuple`
helpers, from [vLLM-Omni's H3 packed tokens](https://github.com/vllm-project/vllm-omni/blob/b1fc5fcfa3e1a4f76c6686eaf09242f0fc6464bc/vllm_omni/diffusion/models/minimax_h3/packed_tokens.py).
The selected function bodies are vendored unchanged under Apache-2.0; source
attribution and the pin are retained in the module. No vLLM-Omni runtime install
is required.

| Source identity | Value |
| --- | --- |
| Revision | `b1fc5fcfa3e1a4f76c6686eaf09242f0fc6464bc` |
| Source | `vllm_omni/diffusion/models/minimax_h3/packed_tokens.py` |
| Full source SHA-256 | `20994e723054071e34acc41398e9d1636c226802e49894ac9162f5a808feb06f` |

The functions use reshape, an einsum that only permutes axes, and contiguous.
Each output has exactly one source element; there is no reduction or interaction
between samples. Their autograd derivatives are inverse permutations. This makes
the algorithm batch-invariant; focused tests check outputs and gradients with
changed batch sizes, sample positions, unrelated samples, and temporal padding.
Device coverage is limited to the hardware actually tested.

The public adapter fixes H3's channels and patch size, checks the input contract,
and converts the upstream `(B*S,96)` output to `(B,S,96)` with a view. Unpack
passes the patch-grid shape `(T,H/2,W/2,24)` to upstream, rather than the full
latent shape. No new copy is introduced by these batch views.

[SGLang's pinned H3 implementation](https://github.com/sgl-project/sglang/blob/ad8b1e1f851a6445e1211f7f0cf2130028ecae0c/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_tokens.py)
provides the same native permutation. Diffusers 0.40.0 supplies a separate
correctness and performance reference. Qwen-Image #410's tile design was used
by the earlier custom kernel; this contribution now uses the existing BI native
implementation directly, as requested in issue #420.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` in `pack(x)` | `(B,24,T,H,W)` | FP16, BF16, FP32 | Contiguous; `B,T,H,W>0`; even `H,W` |
| `pack` output / `tokens` in `unpack(tokens, shape)` | `(B,T*(H/2)*(W/2),96)` | Same as latents | Contiguous unpack input with dimensions matching `shape` |
| `shape` in `unpack` | `(B,24,T,H,W)` | Five integers in a `tuple` or `torch.Size` | Same shape restrictions as `pack` |
| `unpack` output | `(B,24,T,H,W)` | Same as `tokens` | Contiguous; same device as the input |

Tokens are ordered by batch, frame, spatial patch row, and spatial patch column.
Features within a token are ordered by channel, patch row, and patch column:

```text
tokens[b, t*(H/2)*(W/2) + hi*(W/2) + wi, c*4 + dy*2 + dx]
    = latents[b, c, t, 2*hi + dy, 2*wi + dx]
```

The public adapter rejects invalid inputs with explicit errors. The upstream
vLLM-Omni and Diffusers paths return `(B*S,96)`; comparisons change only the batch
view, preserving element order.

## Dispatch Behavior

`KernelRegistry.get_op("video_patch_pack_unpack", device=...)` selects the
PyTorch reuse wrapper for CPU, CUDA, ROCm, and NPU. ROCm and NPU are available for
experimental ordinary dispatch; they are not declared as validated strict
capabilities. No dedicated operator getter or custom CUDA/Triton fallback exists.

Strict execution uses the existing semantic registry session:

```python
from rl_engine.runtime.registry import KernelRegistry
from rl_engine.runtime.semantic_registry import OperatorRequirements

session = KernelRegistry().semantic.session()
resolution = session.resolve(
    semantic_op="video_patch_pack_unpack",
    requested_backend="pytorch-video-patch-pack-unpack-v1",
    target="training",
    requirements=OperatorRequirements(
        device="cpu",
        dtype="float32",
        topology={"world_size": 1, "sharding": "unsharded"},
        alignment_properties={
            "channels": 24,
            "patch_size": {"t": 1, "h": 2, "w": 2},
            "bitwise": True,
        },
    ),
)
op = session.instantiate(resolution)
print(session.instance_provenance(resolution, op).to_dict())
```

Use `device="cuda"` with the same backend ID for NVIDIA. The implementation
fingerprint includes the pinned vLLM-Omni revision and adapter version. Strict
requests for retired CUDA/Triton backend IDs, unvalidated devices, unsupported
dtypes, or incompatible topology fail without selecting another backend.

## Accuracy

The independent PyTorch gold uses `reshape -> permute -> reshape -> contiguous`
for the permutation and its inverse. Each output corresponds to one input value,
with no floating-point arithmetic or dtype conversion. Outputs and first-order
gradients must therefore match bitwise with zero numerical tolerance.

Tests compare raw tensor bytes, including signed zero, infinities, and NaN
payloads. Ordinary floating-point equality does not cover all these cases.
PyTorch autograd handles noncontiguous upstream gradients and higher-order
`create_graph=True` derivatives without a custom backward implementation.

The second reference comes from `patchify_video_latents` and the video inverse
statements in `MiniMaxH3AfterDenoiseStep.__call__`, installed from
`diffusers==0.40.0`. Tests and benchmarks share the benchmark's `DiffusersVideoPatch`
adapter. It executes the upstream tensor statements unchanged, records source
SHA-256 hashes, and does not call RL-Kernel's reference permutation. The upstream
decoder entry uses `no_grad`; the unpack gradient reference is autograd applied
to the extracted tensor statements.

The shared `OP_SPECS` harness registers `video_patch_pack_unpack` as an
`elementwise` operator with the PyTorch reuse candidate. A validation-only
adapter maps its callable interface to `.pack()` or `.unpack()`; the runtime API
is unchanged. The gold path promotes the independently generated input to FP32
before applying the independent gold permutation. A negative-control test breaks
the reused functions and checks that the gold still works. Both directions declare `x` as the
gradient input and use the shared forward/backward accuracy profile. The focused
pytest tests additionally require exact bits and cover special floating-point values.

Select the direction and original latent geometry explicitly:

```bash
python tools/validation/operators/check_operator.py \
  --op video_patch_pack_unpack --candidate pytorch --device cpu \
  --dtype bf16 --patch-direction unpack \
  --batch 2 --frames 3 --height 6 --width 10 --check-grad --json
```

For unpack, the input builder generates `(B,S,96)` tokens directly, rather than
packing a latent input first. `--frames/--height/--width` describe the original
latent shape in either direction; they do not change the fixed 24 channels or
the `(1,2,2)` patch. `--seq` is unused for this operator.

## Performance Notes

After completing the NVIDIA setup and correctness checks below, run
from the repository root:

```bash
python benchmarks/operators/packing/benchmark_video_patch_pack_unpack.py \
  --output-dir _dev_notes/h3-benchmark
```

Default shapes use latent dimensions for `H,W`:

| Preset | `(T,H,W)` | Tokens per sample |
| --- | --- | --- |
| `small` | `(3,6,10)` | 45 |
| `medium` | `(8,24,42)` | 2016 |
| `768p` | `(32,48,84)` | 32256 |

Reproduce the workload report's 28 layouts across three dtypes:

```bash
python benchmarks/operators/packing/benchmark_video_patch_pack_unpack.py \
  --workloads --output-dir _dev_notes/h3-benchmark/workloads
```

`--workloads` covers six aspect ratios at `T=37/72/102, B=1`, landscape batch
sweeps at `T=37/102, B=2/3/4/8`, and landscape/portrait conditions at `T=1, B=1`.
It replaces `--shapes` and `--batches`. The pinned upstream spatial compression
factor is 16. Aligned video frame counts `F=17n+5` map to latent `T=5n+2`, so
124/243/345 frames map to `T=37/72/102`. Geometry and frame alignment come from
the pinned
[modular_pipeline.py](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/minimax_h3/modular_pipeline.py).
Use `--shapes 16x9_t37,9x16_t72,16x9_t102,16x9_t1 --batches 1` for selected layouts.

Each default preset sweeps `B=1,2,4,8` and FP32/FP16/BF16, comparing the RL-Kernel reuse
wrapper, direct calls to the pinned vendored vLLM-Omni functions, and Diffusers. Pack and unpack each
measure forward, backward, and forward plus backward. Outputs and first-order
gradients are checked for bitwise equality before timing.

Eager CUDA events and synchronized wall time measure individual calls. CUDA Graph
timing divides replay time by the number of captured calls; report it separately.
Defaults are 10 warmups, 48 samples, and 32 calls per graph. Backward-only timing
uses a forward graph prepared outside the measured interval.

`results.json` records raw samples, device/runtime versions, source revision and
workspace state, strict resolution traces, actual implementation identities,
pinned upstream revision and source hashes, and peak allocated
VRAM. Keep reports, figures, and raw results under the local `_dev_notes/`
directory. Put summary tables and figures on the PR page. Performance
claims must identify the shape, dtype, timing method, and source revision.
Operator speedups do not establish full model speedups.

The earlier custom CUDA/Triton speedups do not apply to this reuse revision.
Measure the reuse wrapper and direct upstream calls on the same GPU before
reporting their latency or adapter overhead. These measurements cover the tensor
functions, not the complete vLLM-Omni or Diffusers inference pipelines.

## Tests

### Local Development Checks

Use Python >=3.10 with PyTorch installed for the target device. Run these commands
from the repository root; CPU development does not require the CUDA extension:

```bash
python -m pip install --no-build-isolation -e ".[dev]"
python -m pytest tests/ops/packing/test_video_patch_pack_unpack.py \
  tests/ops/packing/test_video_patch_pack_unpack_diffusers.py \
  tests/runtime/test_kernel_registry.py tests/validation/operators/test_operator_inputs.py tests/validation/operators/test_op_checks.py -q -rs
```

Tests explicitly skip with a reason if the optional pinned Diffusers version,
or GPU is unavailable. Other CPU contract tests still run.
With the correct Diffusers version installed, source extraction or comparison
errors fail the tests. Local runs with GPU skips do not establish GPU acceptance.

| Test file | Coverage |
| --- | --- |
| `tests/ops/packing/test_video_patch_pack_unpack.py` | Independent coordinate order, invalid inputs, ordinary/strict dispatch, retired-backend rejection, independent-gold negative control, both directions and gradients, special bits, batch invariance, irregular patch columns, noncontiguous gradients, and higher-order gradients |
| `tests/ops/packing/test_video_patch_pack_unpack_diffusers.py` | Reused PyTorch CPU/CUDA outputs and first-order gradients against upstream pack/unpack; six shapes including T=37/72/102 videos and T=1 conditions across three dtypes |

### NVIDIA GPU Operator Acceptance

Use a CUDA-enabled PyTorch installation with one NVIDIA GPU visible. The
operator does not require the RL-Kernel extension or a local CUDA compiler:

```bash
python -m pip install pytest 'diffusers==0.40.0'
mkdir -p _dev_notes/h3-benchmark
python -m pytest tests/ops/packing/test_video_patch_pack_unpack.py \
  tests/ops/packing/test_video_patch_pack_unpack_diffusers.py \
  -q -rs --junitxml=_dev_notes/h3-benchmark/validation.xml
```

Pytest must exit successfully with zero skips. Check the XML so a missing
dependency or backend cannot produce a passing acceptance result through skips:

```bash
python - <<'PY'
import xml.etree.ElementTree as ET

root = ET.parse('_dev_notes/h3-benchmark/validation.xml').getroot()
assert root.findall('.//testcase'), 'No tests executed'
for status in ('skipped', 'failure', 'error'):
    assert not root.findall(f'.//{status}'), f'GPU acceptance contains {status}'
print('GPU acceptance: all tests executed and passed')
PY
```

Run the shared accuracy harness for each dtype and direction. Each JSON
must report `passed=true`; the focused pytest suite remains the bitwise gate:

```bash
for dtype in fp32 bf16 fp16; do
  for direction in pack unpack; do
    python tools/validation/operators/check_operator.py \
      --op video_patch_pack_unpack --candidate pytorch --device cuda \
      --dtype "$dtype" --patch-direction "$direction" \
      --batch 2 --frames 3 --height 6 --width 10 --check-grad --json \
      > "_dev_notes/h3-benchmark/gtest-pytorch-${dtype}-${direction}.json"
  done
done
```

These commands target NVIDIA. ROCm and NPU acceptance is pending.

### Formatting and Documentation Checks

```bash
python -m pip install pre-commit mypy -r requirements/docs.txt
pre-commit run --all-files
mypy --ignore-missing-imports rl_engine/
mkdocs build --strict -f mkdocs.yaml
```

## Known Limitations

- Fixed `C=24`, patch `(1,2,2)`, and token width 96; other configurations are unsupported.
- Contiguous inputs required; zero dimensions, odd `H/W`, FP64, and integer dtypes are rejected.
- ROCm and NPU ordinary dispatch lack target-hardware validation.
- The benchmark targets one NVIDIA GPU and measures an operator. It does not load model weights,
  the VAE, or the full Transformer; the Diffusers comparison does not establish model-level
  end-to-end validation.
