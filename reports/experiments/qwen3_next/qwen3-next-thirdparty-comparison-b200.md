# Qwen3-Next: Miles and verl measured against RL-Kernel (B200)

[Download raw results (ZIP)](https://github.com/fusheng-ji/RL-Kernel/releases/download/qwen3-next-rfc428-evidence/qwen3-next-thirdparty-comparison-b200.zip).

ZIP SHA-256: `6ae9cddf9037ef2cc600630e56da859c92ec716c8a36fcf6229ae34322575f92`.

Measured at commit `350ea66d` (`measured-commit.txt`; every report has
`tracked_tree_dirty: false`) on one node with four B200 GPUs, torch 2.13.0+cu130,
vLLM 0.30.0, Megatron-core 0.16.0rc0, SGLang 0.5.21, NCCL 2.29.7, the official
`Qwen/Qwen3-Next-80B-A3B-Instruct` revision
`9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`. A first run at `af5ab9c2` gave
the same conclusions in every row.

Miles (`radixark/miles@23d3d38`) and verl (`volcengine/verl@e5a627c`) are not
installed: their own kernels are imported unmodified from checkouts of those
commits (`benchmarks/models/qwen3_next/thirdparty_sources.py`) and the SHA-256 of
every file used is in each report's `third_party`. Two accommodations, neither
touching their kernel code: Miles' MoE forward imports four SGLang functions from
a module path that SGLang 0.5.21 no longer has, so that path is forwarded to the
functions' new location; verl's `__init__` imports `ray` and the trainer, so a
temporary package with empty `__init__` files links verl's real `utils/device.py`,
`utils/kernel/` and `plugin/`.

```bash
# Miles' routed-MoE backward (its own autograd functions) -- SGLang environment
python benchmarks/models/qwen3_next/qwen3_next_miles_moe_backward.py \
    --checkpoint <ckpt> --miles-src <miles> --output miles-moe-backward --repeats 20
# logprob backward of verl, Miles and RL-Kernel -- one GPU
python benchmarks/models/qwen3_next/qwen3_next_logprob_backward.py \
    --checkpoint <ckpt> --miles-src <miles> --verl-src <verl> --output logprob-backward --repeats 20
# logprob forward at TP4 under 13 NCCL settings and TP1 -- training environment (as in #549)
torchrun --nproc-per-node 4 benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py \
    --checkpoint <ckpt> --miles-src <miles> --verl-src <verl> --output logprob/tp4-default
python benchmarks/models/qwen3_next/qwen3_next_logprob_prior_art.py --merge logprob/tp4-* logprob/tp1 \
    --out logprob-report.json
python benchmarks/models/qwen3_next/plot_qwen3_next_thirdparty.py <extracted ZIP dir>
```

## Routed-MoE backward (layer-0 router and 512 experts, 1024 tokens)

Identical inputs, 20 launches; "equal" is bit for bit against the first launch.

| Tensor | Miles | RL-Kernel |
| --- | --- | --- |
| output | 20 / 20 equal | 20 / 20 |
| input gradient `dx` | **1 / 20** (90 % of the elements differ, max abs 7.3e-4) | 20 / 20 |
| `dW` gate/up | **1 / 20** (54 %, max abs 3.7e-4) | 20 / 20 |
| `dW` down | **1 / 20** (0.2 %, max abs 1.2e-4) | 20 / 20 |
| route-weight gradient | 20 / 20 | 20 / 20 |
| `dx` of 8 probe rows alone == inside batches of 8-1024 rows (10 cells) | **0 / 10** | 10 / 10 |
| forward + backward latency | 10.3 ms | 158 ms |

Miles' backward kernels accumulate with `tl.atomic_add` (`fused_experts_bwd.py`)
in BF16. The route-weight gradient and the forward are deterministic because each
output element receives one addition. Miles passes the routing weights to
`fused_experts` in BF16 (its Qwen3-MoE path casts them); its kernels do not
compile with FP32 weights, so the inputs here follow its own usage and RL-Kernel
receives the same values.

## Selected-token logprob

Forward at TP4 (real LM head, 1024 tokens, 13 NCCL settings) and TP1:

| Implementation | TP4 == TP1 | Distinct TP4 results over 13 settings | Repeat (10 calls) | Batch invariant | Max abs vs FP64 | TP4 latency |
| --- | --- | --- | --- | --- | --- | --- |
| RL-Kernel WS2 | yes | **1** | same | yes | 7.8e-7 | 2.55 ms |
| vLLM gather + `log_softmax` | yes | 1 | same | yes | 8.1e-7 | 1.51 ms |
| Megatron-core CE | no | 2 | same | yes | 8.1e-7 | 0.57 ms |
| VIME default | no | 2 | same | yes | 8.1e-7 | 0.56 ms |
| **Miles `fused_log_probs_and_entropy`** | no | **2** | same | yes | 8.1e-7 | 0.33 ms |
| **verl `linear_cross_entropy`** | no | **2** | same | yes | 1.2e-5 ¹ | 0.63 ms |

¹ verl fuses the LM-head GEMM into the kernel, so it computes its own logits
(from the hidden states and its head shard) and is compared with an FP64
log-softmax of that product; its output differs from the shared-logits
candidates by up to 7.7e-3.

Miles and verl sum the per-shard statistics with NCCL all-reduces, like Megatron
and VIME: the TP4 result depends on the NCCL algorithm and differs from TP1. Their
forward is repeatable at a fixed setting. verl's forward `tl.atomic_add` only runs
for `reduction` sum / mean (not the `"none"` used here).

Backward (TP1, the whole 151936-row head, 20 launches):

| Implementation | Gradient | Equal to the first launch | 8 probe rows alone == inside batches (10 cells) |
| --- | --- | --- | --- |
| verl | `d_hidden` | 20 / 20 | **1 / 10** |
| verl | `d_weight` | 20 / 20 | – |
| Miles | `d_logits` (store-only kernels, no atomics) | 20 / 20 | 10 / 10 |
| RL-Kernel WS2 | `d_logits` | 20 / 20 | 10 / 10 |

verl's backward accumulates `d_hidden` and `d_weight` with `tl.atomic_add`, yet
repeated launches gave the same bits here; its row gradient changes with the
batch.

Every result file is JSON with the commit, environment and third-party
provenance; `figure.png` renders them; `SHA256SUMS` covers every file. Site paths,
host names and the scheduler do not appear in any file.
