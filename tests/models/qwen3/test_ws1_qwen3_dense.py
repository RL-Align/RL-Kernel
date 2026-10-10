# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU-safe C9 topology / profile / identity tests. No 16 GB allocation."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from rl_engine.config.workload import load_manifest, weight_snapshot_hash
from rl_engine.models.qwen3.reference import (
    NODE_KINDS,
    OFFICIAL_FINGERPRINT,
    ProfileOps,
    Qwen3DenseBIModel,
    Qwen3DenseSpec,
    _CanonicalChunkedAttentionFn,
    load_profile_ops,
    verify_hf_weight_snapshot,
)
from rl_engine.reference.attention.standard_attn import NativeAttentionOp
from rl_engine.reference.attention.stateful_kv import StatefulKVCache


@pytest.fixture()
def manifest():
    return load_manifest()


def test_c9_spec_matches_official_fingerprint(manifest):
    spec = Qwen3DenseSpec.from_manifest(manifest)
    assert spec.num_hidden_layers == 36
    assert spec.hidden_size == 4096
    assert spec.num_attention_heads == 32
    assert spec.num_key_value_heads == 8
    assert spec.head_dim == 128
    assert spec.vocab_size == 151936
    assert spec.qk_norm is True
    assert spec.tie_word_embeddings is False
    assert spec.swiglu is True
    for key, expected in OFFICIAL_FINGERPRINT.items():
        assert getattr(spec, key) == expected


def test_c9_node_names_cover_full_topology(manifest):
    spec = Qwen3DenseSpec.from_manifest(manifest)
    names = spec.node_names()
    assert names[0] == "embedding"
    assert names[-4:] == ("final_layernorm", "lm_head", "logprob", "loss")
    assert any(name == "layers.0.attn" for name in names)
    assert any(name == "layers.35.swiglu" for name in names)
    assert any(name == "layers.0.q_norm" for name in names)
    # 1 embed + 36 * 17 layer nodes + 4 tail
    assert len(names) == 1 + 36 * 17 + 4


def test_c9_forbids_shrinking_layers(manifest):
    raw = dict(manifest.raw)
    ident = dict(raw["model_identity"])
    fp = dict(ident["config_fingerprint"])
    fp["num_hidden_layers"] = 2
    ident["config_fingerprint"] = fp
    raw["model_identity"] = ident
    from rl_engine.config.workload import WS1Manifest

    shrunk = WS1Manifest(raw=raw, path=manifest.path)
    with pytest.raises(ValueError, match="architecture shrink"):
        Qwen3DenseSpec.from_manifest(shrunk)


def test_c9_gold_profile_ops_resolve_without_gpu(manifest):
    ops = load_profile_ops("cuda_bf16", manifest, allow_pytorch_gold=True)
    for kind in (
        "embedding",
        "rms_norm",
        "det_gemm",
        "qk_norm",
        "rope",
        "attention",
        "swiglu",
        "lm_head",
        "logprob",
    ):
        assert ops.get(kind) is not None
        assert ops.provenance[kind]["status"] == "gold_reference"


def test_c9_declared_profile_candidates_are_family_correct(manifest):
    from rl_engine.validation.operators.gradient_adapters import (
        get_adapter,
        resolve_profile_candidate,
    )

    checks = {
        ("cuda_bf16", "attention"): "cuda",
        ("triton_cuda_bf16", "attention"): "triton",
        ("cuda_bf16", "det_gemm"): "cuda",
        ("triton_cuda_bf16", "det_gemm"): "triton",
        ("cuda_bf16", "embedding"): "cuda-sm90",
        ("triton_cuda_bf16", "embedding"): "triton",
        ("cuda_bf16", "logp"): "cuda",
        ("triton_cuda_bf16", "logp"): "triton",
    }
    for (profile, op_name), expected in checks.items():
        resolved = resolve_profile_candidate(get_adapter(op_name), profile, manifest)
        assert resolved["status"] == "declared"
        assert resolved["expected_backend_id"] == expected
        assert resolved["candidate_path"]


def test_c9_weight_snapshot_verifies_real_bytes(manifest, tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    shard = tmp_path / "model-00001-of-00001.safetensors"
    index.write_bytes(b'{"weight_map": {}}\n')
    shard.write_bytes(b"pinned-test-shard")
    shard_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
    records = [
        {
            "filename": shard.name,
            "sha256": shard_hash,
            "size_bytes": shard.stat().st_size,
        }
    ]
    spec = replace(
        Qwen3DenseSpec.from_manifest(manifest),
        weight_index_file=index.name,
        weight_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        weight_shards=((shard.name, shard_hash, shard.stat().st_size),),
        weight_content_hash=weight_snapshot_hash(records),
    )
    assert verify_hf_weight_snapshot(spec, tmp_path) == [shard]

    shard.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="size mismatch|SHA-256 mismatch"):
        verify_hf_weight_snapshot(spec, tmp_path)


def test_c9_runtime_observations_are_complete_and_class_checked():
    class FakeOp:
        pass

    op = FakeOp()
    path = f"{FakeOp.__module__}.{FakeOp.__qualname__}"
    provenance = {
        kind: {
            "requested_backend": "pytorch",
            "actual_backend": "pytorch",
            "candidate_path": path,
            "status": "gold_reference",
        }
        for kind in NODE_KINDS
    }
    profile = ProfileOps(
        backend_profile="test_gold",
        ops={kind: op for kind in NODE_KINDS},
        provenance=provenance,
    )
    for kind in NODE_KINDS:
        profile.observe(kind, torch.ones(1))
    observations = profile.validated_runtime_observations()
    assert set(observations) == set(NODE_KINDS)
    assert all(item["execution_count"] == 1 for item in observations.values())


def test_c9_chunked_attention_mask_covers_cached_prefix():
    cache = StatefulKVCache.allocate(
        n_layers=1,
        batch=2,
        n_kv_heads=1,
        max_seq_len=8,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    prefix = torch.zeros(2, 1, 3, 2)
    cache.write(prefix, prefix, layer=0)
    current_mask = torch.tensor([[True, True], [True, False]])
    model = object.__new__(Qwen3DenseBIModel)
    combined = model._key_padding_mask(current_mask, cache, layer=0, new_len=current_mask.shape[1])
    assert combined.shape == (2, 5)
    assert torch.equal(combined[:, :3], torch.ones(2, 3, dtype=torch.bool))
    assert torch.equal(combined[:, 3:], current_mask)


def test_c9_chunked_attention_uses_real_chunks_and_canonical_backward():
    class RecordingAttention:
        def __init__(self):
            self.op = NativeAttentionOp()
            self.calls = []

        def forward_fp32(self, q, k, v, **kwargs):
            self.calls.append((q.shape[2], k.shape[2]))
            return self.op.forward_fp32(q, k, v, **kwargs)

    torch.manual_seed(7)
    op = RecordingAttention()
    q = torch.randn(1, 2, 7, 4, requires_grad=True)
    k = torch.randn(1, 1, 7, 4, requires_grad=True)
    v = torch.randn(1, 1, 7, 4, requires_grad=True)
    mask = torch.ones(1, 7, dtype=torch.bool)
    grad_out = torch.randn(1, 2, 7, 4)

    chunked = _CanonicalChunkedAttentionFn.apply(q, k, v, mask, 3, op)
    chunked.backward(grad_out)
    chunked_grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())

    q_ref = q.detach().requires_grad_(True)
    k_ref = k.detach().requires_grad_(True)
    v_ref = v.detach().requires_grad_(True)
    full = NativeAttentionOp().forward_fp32(q_ref, k_ref, v_ref, causal=True, key_padding_mask=mask)
    full.backward(grad_out)

    assert op.calls == [(3, 3), (3, 6), (1, 7), (7, 7)]
    for actual, expected in zip(chunked_grads, (q_ref.grad, k_ref.grad, v_ref.grad)):
        assert torch.equal(actual, expected)
