# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Full Qwen3-8B Dense BI model (WS1 C9 / #275).

Official depth/width/heads/vocab from the C2 manifest. C9 is assembly only:
model-level EXIT is C10 + C11. Silent cuBLAS / cross-profile fallback is a
hard fail. Concat-only NativeKVCacheAttnOp is not used on this path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rl_engine.kernels.gtest.accelerator import (
    candidate_family,
    device_type_for_profile,
    disable_tf32,
)
from rl_engine.kernels.gtest.gradient_adapters import resolve_profile_candidate
from rl_engine.kernels.gtest.operator_specs import OP_SPECS, _load_object
from rl_engine.kernels.ops.canonical_backward import active_session
from rl_engine.kernels.ops.canonical_embedding import canonical_embedding
from rl_engine.kernels.ops.canonical_linear import canonical_linear_fp32
from rl_engine.kernels.ops.canonical_lm_head import (
    canonical_cuda_lm_head_fp32,
    canonical_row_lm_head,
)
from rl_engine.kernels.ops.canonical_rmsnorm import (
    canonical_ascend_rmsnorm,
    canonical_cuda_rmsnorm,
    canonical_row_rmsnorm,
)
from rl_engine.kernels.ops.pytorch.attention.stateful_kv import StatefulKVCache
from rl_engine.testing.ws1_workload import WS1Manifest, load_manifest, weight_snapshot_hash

OFFICIAL_FINGERPRINT = {
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
    "swiglu": True,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "qk_norm": True,
}

NODE_KINDS = (
    "embedding",
    "rms_norm",
    "det_gemm",
    "qk_norm",
    "rope",
    "attention",
    "swiglu",
    "lm_head",
    "logprob",
)
_VJP_NODES = frozenset(
    {
        "final_layernorm",
        "layers.0.input_layernorm",
        "layers.0.q_proj",
        "lm_head",
    }
)


class _CanonicalChunkedAttentionFn(torch.autograd.Function):
    """Run chunked attention forward with a partition-independent backward."""

    @staticmethod
    def forward(ctx, q, k, v, key_padding_mask, chunk_size, op):
        outputs = []
        seq = q.shape[2]
        for start in range(0, seq, int(chunk_size)):
            end = min(start + int(chunk_size), seq)
            outputs.append(
                op.forward_fp32(
                    q[:, :, start:end, :],
                    k[:, :, :end, :],
                    v[:, :, :end, :],
                    causal=True,
                    key_padding_mask=key_padding_mask[:, :end],
                )
            )
        ctx.save_for_backward(q, k, v, key_padding_mask)
        ctx.op = op
        return torch.cat(outputs, dim=2)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, key_padding_mask = ctx.saved_tensors
        with torch.enable_grad():
            q_ref = q.detach().requires_grad_(True)
            k_ref = k.detach().requires_grad_(True)
            v_ref = v.detach().requires_grad_(True)
            out = ctx.op.forward_fp32(
                q_ref,
                k_ref,
                v_ref,
                causal=True,
                key_padding_mask=key_padding_mask,
            )
            dq, dk, dv = torch.autograd.grad(
                out,
                (q_ref, k_ref, v_ref),
                grad_out,
                retain_graph=False,
                create_graph=False,
            )
        return dq, dk, dv, None, None, None


@dataclass(frozen=True)
class Qwen3DenseSpec:
    """Pinned official Qwen3-8B Dense identity. Shrinking any field is forbidden."""

    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    hidden_act: str
    swiglu: bool
    tie_word_embeddings: bool
    attention_bias: bool
    qk_norm: bool
    workload_id: str
    model_id: str
    revision: str
    weight_content_hash: str
    weight_index_file: str
    weight_index_sha256: str
    weight_shards: tuple[tuple[str, str, int], ...]

    @classmethod
    def from_manifest(cls, manifest: WS1Manifest | None = None) -> Qwen3DenseSpec:
        m = manifest if manifest is not None else load_manifest()
        ident = m.model_identity
        fp = ident["config_fingerprint"]
        for key, expected in OFFICIAL_FINGERPRINT.items():
            got = fp.get(key)
            if got != expected:
                raise ValueError(
                    f"C9 forbids architecture shrink/drift: fingerprint {key}={got!r} "
                    f"!= official {expected!r}"
                )
        snap = ident["weight_snapshot"]
        shards = tuple(
            (
                str(item["filename"]),
                str(item["sha256"]),
                int(item["size_bytes"]),
            )
            for item in snap["shards"]
        )
        return cls(
            num_hidden_layers=int(fp["num_hidden_layers"]),
            hidden_size=int(fp["hidden_size"]),
            intermediate_size=int(fp["intermediate_size"]),
            num_attention_heads=int(fp["num_attention_heads"]),
            num_key_value_heads=int(fp["num_key_value_heads"]),
            head_dim=int(fp["head_dim"]),
            vocab_size=int(fp["vocab_size"]),
            max_position_embeddings=int(fp["max_position_embeddings"]),
            rope_theta=float(fp["rope_theta"]),
            rms_norm_eps=float(fp["rms_norm_eps"]),
            hidden_act=str(fp["hidden_act"]),
            swiglu=bool(fp["swiglu"]),
            tie_word_embeddings=bool(fp["tie_word_embeddings"]),
            attention_bias=bool(fp["attention_bias"]),
            qk_norm=bool(fp["qk_norm"]),
            workload_id=m.workload_id,
            model_id=str(ident["model_id"]),
            revision=str(ident["revision"]),
            weight_content_hash=str(snap["content_hash"]),
            weight_index_file=str(snap["index_file"]),
            weight_index_sha256=str(snap["index_sha256"]),
            weight_shards=shards,
        )

    def node_names(self) -> tuple[str, ...]:
        names: list[str] = ["embedding"]
        for index in range(self.num_hidden_layers):
            prefix = f"layers.{index}"
            names.extend(
                [
                    f"{prefix}.input_layernorm",
                    f"{prefix}.q_proj",
                    f"{prefix}.k_proj",
                    f"{prefix}.v_proj",
                    f"{prefix}.q_norm",
                    f"{prefix}.k_norm",
                    f"{prefix}.rope_q",
                    f"{prefix}.rope_k",
                    f"{prefix}.attn",
                    f"{prefix}.o_proj",
                    f"{prefix}.residual_attn",
                    f"{prefix}.post_attention_layernorm",
                    f"{prefix}.gate_proj",
                    f"{prefix}.up_proj",
                    f"{prefix}.swiglu",
                    f"{prefix}.down_proj",
                    f"{prefix}.residual_mlp",
                ]
            )
        names.extend(["final_layernorm", "lm_head", "logprob", "loss"])
        return tuple(names)

    def node_kind(self, node_name: str) -> str:
        if node_name == "embedding":
            return "embedding"
        if node_name in {"final_layernorm"} or node_name.endswith("layernorm"):
            return "rms_norm"
        if node_name.endswith((".q_norm", ".k_norm")):
            return "qk_norm"
        if node_name.endswith((".rope_q", ".rope_k")):
            return "rope"
        if node_name.endswith(".attn"):
            return "attention"
        if node_name.endswith(".swiglu"):
            return "swiglu"
        if node_name.endswith(
            (
                ".q_proj",
                ".k_proj",
                ".v_proj",
                ".o_proj",
                ".gate_proj",
                ".up_proj",
                ".down_proj",
            )
        ):
            return "det_gemm"
        if node_name.endswith((".residual_attn", ".residual_mlp")):
            return "residual_add"
        if node_name == "lm_head":
            return "lm_head"
        if node_name == "logprob":
            return "logprob"
        if node_name == "loss":
            return "masked_loss"
        raise KeyError(f"unknown node {node_name!r}")


@dataclass
class ProfileOps:
    """Resolved per-kind operators for one backend profile."""

    backend_profile: str
    ops: dict[str, Any]
    provenance: dict[str, dict[str, str]]
    observations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, kind: str) -> Any:
        if kind in {"residual_add", "masked_loss"}:
            return None
        if kind not in self.ops:
            raise RuntimeError(
                f"profile {self.backend_profile!r} missing op for node kind {kind!r}"
            )
        return self.ops[kind]

    def observe(self, kind: str, output: torch.Tensor) -> None:
        """Record the candidate object that actually produced a model tensor."""

        op = self.get(kind)
        actual_path = _object_path(op)
        declared = self.provenance[kind]
        expected_path = declared["candidate_path"]
        if actual_path != expected_path:
            raise RuntimeError(
                f"profile {self.backend_profile!r} node {kind!r} executed "
                f"{actual_path!r}, expected {expected_path!r}"
            )
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"profile node {kind!r} did not return a Tensor")
        if declared["status"] != "gold_reference":
            # A gold_reference node is the PyTorch harness path and may run on
            # CPU; a real candidate must land on its profile's accelerator.
            expected_device = device_type_for_profile(self.backend_profile)
            if output.device.type != expected_device:
                raise RuntimeError(
                    f"profile {self.backend_profile!r} node {kind!r} returned "
                    f"non-{expected_device} output on {output.device}"
                )
        previous = self.observations.get(kind)
        count = 1 if previous is None else int(previous["execution_count"]) + 1
        self.observations[kind] = {
            "requested_backend": declared["requested_backend"],
            "actual_backend": declared["actual_backend"],
            "expected_kernel_id": expected_path,
            "observed_kernel_id": actual_path,
            "execution_count": count,
            "output_device": str(output.device),
            "output_dtype": str(output.dtype).removeprefix("torch."),
            "fallback_observed": False,
            "backward_impl": str(getattr(op, "backward_impl", "autograd")),
        }

    def validated_runtime_observations(self) -> dict[str, dict[str, Any]]:
        """Return complete model-level observations, failing closed if any are absent."""

        missing = sorted(set(NODE_KINDS) - set(self.observations))
        if missing:
            raise RuntimeError(
                f"profile {self.backend_profile!r} has no runtime observation for {missing}"
            )
        for kind, observation in self.observations.items():
            if observation["execution_count"] <= 0:
                raise RuntimeError(f"profile node {kind!r} was not executed")
            if observation["observed_kernel_id"] != observation["expected_kernel_id"]:
                raise RuntimeError(f"profile node {kind!r} used an unexpected candidate")
            if observation["fallback_observed"]:
                raise RuntimeError(f"profile node {kind!r} reported a fallback")
        return {kind: dict(value) for kind, value in self.observations.items()}


def load_profile_ops(
    backend_profile: str,
    manifest: WS1Manifest | None = None,
    *,
    allow_pytorch_gold: bool = False,
) -> ProfileOps:
    """Load C2-declared candidates. Missing required nodes are red, not N/A."""

    m = manifest if manifest is not None else load_manifest()
    if backend_profile not in m.backend_profiles:
        raise ValueError(f"unknown backend_profile {backend_profile!r}")
    family = str(m.backend_profiles[backend_profile]["backend_family"])
    ops: dict[str, Any] = {}
    provenance: dict[str, dict[str, str]] = {}
    kind_to_adapter = {
        "embedding": "embedding",
        "rms_norm": "rms_norm",
        "det_gemm": "det_gemm",
        "qk_norm": "qk_norm",
        "rope": "rope",
        "attention": "attention",
        "swiglu": "swiglu",
        "lm_head": "lm_head",
        "logprob": "logp",
    }
    for kind, adapter_name in kind_to_adapter.items():
        if allow_pytorch_gold:
            spec = OP_SPECS[adapter_name]
            ops[kind] = _load_object(spec.gold_path)()
            provenance[kind] = {
                "requested_backend": "pytorch",
                "actual_backend": "pytorch",
                "candidate_path": spec.gold_path,
                "status": "gold_reference",
            }
            continue
        resolved = resolve_profile_candidate(
            _adapter_stub(adapter_name, kind if kind != "logprob" else "logprob"),
            backend_profile,
            m,
        )
        status = str(resolved["status"])
        if status == "missing_required":
            raise RuntimeError(
                f"profile {backend_profile!r} node {kind!r} is missing_required; "
                "C9 treats a missing required node as red on every profile"
            )
        expected = resolved.get("expected_backend_id")
        path = resolved.get("candidate_path")
        if not expected or not path:
            raise RuntimeError(
                f"profile {backend_profile!r} node {kind!r} has no declared candidate path"
            )
        if _family(str(expected)) != family:
            raise RuntimeError(
                f"profile {backend_profile!r} node {kind!r} candidate {expected!r} "
                f"is not family {family!r}"
            )
        ops[kind] = _load_object(str(path))()
        provenance[kind] = {
            "requested_backend": str(expected),
            "actual_backend": str(expected),
            "candidate_path": str(path),
            "status": status,
        }
    return ProfileOps(backend_profile=backend_profile, ops=ops, provenance=provenance)


def _adapter_stub(op_name: str, chain_node: str) -> Any:
    from rl_engine.kernels.gtest.gradient_adapters import get_adapter

    adapter = get_adapter(op_name)
    if adapter.chain_node != chain_node and op_name != "logp":
        # logp adapter chain_node is "logprob"
        pass
    return adapter


def _family(candidate: str) -> str:
    return candidate_family(candidate)


def _object_path(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


class Qwen3DenseWeights:
    """Official-shape parameter bag. Not allocated in ordinary CPU tests."""

    def __init__(self, tensors: dict[str, torch.Tensor], source: str, content_hash: str):
        self.tensors = tensors
        self.source = source
        self.content_hash = content_hash

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    @classmethod
    def synthetic(
        cls,
        spec: Qwen3DenseSpec,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        seed: int,
    ) -> Qwen3DenseWeights:
        """Official shapes, seeded values. Valid for C9 wiring, not C10/C11 EXIT."""

        dev = torch.device(device)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        tensors: dict[str, torch.Tensor] = {}

        def randn(name: str, shape: tuple[int, ...]) -> None:
            cpu = torch.randn(shape, generator=gen, dtype=torch.float32)
            tensors[name] = cpu.to(device=dev, dtype=dtype)

        def ones(name: str, shape: tuple[int, ...]) -> None:
            tensors[name] = torch.ones(shape, device=dev, dtype=dtype)

        randn("embed_tokens.weight", (spec.vocab_size, spec.hidden_size))
        for index in range(spec.num_hidden_layers):
            p = f"layers.{index}"
            ones(f"{p}.input_layernorm.weight", (spec.hidden_size,))
            randn(f"{p}.self_attn.q_proj.weight", (spec.hidden_size, spec.hidden_size))
            randn(
                f"{p}.self_attn.k_proj.weight",
                (spec.num_key_value_heads * spec.head_dim, spec.hidden_size),
            )
            randn(
                f"{p}.self_attn.v_proj.weight",
                (spec.num_key_value_heads * spec.head_dim, spec.hidden_size),
            )
            randn(f"{p}.self_attn.o_proj.weight", (spec.hidden_size, spec.hidden_size))
            ones(f"{p}.self_attn.q_norm.weight", (spec.head_dim,))
            ones(f"{p}.self_attn.k_norm.weight", (spec.head_dim,))
            ones(f"{p}.post_attention_layernorm.weight", (spec.hidden_size,))
            randn(f"{p}.mlp.gate_proj.weight", (spec.intermediate_size, spec.hidden_size))
            randn(f"{p}.mlp.up_proj.weight", (spec.intermediate_size, spec.hidden_size))
            randn(f"{p}.mlp.down_proj.weight", (spec.hidden_size, spec.intermediate_size))
        ones("norm.weight", (spec.hidden_size,))
        randn("lm_head.weight", (spec.vocab_size, spec.hidden_size))
        return cls(tensors, source="synthetic_official_shape", content_hash="synthetic")

    @classmethod
    def from_hf(
        cls,
        spec: Qwen3DenseSpec,
        source: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Qwen3DenseWeights:
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required to load pinned Qwen3-8B weights") from exc

        path = Path(source)
        if not path.is_dir():
            raise FileNotFoundError(f"weight path does not exist: {path}")
        files = verify_hf_weight_snapshot(spec, path)

        raw: dict[str, torch.Tensor] = {}
        for shard in files:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    raw[key] = handle.get_tensor(key)

        mapped = _map_hf_keys(raw)
        required = _required_weight_keys(spec)
        missing = [key for key in required if key not in mapped]
        if missing:
            raise RuntimeError(f"HF snapshot missing required tensors (first 8): {missing[:8]}")
        dev = torch.device(device)
        tensors = {key: mapped[key].to(device=dev, dtype=dtype) for key in required}
        return cls(tensors, source=f"hf:{path}", content_hash=spec.weight_content_hash)


class Qwen3DenseWeightsOffloaded(Qwen3DenseWeights):
    """CPU-resident weights paged onto the accelerator per access.

    The C10 FP32 reference on 64 GB HBM hosts uses this: the FP32 weights
    (~32 GB) plus their FP32 gradients (~32 GB) plus activations cannot all
    be resident, but the reference forward touches one weight at a time.
    Each ``__getitem__`` issues an exact device copy; the autograd graph
    keeps each copy alive until its VJP consumes it, so the peak HBM is the
    sum of one copy per use (~36 GB) and the FP32 gradients accumulate on
    the CPU-resident leaves instead of on the accelerator. Copies are
    exact, so the forward and backward numerics are bitwise identical to
    the resident-FP32 model.
    """

    def __init__(self, weights: Qwen3DenseWeights, device: torch.device | str):
        super().__init__(weights.tensors, weights.source, weights.content_hash)
        self._device = torch.device(device)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.tensors[key].to(self._device)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hf_weight_snapshot(spec: Qwen3DenseSpec, source: str | Path) -> list[Path]:
    """Verify the pinned index and every shard before loading any tensor."""

    path = Path(source)
    if not path.is_dir():
        raise FileNotFoundError(f"weight snapshot directory does not exist: {path}")

    index_path = path / spec.weight_index_file
    if not index_path.is_file():
        raise FileNotFoundError(f"pinned weight index is missing: {index_path}")
    actual_index_hash = _sha256_file(index_path)
    if actual_index_hash != spec.weight_index_sha256:
        raise RuntimeError(
            f"weight index SHA-256 mismatch: {actual_index_hash} " f"!= {spec.weight_index_sha256}"
        )

    observed_records: list[dict[str, Any]] = []
    files: list[Path] = []
    for filename, expected_hash, expected_size in spec.weight_shards:
        shard = path / filename
        if not shard.is_file():
            raise FileNotFoundError(f"pinned weight shard is missing: {shard}")
        actual_size = shard.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"weight shard size mismatch for {filename}: " f"{actual_size} != {expected_size}"
            )
        actual_hash = _sha256_file(shard)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"weight shard SHA-256 mismatch for {filename}: "
                f"{actual_hash} != {expected_hash}"
            )
        observed_records.append(
            {"filename": filename, "sha256": actual_hash, "size_bytes": actual_size}
        )
        files.append(shard)

    observed_content_hash = weight_snapshot_hash(observed_records)
    if observed_content_hash != spec.weight_content_hash:
        raise RuntimeError(
            f"weight snapshot content hash mismatch: {observed_content_hash} "
            f"!= {spec.weight_content_hash}"
        )
    return files


def _required_weight_keys(spec: Qwen3DenseSpec) -> list[str]:
    keys = ["embed_tokens.weight", "norm.weight", "lm_head.weight"]
    for index in range(spec.num_hidden_layers):
        p = f"layers.{index}"
        keys.extend(
            [
                f"{p}.input_layernorm.weight",
                f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.o_proj.weight",
                f"{p}.self_attn.q_norm.weight",
                f"{p}.self_attn.k_norm.weight",
                f"{p}.post_attention_layernorm.weight",
                f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight",
            ]
        )
    return keys


def _map_hf_keys(raw: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        name = key
        if name.startswith("model."):
            name = name[len("model.") :]
        mapped[name] = value
    if "lm_head.weight" not in mapped and "embed_tokens.weight" in mapped:
        # Official Qwen3-8B is untied; refuse to silently tie.
        raise RuntimeError("HF snapshot has no lm_head.weight; refusing to tie embeddings")
    return mapped


class Qwen3DenseBIModel:
    """Full official Qwen3-8B Dense topology on the in-repo BI operator stack."""

    def __init__(
        self,
        spec: Qwen3DenseSpec,
        weights: Qwen3DenseWeights,
        profile_ops: ProfileOps,
        *,
        execution_dtype: torch.dtype = torch.bfloat16,
    ):
        self.spec = spec
        self.weights = weights
        self.profile_ops = profile_ops
        self.execution_dtype = execution_dtype
        self._last_node_outputs: dict[str, torch.Tensor] = {}
        self._capture_nodes = False
        self._vjp_enabled = False
        self._vjp_inputs: dict[str, list[dict[str, Any]]] = {}
        self._vjp_grads: dict[str, dict[int, torch.Tensor]] = {}
        self._vjp_hooks: list[Any] = []
        disable_tf32(device_type_for_profile(self.profile_ops.backend_profile))

    @property
    def backend_profile(self) -> str:
        return self.profile_ops.backend_profile

    def node_names(self) -> tuple[str, ...]:
        return self.spec.node_names()

    def allocate_cache(self, batch: int, max_seq_len: int, device: torch.device) -> StatefulKVCache:
        return StatefulKVCache.allocate(
            n_layers=self.spec.num_hidden_layers,
            batch=batch,
            n_kv_heads=self.spec.num_key_value_heads,
            max_seq_len=max_seq_len,
            head_dim=self.spec.head_dim,
            dtype=self.execution_dtype,
            device=device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        target_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        kv_cache: StatefulKVCache | None = None,
        capture_nodes: bool = False,
        segment_lengths: Sequence[int] | None = None,
        logical_keys: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forcing prefill (or one decode step when ``input_ids`` is length 1)."""

        self._capture_nodes = capture_nodes
        self._last_node_outputs = {}
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, S], got {tuple(input_ids.shape)}")
        batch, seq = input_ids.shape
        device = input_ids.device
        if logical_keys is not None:
            if logical_keys.shape[:2] != (batch, seq) or logical_keys.shape[-1] != 2:
                raise ValueError("logical_keys must have shape [B, S, 2]")
            logical_keys = logical_keys.to(device=device, dtype=torch.long)
        if active_session() is not None and logical_keys is None:
            raise RuntimeError("active canonical backward requires logical_keys")
        self._current_logical_keys = logical_keys
        if position_ids is None:
            position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, seq)
        if attention_mask is None:
            attention_mask = torch.ones((batch, seq), device=device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
        if segment_lengths is not None:
            if batch != 1:
                raise ValueError("packed segment_lengths require B=1")
            if int(sum(int(x) for x in segment_lengths)) != seq:
                raise ValueError(f"segment_lengths sum {sum(segment_lengths)} != seq {seq}")

        hidden = self._embed(input_ids)
        hidden = hidden.float()
        for layer in range(self.spec.num_hidden_layers):
            hidden = self._decoder_layer(
                hidden,
                position_ids,
                attention_mask,
                layer=layer,
                kv_cache=kv_cache,
                segment_lengths=segment_lengths,
            )
        hidden = self._rms(
            hidden.to(dtype=self.execution_dtype),
            self.weights["norm.weight"],
            node="final_layernorm",
        )
        lm_head_op = self.profile_ops.get("lm_head")
        keys = getattr(self, "_current_logical_keys", None)
        lm_family = self.profile_ops.provenance["lm_head"]["actual_backend"]
        if (
            torch.is_grad_enabled()
            and active_session() is not None
            and keys is not None
            and lm_family.startswith("cuda")
        ):
            score_logits = canonical_cuda_lm_head_fp32(
                hidden,
                self.weights["lm_head.weight"],
                keys.reshape(-1, 2),
            )
        elif (
            torch.is_grad_enabled()
            and active_session() is not None
            and keys is not None
            and lm_family in ("triton", "ascend")
        ):
            score_logits = canonical_row_lm_head(
                hidden,
                self.weights["lm_head.weight"],
                keys.reshape(-1, 2),
                forward_op=lm_head_op.forward_fp32,
                matmul_op=self.profile_ops.get("det_gemm").forward_accum_fp32,
                family=lm_family,
            )
        else:
            score_logits = lm_head_op.forward_fp32(
                hidden, self.weights["lm_head.weight"], bias=None
            )
        logits = score_logits.to(dtype=self.execution_dtype)
        self.profile_ops.observe("lm_head", logits)
        self._maybe_save_vjp(
            "lm_head",
            {"hidden": hidden.detach(), "weight": self.weights["lm_head.weight"].detach()},
            score_logits,
        )
        logits = self._record("lm_head", logits)
        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "score_logits": score_logits,
            "hidden": hidden,
        }
        if target_ids is not None:
            loss_logits = score_logits
            score_targets = target_ids
            score_mask = loss_mask
            if target_ids.shape == input_ids.shape:
                if input_ids.shape[1] < 2:
                    raise ValueError("causal selected-logprob requires sequence length >= 2")
                loss_logits = score_logits[:, :-1]
                score_targets = target_ids[:, 1:]
                score_mask = None if loss_mask is None else loss_mask[:, 1:]
            logp = self._record("logprob", self._selected_logp(loss_logits, score_targets))
            result["selected_logp"] = logp
            if score_mask is not None:
                result["loss"] = self._masked_loss(logp, score_mask)
        return result

    def forward_chunked_training(
        self,
        input_ids: torch.Tensor,
        *,
        chunk_size: int,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        logical_keys: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Differentiable chunked-prefill with a canonical attention backward."""

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, S], got {tuple(input_ids.shape)}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        batch, seq = input_ids.shape
        if attention_mask.shape != (batch, seq):
            raise ValueError("attention_mask must match input_ids")
        if position_ids.shape != (batch, seq):
            raise ValueError("position_ids must match input_ids")
        if logical_keys.shape != (batch, seq, 2):
            raise ValueError("logical_keys must have shape [B, S, 2]")

        self._capture_nodes = False
        self._last_node_outputs = {}
        chunks = tuple(
            (start, min(start + int(chunk_size), seq)) for start in range(0, seq, int(chunk_size))
        )
        hidden_parts: list[torch.Tensor] = []
        for start, end in chunks:
            self._current_logical_keys = logical_keys[:, start:end]
            hidden_parts.append(self._embed(input_ids[:, start:end]).float())

        attention_op = self.profile_ops.get("attention")
        for layer in range(self.spec.num_hidden_layers):
            prefix = f"layers.{layer}"
            q_parts: list[torch.Tensor] = []
            k_parts: list[torch.Tensor] = []
            v_parts: list[torch.Tensor] = []
            for hidden, (start, end) in zip(hidden_parts, chunks, strict=True):
                self._current_logical_keys = logical_keys[:, start:end]
                normed = self._rms(
                    hidden.to(dtype=self.execution_dtype),
                    self.weights[f"{prefix}.input_layernorm.weight"],
                    node=f"{prefix}.input_layernorm",
                )
                q = self._linear(
                    normed,
                    self.weights[f"{prefix}.self_attn.q_proj.weight"],
                    node=f"{prefix}.q_proj",
                )
                k = self._linear(
                    normed,
                    self.weights[f"{prefix}.self_attn.k_proj.weight"],
                    node=f"{prefix}.k_proj",
                )
                v = self._linear(
                    normed,
                    self.weights[f"{prefix}.self_attn.v_proj.weight"],
                    node=f"{prefix}.v_proj",
                )
                q = self._to_heads(q, self.spec.num_attention_heads)
                k = self._to_heads(k, self.spec.num_key_value_heads)
                v = self._to_heads(v, self.spec.num_key_value_heads)
                q = self._qk_norm(
                    q,
                    self.weights[f"{prefix}.self_attn.q_norm.weight"],
                    node=f"{prefix}.q_norm",
                )
                k = self._qk_norm(
                    k,
                    self.weights[f"{prefix}.self_attn.k_norm.weight"],
                    node=f"{prefix}.k_norm",
                )
                q_parts.append(self._rope(q, position_ids[:, start:end], node=f"{prefix}.rope_q"))
                k_parts.append(self._rope(k, position_ids[:, start:end], node=f"{prefix}.rope_k"))
                v_parts.append(v)

            attn_all = _CanonicalChunkedAttentionFn.apply(
                torch.cat(q_parts, dim=2),
                torch.cat(k_parts, dim=2),
                torch.cat(v_parts, dim=2),
                attention_mask,
                int(chunk_size),
                attention_op,
            )
            attn_public = attn_all.to(dtype=self.execution_dtype)
            self.profile_ops.observe("attention", attn_public)
            self._record(f"{prefix}.attn", attn_public)

            next_hidden_parts: list[torch.Tensor] = []
            for hidden, (start, end) in zip(hidden_parts, chunks, strict=True):
                self._current_logical_keys = logical_keys[:, start:end]
                attn = attn_all[:, :, start:end, :]
                attn_merged = (
                    attn.transpose(1, 2)
                    .contiguous()
                    .reshape(hidden.shape[0], end - start, self.spec.hidden_size)
                )
                projected = self._linear(
                    attn_merged,
                    self.weights[f"{prefix}.self_attn.o_proj.weight"],
                    node=f"{prefix}.o_proj",
                )
                hidden = self._record(
                    f"{prefix}.residual_attn", hidden + projected.to(dtype=hidden.dtype)
                )
                mlp_in = self._rms(
                    hidden.to(dtype=self.execution_dtype),
                    self.weights[f"{prefix}.post_attention_layernorm.weight"],
                    node=f"{prefix}.post_attention_layernorm",
                )
                gate = self._linear(
                    mlp_in,
                    self.weights[f"{prefix}.mlp.gate_proj.weight"],
                    node=f"{prefix}.gate_proj",
                    internal_fp32=True,
                )
                up = self._linear(
                    mlp_in,
                    self.weights[f"{prefix}.mlp.up_proj.weight"],
                    node=f"{prefix}.up_proj",
                    internal_fp32=True,
                )
                swiglu_op = self.profile_ops.get("swiglu")
                if self.execution_dtype == torch.bfloat16:
                    swiglu = swiglu_op.forward_fp32(gate, up)
                    swiglu_public = swiglu.to(dtype=self.execution_dtype)
                else:
                    swiglu = swiglu_op.forward(gate, up)
                    swiglu_public = swiglu
                self.profile_ops.observe("swiglu", swiglu_public)
                self._record(f"{prefix}.swiglu", swiglu_public)
                down = self._linear(
                    swiglu,
                    self.weights[f"{prefix}.mlp.down_proj.weight"],
                    node=f"{prefix}.down_proj",
                )
                next_hidden_parts.append(
                    self._record(
                        f"{prefix}.residual_mlp",
                        hidden + down.to(dtype=hidden.dtype),
                    )
                )
            hidden_parts = next_hidden_parts

        score_parts: list[torch.Tensor] = []
        logits_parts: list[torch.Tensor] = []
        hidden_outputs: list[torch.Tensor] = []
        lm_head_op = self.profile_ops.get("lm_head")
        lm_family = self.profile_ops.provenance["lm_head"]["actual_backend"]
        for hidden, (start, end) in zip(hidden_parts, chunks, strict=True):
            keys = logical_keys[:, start:end]
            self._current_logical_keys = keys
            final_hidden = self._rms(
                hidden.to(dtype=self.execution_dtype),
                self.weights["norm.weight"],
                node="final_layernorm",
            )
            if active_session() is not None and lm_family.startswith("cuda"):
                score_logits = canonical_cuda_lm_head_fp32(
                    final_hidden,
                    self.weights["lm_head.weight"],
                    keys.reshape(-1, 2),
                )
            elif active_session() is not None and lm_family in ("triton", "ascend"):
                score_logits = canonical_row_lm_head(
                    final_hidden,
                    self.weights["lm_head.weight"],
                    keys.reshape(-1, 2),
                    forward_op=lm_head_op.forward_fp32,
                    matmul_op=self.profile_ops.get("det_gemm").forward_accum_fp32,
                    family=lm_family,
                )
            else:
                score_logits = lm_head_op.forward_fp32(
                    final_hidden, self.weights["lm_head.weight"], bias=None
                )
            logits = score_logits.to(dtype=self.execution_dtype)
            self.profile_ops.observe("lm_head", logits)
            hidden_outputs.append(final_hidden)
            score_parts.append(score_logits)
            logits_parts.append(logits)
        return {
            "logits": torch.cat(logits_parts, dim=1),
            "score_logits": torch.cat(score_parts, dim=1),
            "hidden": torch.cat(hidden_outputs, dim=1),
        }

    def decode_step(
        self,
        input_ids: torch.Tensor,
        kv_cache: StatefulKVCache,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.shape[1] != 1:
            raise ValueError("decode_step expects a single new token [B, 1]")
        return self.forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

    def selected_logprobs(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forcing selected logprob of tokens[1:] from logits[:-1]."""

        outputs = self.forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.get("score_logits", outputs["logits"])[:, :-1, :]
        targets = input_ids[:, 1:]
        logp = self._selected_logp(logits, targets)
        if loss_mask is None:
            return logp
        return logp.masked_fill(~loss_mask[:, 1:].to(dtype=torch.bool), 0.0)

    def captured_node_outputs(self) -> dict[str, torch.Tensor]:
        return dict(self._last_node_outputs)

    def begin_vjp_capture(self) -> None:
        self.end_vjp_capture()
        self._vjp_enabled = True
        self._vjp_inputs = {}
        self._vjp_grads = {}

    def end_vjp_capture(self) -> None:
        for hook in self._vjp_hooks:
            hook.remove()
        self._vjp_hooks = []
        self._vjp_enabled = False

    def take_vjp_captures(
        self,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[int, torch.Tensor]]]:
        inputs = {key: list(value) for key, value in self._vjp_inputs.items()}
        grads = {key: dict(value) for key, value in self._vjp_grads.items()}
        self.end_vjp_capture()
        return inputs, grads

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        op = self.profile_ops.get("embedding")
        weight = self.weights["embed_tokens.weight"]
        keys = getattr(self, "_current_logical_keys", None)
        if torch.is_grad_enabled() and active_session() is not None and keys is not None:
            family = self.profile_ops.provenance["embedding"]["actual_backend"]
            out = canonical_embedding(
                input_ids,
                weight,
                keys,
                forward_op=op.forward,
                family=family,
            )
        else:
            out = op.forward(input_ids, weight)
        self.profile_ops.observe("embedding", out)
        return self._record("embedding", out)

    def _decoder_layer(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        layer: int,
        kv_cache: StatefulKVCache | None,
        segment_lengths: Sequence[int] | None = None,
    ) -> torch.Tensor:
        prefix = f"layers.{layer}"
        normed = self._rms(
            hidden.to(dtype=self.execution_dtype),
            self.weights[f"{prefix}.input_layernorm.weight"],
            node=f"{prefix}.input_layernorm",
        )
        q = self._linear(
            normed,
            self.weights[f"{prefix}.self_attn.q_proj.weight"],
            node=f"{prefix}.q_proj",
        )
        k = self._linear(
            normed,
            self.weights[f"{prefix}.self_attn.k_proj.weight"],
            node=f"{prefix}.k_proj",
        )
        v = self._linear(
            normed,
            self.weights[f"{prefix}.self_attn.v_proj.weight"],
            node=f"{prefix}.v_proj",
        )
        q = self._to_heads(q, self.spec.num_attention_heads)
        k = self._to_heads(k, self.spec.num_key_value_heads)
        v = self._to_heads(v, self.spec.num_key_value_heads)
        q = self._qk_norm(
            q,
            self.weights[f"{prefix}.self_attn.q_norm.weight"],
            node=f"{prefix}.q_norm",
        )
        k = self._qk_norm(
            k,
            self.weights[f"{prefix}.self_attn.k_norm.weight"],
            node=f"{prefix}.k_norm",
        )
        q = self._rope(q, position_ids, node=f"{prefix}.rope_q")
        k = self._rope(k, position_ids, node=f"{prefix}.rope_k")
        key_mask = self._key_padding_mask(attention_mask, kv_cache, layer=layer, new_len=q.shape[2])
        if kv_cache is not None:
            kv_cache.write(k, v, layer=layer, valid_mask=attention_mask)
            k, v, _length = kv_cache.read(layer=layer)
        attn = self._attn(
            q,
            k,
            v,
            key_mask,
            node=f"{prefix}.attn",
            segment_lengths=segment_lengths,
            output_fp32=self.execution_dtype == torch.bfloat16,
        )
        attn_merged = (
            attn.transpose(1, 2)
            .contiguous()
            .reshape(hidden.shape[0], q.shape[2], self.spec.hidden_size)
        )
        projected = self._linear(
            attn_merged,
            self.weights[f"{prefix}.self_attn.o_proj.weight"],
            node=f"{prefix}.o_proj",
        )
        hidden = self._record(f"{prefix}.residual_attn", hidden + projected.to(dtype=hidden.dtype))
        mlp_in = self._rms(
            hidden.to(dtype=self.execution_dtype),
            self.weights[f"{prefix}.post_attention_layernorm.weight"],
            node=f"{prefix}.post_attention_layernorm",
        )
        gate = self._linear(
            mlp_in,
            self.weights[f"{prefix}.mlp.gate_proj.weight"],
            node=f"{prefix}.gate_proj",
            internal_fp32=True,
        )
        up = self._linear(
            mlp_in,
            self.weights[f"{prefix}.mlp.up_proj.weight"],
            node=f"{prefix}.up_proj",
            internal_fp32=True,
        )
        swiglu_op = self.profile_ops.get("swiglu")
        if self.execution_dtype == torch.bfloat16:
            swiglu = swiglu_op.forward_fp32(gate, up)
            swiglu_public = swiglu.to(dtype=self.execution_dtype)
        else:
            swiglu = swiglu_op.forward(gate, up)
            swiglu_public = swiglu
        self.profile_ops.observe("swiglu", swiglu_public)
        self._record(f"{prefix}.swiglu", swiglu_public)
        down = self._linear(
            swiglu,
            self.weights[f"{prefix}.mlp.down_proj.weight"],
            node=f"{prefix}.down_proj",
        )
        return self._record(f"{prefix}.residual_mlp", hidden + down.to(dtype=hidden.dtype))

    def _linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        node: str,
        internal_fp32: bool = False,
    ) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        # The BF16 candidate uses a fixed rowwise FP32 reduction for every
        # projection. Public node values remain BF16; selected composite
        # edges may retain the FP32 accumulator internally.
        op = self.profile_ops.get("det_gemm")
        if self.execution_dtype == torch.bfloat16:
            keys = getattr(self, "_current_logical_keys", None)
            if torch.is_grad_enabled() and active_session() is not None and keys is not None:
                family = self.profile_ops.provenance["det_gemm"]["actual_backend"]
                out = canonical_linear_fp32(
                    flat,
                    weight,
                    keys.reshape(-1, 2),
                    parameter_id=node,
                    family=family,
                )
            else:
                out = op.forward_accum_fp32(flat, weight.t().contiguous())
            shaped = out.reshape(*x.shape[:-1], weight.shape[0])
            public = shaped.to(dtype=self.execution_dtype)
        else:
            out = op(flat, weight.t().contiguous())
            shaped = out.reshape(*x.shape[:-1], weight.shape[0])
            public = shaped
        self.profile_ops.observe("det_gemm", public)
        self._maybe_save_vjp(node, {"x": x.detach(), "weight": weight.detach()}, shaped)
        self._record(node, public)
        return shaped if internal_fp32 and self.execution_dtype == torch.bfloat16 else public

    def _rms(self, x: torch.Tensor, weight: torch.Tensor, *, node: str) -> torch.Tensor:
        op = self.profile_ops.get("rms_norm")
        keys = getattr(self, "_current_logical_keys", None)
        family = self.profile_ops.provenance["rms_norm"]["actual_backend"]
        if torch.is_grad_enabled() and active_session() is not None and keys is not None:
            hidden = x.shape[-1]
            x_rows = x.contiguous().view(-1, hidden)
            row_keys = keys.reshape(-1, 2)
            if family == "cuda":
                out = canonical_cuda_rmsnorm(
                    x_rows,
                    weight.contiguous(),
                    eps=self.spec.rms_norm_eps,
                    logical_keys=row_keys,
                    parameter_id=node,
                ).view_as(x)
            elif family == "triton":
                out = canonical_row_rmsnorm(
                    x_rows,
                    weight.contiguous(),
                    eps=self.spec.rms_norm_eps,
                    logical_keys=row_keys,
                    parameter_id=node,
                    forward_op=op.forward,
                ).view_as(x)
            elif family == "ascend":
                out = canonical_ascend_rmsnorm(
                    x_rows,
                    weight.contiguous(),
                    eps=self.spec.rms_norm_eps,
                    logical_keys=row_keys,
                    parameter_id=node,
                ).view_as(x)
            else:
                out = op.forward(x, weight, eps=self.spec.rms_norm_eps)
        else:
            out = op.forward(x, weight, eps=self.spec.rms_norm_eps)
        self.profile_ops.observe("rms_norm", out)
        self._maybe_save_vjp(
            node,
            {
                "x": x.detach(),
                "weight": weight.detach(),
                "eps": float(self.spec.rms_norm_eps),
            },
            out,
        )
        return self._record(node, out)

    def _qk_norm(self, x: torch.Tensor, weight: torch.Tensor, *, node: str) -> torch.Tensor:
        # x: [B, H, S, D] -> RMS over D
        batch, heads, seq, dim = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(batch, seq * heads, dim)
        op = self.profile_ops.get("qk_norm")
        keys = getattr(self, "_current_logical_keys", None)
        family = self.profile_ops.provenance["qk_norm"]["actual_backend"]
        if torch.is_grad_enabled() and active_session() is not None and keys is not None:
            head_keys = keys[:, :, None, :].expand(batch, seq, heads, 2).reshape(-1, 2)
            flat_rows = flat.contiguous().view(-1, dim)
            if family == "cuda":
                out = canonical_cuda_rmsnorm(
                    flat_rows,
                    weight.contiguous(),
                    eps=self.spec.rms_norm_eps,
                    logical_keys=head_keys,
                    parameter_id=node,
                ).view_as(flat)
            elif family == "triton":
                out = canonical_row_rmsnorm(
                    flat_rows,
                    weight.contiguous(),
                    eps=self.spec.rms_norm_eps,
                    logical_keys=head_keys,
                    parameter_id=node,
                    forward_op=op.forward,
                ).view_as(flat)
            elif family == "ascend":
                out = canonical_ascend_rmsnorm(
                    flat_rows,
                    weight.contiguous(),
                    eps=self.spec.rms_norm_eps,
                    logical_keys=head_keys,
                    parameter_id=node,
                ).view_as(flat)
            else:
                out = op.forward(flat, weight, eps=self.spec.rms_norm_eps)
        else:
            out = op.forward(flat, weight, eps=self.spec.rms_norm_eps)
        self.profile_ops.observe("qk_norm", out)
        return self._record(
            node, out.reshape(batch, seq, heads, dim).permute(0, 2, 1, 3).contiguous()
        )

    def _rope(self, x: torch.Tensor, position_ids: torch.Tensor, *, node: str) -> torch.Tensor:
        op = self.profile_ops.get("rope")
        out = op.forward(x, position_ids, theta=self.spec.rope_theta)
        self.profile_ops.observe("rope", out)
        return self._record(node, out)

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor | None,
        *,
        node: str,
        segment_lengths: Sequence[int] | None = None,
        output_fp32: bool = False,
    ) -> torch.Tensor:
        op = self.profile_ops.get("attention")
        forward = op.forward_fp32 if output_fp32 else op.forward
        if segment_lengths:
            pieces: list[torch.Tensor] = []
            start = 0
            for length in segment_lengths:
                end = start + int(length)
                mask_s = key_mask[:, start:end] if key_mask is not None else None
                pieces.append(
                    forward(
                        q[:, :, start:end, :],
                        k[:, :, start:end, :],
                        v[:, :, start:end, :],
                        causal=True,
                        key_padding_mask=mask_s,
                    )
                )
                start = end
            out = torch.cat(pieces, dim=2)
        else:
            out = forward(q, k, v, causal=True, key_padding_mask=key_mask)
        public = out.to(dtype=self.execution_dtype) if output_fp32 else out
        self.profile_ops.observe("attention", public)
        self._record(node, public)
        return out if output_fp32 else public

    def _selected_logp(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        op = self.profile_ops.get("logprob")
        if hasattr(op, "forward"):
            out = op.forward(logits, token_ids)
        else:
            out = op(logits, token_ids)
        self.profile_ops.observe("logprob", out)
        return out

    def _masked_loss(self, logp: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        mask = loss_mask.to(dtype=torch.bool, device=logp.device)
        if mask.shape != logp.shape:
            raise ValueError(f"loss_mask {tuple(mask.shape)} != logp {tuple(logp.shape)}")
        active = mask.sum().clamp_min(1)
        loss = -(logp.float().masked_fill(~mask, 0.0).sum() / active.float())
        return self._record("loss", loss)

    def _to_heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        batch, seq, hidden = x.shape
        return x.reshape(batch, seq, n_heads, self.spec.head_dim).transpose(1, 2).contiguous()

    def _key_padding_mask(
        self,
        attention_mask: torch.Tensor,
        kv_cache: StatefulKVCache | None,
        *,
        layer: int,
        new_len: int,
    ) -> torch.Tensor:
        if kv_cache is None:
            return attention_mask
        _k, _v, length = kv_cache.read(layer=layer)
        # This is called before the current K/V append, so ``length`` is the
        # complete cached prefix. Callers pass only the new-token mask.
        if attention_mask.shape[1] != new_len:
            raise ValueError(
                f"new-token attention mask width {attention_mask.shape[1]} " f"!= new_len {new_len}"
            )
        prefix = length
        prefix_mask = kv_cache.read_valid_mask(layer=layer)
        if prefix_mask.shape != (attention_mask.shape[0], prefix):
            raise RuntimeError(
                f"cache validity mask shape {tuple(prefix_mask.shape)} "
                f"!= expected {(attention_mask.shape[0], prefix)}"
            )
        return torch.cat([prefix_mask, attention_mask], dim=1)

    def _record(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if self._capture_nodes:
            self._last_node_outputs[name] = value.detach()
        return value

    def _maybe_save_vjp(
        self,
        node: str,
        inputs: dict[str, Any],
        output: torch.Tensor,
    ) -> None:
        if not self._vjp_enabled or node not in _VJP_NODES:
            return
        index = len(self._vjp_inputs.setdefault(node, []))
        self._vjp_inputs[node].append(inputs)
        if not output.requires_grad:
            return

        def _hook(grad: torch.Tensor, captured_node: str = node, slot: int = index) -> None:
            self._vjp_grads.setdefault(captured_node, {})[slot] = grad.detach()

        self._vjp_hooks.append(output.register_hook(_hook))


def iter_parameter_tensors(
    weights: Qwen3DenseWeights,
) -> Iterator[tuple[str, torch.Tensor]]:
    yield from weights.tensors.items()


__all__ = [
    "OFFICIAL_FINGERPRINT",
    "NODE_KINDS",
    "ProfileOps",
    "Qwen3DenseBIModel",
    "Qwen3DenseSpec",
    "Qwen3DenseWeights",
    "iter_parameter_tensors",
    "load_profile_ops",
    "verify_hf_weight_snapshot",
]
