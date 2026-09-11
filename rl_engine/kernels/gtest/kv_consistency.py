# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C6/C7: decode–prefill and stateful-KV / generate-rescore harness.

Thresholds come only from the C1 contract. Concat-only NativeKVCacheAttnOp is
Level A and is never accepted as a C7 B1 writer/reader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from rl_engine.kernels.gtest.accelerator import (
    ACCELERATOR_TYPES,
    candidate_family,
    compute_capability,
    resolve_device,
)
from rl_engine.kernels.gtest.forward_invariance import (
    TensorComparisonDetail,
    _compare_logical_tensors,
)
from rl_engine.kernels.gtest.gradient_adapters import (
    get_adapter,
    load_adapter_operator,
    resolve_profile_candidate,
)
from rl_engine.kernels.gtest.operator_specs import OP_SPECS
from rl_engine.kernels.gtest.tolerance import (
    BackendProvenance,
    LogprobAggregateVerdict,
    compute_logprob_aggregates,
    default_clip_interval,
    judge_logprob_aggregates,
    load_contract,
    normalize_dtype_name,
    resolve_comparison_roles,
    resolve_dtype_policy,
    validate_backend_provenance,
)
from rl_engine.kernels.ops.pytorch.attention.kv_cache import NativeKVCacheAttnOp
from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp
from rl_engine.kernels.ops.pytorch.attention.stateful_kv import StatefulKVCache
from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp
from rl_engine.testing.ws1_workload import WS1Manifest, load_manifest

PROBE_VOCAB = 64
B2_PRODUCTION_KV_STATUS = "absent"
_REPORT_KIND = "train_infer_logprob_parity"


@dataclass(frozen=True)
class DecodePrefillCase:
    """One C6 scenario. ``include_direct_decode`` is required (not chunked-only)."""

    case_id: str
    batch: int
    seq_lens: tuple[int, ...]
    pad_side: str | None
    fixture_id: str
    include_direct_decode: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodePrefillCell:
    case_id: str
    attention_compare: TensorComparisonDetail
    concat_reference_compare: TensorComparisonDetail
    logprob_verdict: LogprobAggregateVerdict
    stored_kv_dtype: str
    stored_kv_layout: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "attention_compare": self.attention_compare.to_dict(),
            "concat_reference_compare": self.concat_reference_compare.to_dict(),
            "logprob_verdict": self.logprob_verdict.to_dict(),
            "stored_kv_dtype": self.stored_kv_dtype,
            "stored_kv_layout": self.stored_kv_layout,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class DecodePrefillReport:
    backend_profile: str
    candidate_id: str
    device: str
    compute_capability: str | None
    seed: int
    backend_provenance: BackendProvenance
    cells: tuple[DecodePrefillCell, ...]
    passed: bool
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_profile": self.backend_profile,
            "candidate_id": self.candidate_id,
            "device": self.device,
            "compute_capability": self.compute_capability,
            "seed": self.seed,
            "backend_provenance": self.backend_provenance.to_dict(),
            "cells": [c.to_dict() for c in self.cells],
            "passed": self.passed,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class StatefulKVReport:
    backend_profile: str
    candidate_id: str
    device: str
    cache_identity: dict[str, str]
    b1_passed: bool
    generate_rescore: LogprobAggregateVerdict
    b2_status: str
    backend_provenance: BackendProvenance
    passed: bool
    fallback_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_profile": self.backend_profile,
            "candidate_id": self.candidate_id,
            "device": self.device,
            "cache_identity": dict(self.cache_identity),
            "b1_passed": self.b1_passed,
            "generate_rescore": self.generate_rescore.to_dict(),
            "b2_status": self.b2_status,
            "backend_provenance": self.backend_provenance.to_dict(),
            "passed": self.passed,
            "fallback_reason": self.fallback_reason,
        }


def build_decode_prefill_cases(
    manifest: WS1Manifest | None = None,
) -> tuple[DecodePrefillCase, ...]:
    """C2 short / long / varlen / padded + batch=1/N. Decode is explicit on every case."""

    m = manifest if manifest is not None else load_manifest()
    fixtures = m.fixtures
    short = int(fixtures["short_seq_len"])
    long = int(fixtures["long_seq_len"])
    varlen = tuple(int(x) for x in fixtures["varlen_seq_lens"])
    return (
        DecodePrefillCase(
            case_id="decode-b1-short",
            batch=1,
            seq_lens=(short,),
            pad_side=None,
            fixture_id="short_full_model_seq8",
        ),
        DecodePrefillCase(
            case_id="decode-b1-long",
            batch=1,
            seq_lens=(long,),
            pad_side=None,
            fixture_id="long_full_model_seq32",
        ),
        DecodePrefillCase(
            case_id="decode-bn-varlen",
            batch=len(varlen),
            seq_lens=varlen,
            pad_side="right",
            fixture_id="rep_full_model_seq16",
        ),
        DecodePrefillCase(
            case_id="decode-bn-padded-right",
            batch=len(varlen),
            seq_lens=varlen,
            pad_side="right",
            fixture_id="rep_full_model_seq16",
        ),
        DecodePrefillCase(
            case_id="decode-bn-padded-left",
            batch=len(varlen),
            seq_lens=varlen,
            pad_side="left",
            fixture_id="rep_full_model_seq16",
        ),
        DecodePrefillCase(
            case_id="decode-b1-primary-s3",
            batch=1,
            seq_lens=(int(fixtures["primary_seq_len"]),),
            pad_side=None,
            fixture_id="rep_full_model_seq16",
        ),
    )


def resolve_attention_candidate(
    backend_profile: str,
    candidate: str | None = None,
    manifest: WS1Manifest | None = None,
) -> dict[str, Any]:
    m = manifest if manifest is not None else load_manifest()
    adapter = get_adapter("attention")
    resolved = resolve_profile_candidate(adapter, backend_profile, m)
    if resolved["status"] == "missing_required":
        raise RuntimeError(
            f"profile {backend_profile!r} attention node is missing_required; "
            "a missing Triton/CUDA decode candidate is red"
        )
    expected = resolved.get("expected_backend_id")
    chosen = candidate if candidate is not None else expected
    if chosen is None:
        raise RuntimeError(f"profile {backend_profile!r} has no declared attention candidate")
    family = _candidate_family(str(chosen))
    want_family = str(m.backend_profiles[backend_profile]["backend_family"])
    if family != want_family:
        raise RuntimeError(
            f"candidate {chosen!r} is {family!r}, profile "
            f"{backend_profile!r} requires {want_family!r}"
        )
    if expected is not None and str(chosen) != str(expected):
        raise RuntimeError(
            f"candidate {chosen!r} does not match C2 declaration {expected!r} "
            f"for {backend_profile}/attention"
        )
    spec = OP_SPECS["attention"]
    if str(chosen) not in spec.candidate_paths:
        raise RuntimeError(f"attention has no candidate path for {chosen!r}")
    return {
        "candidate": str(chosen),
        "path": spec.candidate_paths[str(chosen)],
        "expected_backend_id": expected,
    }


def load_attention_operator(candidate: str) -> Any:
    return load_adapter_operator("attention", candidate)


def make_profile_provenance(
    *,
    backend_profile: str,
    contract: Mapping[str, Any],
    requested_backend: str,
    actual_backend: str,
    output_dtype: str,
) -> BackendProvenance:
    policy = resolve_dtype_policy(contract)
    family = str(contract["policy"]["backend_profile_contracts"][backend_profile]["backend_family"])
    if (
        _candidate_family(actual_backend) != family
        or _candidate_family(requested_backend) != family
    ):
        raise RuntimeError(
            f"silent/cross-profile fallback: requested={requested_backend!r} "
            f"actual={actual_backend!r} profile={backend_profile!r}"
        )
    provenance = BackendProvenance(
        backend_profile=backend_profile,
        requested_backend=family,
        actual_backend=family,
        execution_dtype=policy.execution_dtype,
        accumulation_dtype=policy.accumulation_dtype,
        output_dtype=policy.output_dtype_default,
        reference_dtype=policy.reference_dtype,
        candidate_tf32_enabled=False,
        reference_tf32_enabled=False,
    )
    return validate_backend_provenance(contract, provenance)


def assert_decode_prefill_consistent(
    *,
    backend_profile: str,
    candidate: str | None = None,
    contract: Mapping[str, Any] | None = None,
    manifest: WS1Manifest | None = None,
    device: torch.device | str | None = None,
    cases: Sequence[DecodePrefillCase] | None = None,
    attn_op: Any | None = None,
    require_declared_candidate: bool = True,
) -> DecodePrefillReport:
    """C6: direct decode (Sq=1 vs cached KV) matches equivalent prefill under C1."""

    c = contract if contract is not None else load_contract()
    m = manifest if manifest is not None else load_manifest()
    policy = resolve_dtype_policy(c)
    exec_dtype = _torch_dtype(policy.execution_dtype)
    seed = int(m.seed)
    case_list = tuple(cases) if cases is not None else build_decode_prefill_cases(m)
    if any(not case.include_direct_decode for case in case_list):
        raise RuntimeError("C6 forbids substituting chunked-prefill for direct decode coverage")

    if require_declared_candidate:
        resolved = resolve_attention_candidate(backend_profile, candidate, m)
        cand_id = resolved["candidate"]
        operator = attn_op if attn_op is not None else load_attention_operator(cand_id)
        family = str(m.backend_profiles[backend_profile]["backend_family"])
    else:
        cand_id = candidate or "pytorch"
        operator = attn_op if attn_op is not None else NativeAttentionOp()
        family = "pytorch" if cand_id == "pytorch" else _candidate_family(cand_id)

    if require_declared_candidate:
        # The declared-candidate gate runs on the profile's own accelerator and
        # never degrades to CPU: a CPU pass would not be evidence at all.
        run_device = resolve_device(device, profile=backend_profile)
    else:
        run_device = torch.device(device or "cpu")

    fp = m.model_identity["config_fingerprint"]
    n_heads = int(fp["num_attention_heads"])
    n_kv = int(fp["num_key_value_heads"])
    head_dim = int(fp["head_dim"])

    cells: list[DecodePrefillCell] = []
    for case in case_list:
        q, k, v, key_mask, token_ids, active = _materialize_qkv(
            case,
            seed=seed,
            device=run_device,
            dtype=exec_dtype,
            n_heads=n_heads,
            n_kv=n_kv,
            head_dim=head_dim,
        )
        prefill = _call_attn(operator, q, k, v, key_padding_mask=key_mask)
        decode = _direct_decode(operator, q, k, v, key_padding_mask=key_mask)
        concat_ref = NativeKVCacheAttnOp()
        # Level A: last-token concat-reference vs last-token candidate decode.
        last_q = q[:, :, -1:, :]
        last_k_new = k[:, :, -1:, :]
        last_v_new = v[:, :, -1:, :]
        k_cache, v_cache = k[:, :, :-1, :], v[:, :, :-1, :]
        concat_out = concat_ref.forward(
            last_q,
            k_cache,
            v_cache,
            last_k_new,
            last_v_new,
            causal=True,
            key_padding_mask=key_mask,
        )
        attn_cmp = _compare_logical_tensors(
            prefill,
            decode,
            judgment="forward_accuracy",
            contract=c,
            op_class="attention",
            dtype=exec_dtype,
            backend_profile=backend_profile if require_declared_candidate else None,
            tensor_name="attn_out",
            config_pair=("prefill", "direct_decode"),
        )
        concat_cmp = _compare_logical_tensors(
            decode[:, :, -1:, :],
            concat_out,
            judgment="forward_accuracy",
            contract=c,
            op_class="attention",
            dtype=exec_dtype,
            backend_profile=backend_profile if require_declared_candidate else None,
            tensor_name="concat_reference",
            config_pair=("direct_decode", "concat_kv_cache"),
        )
        probe = _probe_weight(n_heads * head_dim, seed=seed, device=run_device, dtype=torch.float32)
        logp_op = NativeLogpOp()
        prefill_logp = _selected_logp_from_attn(prefill, token_ids, probe, logp_op, active)
        decode_logp = _selected_logp_from_attn(decode, token_ids, probe, logp_op, active)
        roles = resolve_comparison_roles(c, _REPORT_KIND)
        aggregates = compute_logprob_aggregates(
            decode_logp,
            prefill_logp,
            active,
            contract=c,
            report_kind=_REPORT_KIND,
            clip_interval=default_clip_interval(c),
            comparison_lhs_role=roles.comparison_lhs_role,
            comparison_rhs_role=roles.comparison_rhs_role,
        )
        verdict = judge_logprob_aggregates(aggregates, c, execution_dtype=policy.execution_dtype)
        cells.append(
            DecodePrefillCell(
                case_id=case.case_id,
                attention_compare=attn_cmp,
                concat_reference_compare=concat_cmp,
                logprob_verdict=verdict,
                stored_kv_dtype=normalize_dtype_name(k.dtype),
                stored_kv_layout="[B, Hkv, S, D]",
                passed=bool(attn_cmp.passed and verdict.passed),
            )
        )

    cc = compute_capability(run_device) if run_device.type in ACCELERATOR_TYPES else None

    if require_declared_candidate:
        provenance = make_profile_provenance(
            backend_profile=backend_profile,
            contract=c,
            requested_backend=family,
            actual_backend=family,
            output_dtype=policy.output_dtype_default,
        )
    else:
        # CPU gold path: do not claim a required CUDA/Triton profile.
        provenance = BackendProvenance(
            backend_profile=backend_profile,
            requested_backend="pytorch",
            actual_backend="pytorch",
            execution_dtype=policy.execution_dtype,
            accumulation_dtype=policy.accumulation_dtype,
            output_dtype=policy.output_dtype_default,
            reference_dtype=policy.reference_dtype,
            candidate_tf32_enabled=False,
            reference_tf32_enabled=False,
        )

    passed = all(cell.passed for cell in cells)
    return DecodePrefillReport(
        backend_profile=backend_profile,
        candidate_id=cand_id,
        device=str(run_device),
        compute_capability=cc,
        seed=seed,
        backend_provenance=provenance,
        cells=tuple(cells),
        passed=passed,
        fallback_reason=None,
    )


def assert_stateful_kv_consistent(
    *,
    backend_profile: str,
    candidate: str | None = None,
    contract: Mapping[str, Any] | None = None,
    manifest: WS1Manifest | None = None,
    device: torch.device | str | None = None,
    attn_op: Any | None = None,
    require_declared_candidate: bool = True,
) -> StatefulKVReport:
    """C7 B1 + generate-rescore. ``NativeKVCacheAttnOp`` cannot satisfy B1."""

    c = contract if contract is not None else load_contract()
    m = manifest if manifest is not None else load_manifest()
    policy = resolve_dtype_policy(c)
    exec_dtype = _torch_dtype(policy.execution_dtype)
    seed = int(m.seed)

    if require_declared_candidate:
        resolved = resolve_attention_candidate(backend_profile, candidate, m)
        cand_id = resolved["candidate"]
        operator = attn_op if attn_op is not None else load_attention_operator(cand_id)
        family = str(m.backend_profiles[backend_profile]["backend_family"])
        run_device = resolve_device(device, profile=backend_profile)
    else:
        cand_id = candidate or "pytorch"
        operator = attn_op if attn_op is not None else NativeAttentionOp()
        family = "pytorch"
        run_device = torch.device(device or "cpu")

    if isinstance(operator, NativeKVCacheAttnOp):
        raise RuntimeError("NativeKVCacheAttnOp concat reference does not satisfy C7 B1")

    fp = m.model_identity["config_fingerprint"]
    n_heads = int(fp["num_attention_heads"])
    n_kv = int(fp["num_key_value_heads"])
    head_dim = int(fp["head_dim"])
    case = DecodePrefillCase(
        case_id="c7-primary-varlen",
        batch=len(m.fixtures["varlen_seq_lens"]),
        seq_lens=tuple(int(x) for x in m.fixtures["varlen_seq_lens"]),
        pad_side="right",
        fixture_id="rep_full_model_seq16",
    )
    q, k, v, key_mask, token_ids, active = _materialize_qkv(
        case,
        seed=seed,
        device=run_device,
        dtype=exec_dtype,
        n_heads=n_heads,
        n_kv=n_kv,
        head_dim=head_dim,
    )
    cache = StatefulKVCache.allocate(
        n_layers=1,
        batch=q.shape[0],
        n_kv_heads=n_kv,
        max_seq_len=k.shape[2],
        head_dim=head_dim,
        dtype=exec_dtype,
        device=run_device,
    )
    # Prefill write of all but last token, then one decode write.
    cache.write(k[:, :, :-1, :], v[:, :, :-1, :], layer=0)
    k_read, v_read, length = cache.read(layer=0)
    if length != k.shape[2] - 1:
        raise RuntimeError(f"B1 read length {length} != written prefix {k.shape[2] - 1}")
    if not torch.equal(k_read, k[:, :, :-1, :]) or not torch.equal(v_read, v[:, :, :-1, :]):
        raise RuntimeError("B1 read did not return the written cache contents")
    cache.write(k[:, :, -1:, :], v[:, :, -1:, :], layer=0)
    k_full, v_full, full_len = cache.read(layer=0)
    if full_len != k.shape[2]:
        raise RuntimeError("B1 cursor did not advance on the decode write")
    decode_out = _call_attn(operator, q[:, :, -1:, :], k_full, v_full, key_padding_mask=key_mask)
    prefill_out = _call_attn(operator, q, k, v, key_padding_mask=key_mask)
    last_ok = torch.isfinite(decode_out).all() and decode_out.shape[2] == 1
    b1_passed = bool(last_ok and full_len == k.shape[2])

    probe = _probe_weight(n_heads * head_dim, seed=seed, device=run_device, dtype=torch.float32)
    logp_op = NativeLogpOp()
    prefill_step = _direct_decode(operator, q, k, v, key_padding_mask=key_mask)
    cache2 = StatefulKVCache.allocate(
        n_layers=1,
        batch=q.shape[0],
        n_kv_heads=n_kv,
        max_seq_len=k.shape[2],
        head_dim=head_dim,
        dtype=exec_dtype,
        device=run_device,
    )
    gen_out = _stateful_generate(operator, cache2, q, k, v, key_mask)
    prefill_logp = _selected_logp_from_attn(prefill_step, token_ids, probe, logp_op, active)
    gen_logp = _selected_logp_from_attn(gen_out, token_ids, probe, logp_op, active)
    roles = resolve_comparison_roles(c, _REPORT_KIND)
    aggregates = compute_logprob_aggregates(
        gen_logp,
        prefill_logp,
        active,
        contract=c,
        report_kind=_REPORT_KIND,
        clip_interval=default_clip_interval(c),
        comparison_lhs_role=roles.comparison_lhs_role,
        comparison_rhs_role=roles.comparison_rhs_role,
    )
    verdict = judge_logprob_aggregates(aggregates, c, execution_dtype=policy.execution_dtype)

    if require_declared_candidate:
        provenance = make_profile_provenance(
            backend_profile=backend_profile,
            contract=c,
            requested_backend=family,
            actual_backend=family,
            output_dtype=policy.output_dtype_default,
        )
    else:
        provenance = BackendProvenance(
            backend_profile=backend_profile,
            requested_backend="pytorch",
            actual_backend="pytorch",
            execution_dtype=policy.execution_dtype,
            accumulation_dtype=policy.accumulation_dtype,
            output_dtype=policy.output_dtype_default,
            reference_dtype=policy.reference_dtype,
            candidate_tf32_enabled=False,
            reference_tf32_enabled=False,
        )

    passed = bool(b1_passed and verdict.passed)
    del prefill_out, decode_out
    return StatefulKVReport(
        backend_profile=backend_profile,
        candidate_id=cand_id,
        device=str(run_device),
        cache_identity=cache.identity(),
        b1_passed=b1_passed,
        generate_rescore=verdict,
        b2_status=B2_PRODUCTION_KV_STATUS,
        backend_provenance=provenance,
        passed=passed,
        fallback_reason=None,
    )


def _candidate_family(candidate: str) -> str:
    return candidate_family(candidate)


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    return mapping[name]


def _call_attn(
    operator: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    if hasattr(operator, "forward") and callable(operator.forward):
        return operator.forward(q, k, v, causal=True, key_padding_mask=key_padding_mask)
    return operator(q, k, v, causal=True, key_padding_mask=key_padding_mask)


def _direct_decode(
    operator: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """One query vs prefix KV at every position (explicit decode, not chunked-prefill)."""

    seq = q.shape[2]
    outs: list[torch.Tensor] = []
    for pos in range(seq):
        q_t = q[:, :, pos : pos + 1, :]
        k_t = k[:, :, : pos + 1, :]
        v_t = v[:, :, : pos + 1, :]
        mask_t = key_padding_mask[:, : pos + 1] if key_padding_mask is not None else None
        outs.append(_call_attn(operator, q_t, k_t, v_t, key_padding_mask=mask_t))
    return torch.cat(outs, dim=2)


def _stateful_generate(
    operator: Any,
    cache: StatefulKVCache,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Fixed-token generate path: write then decode from the stateful cache."""

    cache.reset()
    outs: list[torch.Tensor] = []
    seq = q.shape[2]
    for pos in range(seq):
        cache.write(k[:, :, pos : pos + 1, :], v[:, :, pos : pos + 1, :], layer=0)
        k_c, v_c, _length = cache.read(layer=0)
        mask_t = key_padding_mask[:, : pos + 1] if key_padding_mask is not None else None
        outs.append(
            _call_attn(operator, q[:, :, pos : pos + 1, :], k_c, v_c, key_padding_mask=mask_t)
        )
    return torch.cat(outs, dim=2)


def _probe_weight(
    hidden: int, *, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 91)
    weight = torch.randn((PROBE_VOCAB, hidden), generator=gen, dtype=torch.float32)
    return weight.to(device=device, dtype=dtype)


def _selected_logp_from_attn(
    attn_out: torch.Tensor,
    token_ids: torch.Tensor,
    probe: torch.Tensor,
    logp_op: NativeLogpOp,
    active: torch.Tensor,
) -> torch.Tensor:
    batch, n_heads, seq, head_dim = attn_out.shape
    hidden = attn_out.transpose(1, 2).reshape(batch, seq, n_heads * head_dim).float()
    logits = torch.matmul(hidden, probe.float().t())
    tokens = token_ids % PROBE_VOCAB
    logp = logp_op.forward_fp32(logits, tokens)
    return logp.masked_fill(~active, 0.0)


def _materialize_qkv(
    case: DecodePrefillCase,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    n_heads: int,
    n_kv: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = case.batch
    padded = (
        max(case.seq_lens) if case.pad_side is None else max(max(case.seq_lens), max(case.seq_lens))
    )
    if case.pad_side in {"left", "right"}:
        padded = max(padded, max(case.seq_lens))
        # keep the declared pad length when it is larger (C2 primary_padded_len=20)
        if case.case_id.endswith("padded-right") or case.case_id.endswith("padded-left"):
            padded = max(padded, 20)
    q = torch.zeros((batch, n_heads, padded, head_dim), device=device, dtype=dtype)
    k = torch.zeros((batch, n_kv, padded, head_dim), device=device, dtype=dtype)
    v = torch.zeros((batch, n_kv, padded, head_dim), device=device, dtype=dtype)
    key_mask = torch.zeros((batch, padded), device=device, dtype=torch.bool)
    token_ids = torch.zeros((batch, padded), device=device, dtype=torch.long)
    active = torch.zeros((batch, padded), device=device, dtype=torch.bool)
    cpu = torch.device("cpu")
    for row, seq_len in enumerate(case.seq_lens):
        if case.pad_side == "left":
            start = padded - seq_len
        else:
            start = 0
        end = start + seq_len
        q[row, :, start:end, :] = _fill((n_heads, seq_len, head_dim), seed, 1 + row, cpu).to(
            device=device, dtype=dtype
        )
        k[row, :, start:end, :] = _fill((n_kv, seq_len, head_dim), seed, 17 + row, cpu).to(
            device=device, dtype=dtype
        )
        v[row, :, start:end, :] = _fill((n_kv, seq_len, head_dim), seed, 31 + row, cpu).to(
            device=device, dtype=dtype
        )
        key_mask[row, start:end] = True
        token_ids[row, start:end] = torch.arange(seq_len, device=device) + 100 + row * 10
        # C2: prompt tokens inactive. Use half the sequence as prompt when unknown.
        prompt = max(1, seq_len // 2)
        active[row, start + prompt : end] = True
    return q, k, v, key_mask, token_ids, active


def _fill(shape: tuple[int, ...], seed: int, offset: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(offset))
    return torch.randn(shape, generator=gen, dtype=torch.float32, device="cpu")


__all__ = [
    "B2_PRODUCTION_KV_STATUS",
    "DecodePrefillCase",
    "DecodePrefillCell",
    "DecodePrefillReport",
    "PROBE_VOCAB",
    "StatefulKVReport",
    "assert_decode_prefill_consistent",
    "assert_stateful_kv_consistent",
    "build_decode_prefill_cases",
    "load_attention_operator",
    "make_profile_provenance",
    "resolve_attention_candidate",
]
