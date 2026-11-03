# SPDX-License-Identifier: Apache-2.0
"""Normal route training, immutable result records and diagnostic fixed-ID replay."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.moe.router_training import implementation
from rl_engine.contracts.operators.router_training import RECORD_SCHEMA, VJP_SCHEMA
from rl_engine.ops.moe.hash_route import hash_route
from rl_engine.ops.moe.learned_route import (
    _digest,
    _logical_ids,
    _row_hashes,
    _validate,
    _version,
    learned_route,
    tensor_sha256,
)


def _bits(tensor):
    return tensor.detach().cpu().contiguous().view(torch.int32).tolist()


@dataclass(frozen=True)
class RouteRecord:
    """Canonical JSON is an owned immutable snapshot, including FP32 payload bits."""

    payload_json: str
    sha256: str

    def payload(self):
        data = json.loads(self.payload_json)
        if _digest(data) != self.sha256:
            raise ValueError("route record checksum mismatch")
        _validate_record(data)
        return data


def _record(data):
    _validate_record(data)
    return RouteRecord(json.dumps(data, sort_keys=True, separators=(",", ":")), _digest(data))


def _validate_record(data):
    if not isinstance(data, dict) or data.get("schema") != RECORD_SCHEMA:
        raise ValueError("unsupported route record schema")
    kind = data.get("kind")
    if kind not in {"learned", "hash"} or data.get("mode") not in {"normal", "diagnostic_fixed"}:
        raise ValueError("invalid routing kind/mode")
    layer = data.get("absolute_layer")
    if type(layer) is not int or not (3 <= layer < 43 if kind == "learned" else 0 <= layer < 3):
        raise ValueError("record absolute-layer mismatch")
    versions = data.get("versions", {})
    keys = {"model_version", "bias_version", "domain_version", "plan_version"}
    if kind == "hash":
        keys = {"model_version", "table_version"}
    if not isinstance(versions, dict) or set(versions) != keys:
        raise ValueError("record version fields mismatch")
    for key, value in versions.items():
        _version(value, key)
    _version(data.get("record_version"), "record_version")
    rank, world = data.get("rank"), data.get("world_size")
    if type(rank) is not int or type(world) is not int or not 0 <= rank < world:
        raise ValueError("invalid record rank/world size")
    gids = data.get("global_ids")
    if not isinstance(gids, list) or any(type(v) is not int for v in gids):
        raise ValueError("invalid global ID map")
    if sorted(gids) != list(range(256)) or (kind == "hash" and gids != list(range(256))):
        raise ValueError("record requires complete global expert coverage")
    scale = data.get("scale")
    if (
        type(scale) not in (int, float)
        or not math.isfinite(scale)
        or not 0 <= scale <= torch.finfo(torch.float32).max
    ):
        raise ValueError("invalid record scale")
    if type(data.get("renormalize")) is not bool or (kind == "learned" and not data["renormalize"]):
        raise ValueError("invalid normalization mode")
    rows = data.get("tokens")
    if not isinstance(rows, list):
        raise ValueError("invalid recorded token rows")
    token_ids = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("logical_id")) is not int:
            raise ValueError("invalid recorded logical ID")
        token_ids.append(row["logical_id"])
        for field in ("experts", "columns", "keep", "positions", "weight_bits"):
            if not isinstance(row.get(field), list) or len(row[field]) != 6:
                raise ValueError("record requires six slots")
        experts, columns = row["experts"], row["columns"]
        if any(type(v) is not int or not 0 <= v < 256 for v in experts + columns):
            raise ValueError("record expert/column out of range")
        if len(set(experts)) != 6 or experts != [gids[c] for c in columns]:
            raise ValueError("record expert/column identity mismatch")
        if any(type(v) is not bool for v in row["keep"]):
            raise ValueError("record mask must be boolean")
        if any(type(v) is not int or v < -1 or v >= 2**63 for v in row["positions"]):
            raise ValueError("invalid dispatch positions")
        if kind == "learned" and row["keep"] != [p >= 0 for p in row["positions"]]:
            raise ValueError("record mask/dispatch positions disagree")
        if kind == "hash" and (not all(row["keep"]) or row["positions"] != [-1] * 6):
            raise ValueError("Hash replay cannot invent capacity")
        if any(type(v) is not int or not -(2**31) <= v < 2**31 for v in row["weight_bits"]):
            raise ValueError("invalid FP32 weight payload")
        if not torch.isfinite(
            torch.tensor(row["weight_bits"], dtype=torch.int32).view(torch.float32)
        ).all():
            raise ValueError("nonfinite recorded weights")
        digest = row.get("logit_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid recorded logit identity")
    if token_ids != sorted(set(token_ids)) or any(t < 0 or t >= 2**63 for t in token_ids):
        raise ValueError("record logical IDs must be unique, canonical and nonnegative I64")
    source = data.get("source_record_sha256")
    if (data["mode"] == "normal" and source is not None) or (
        data["mode"] == "diagnostic_fixed" and (not isinstance(source, str) or len(source) != 64)
    ):
        raise ValueError("invalid diagnostic source record identity")
    if not isinstance(data.get("execution"), dict):
        raise ValueError("record execution provenance required")


def save_route_record(record, path):
    data = record.payload()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"payload": data, "sha256": record.sha256}, indent=2) + "\n")


def load_route_record(path):
    data = json.loads(Path(path).read_text())
    record = RouteRecord(
        json.dumps(data["payload"], sort_keys=True, separators=(",", ":")), data["sha256"]
    )
    record.payload()
    return record


class _BackwardGuard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, protected):
        ctx.save_for_backward(weights, *protected)
        ctx.hashes = tuple(tensor_sha256(t) for t in (weights, *protected))
        return weights

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        for tensor, expected in zip(ctx.saved_tensors, ctx.hashes, strict=True):
            if tensor_sha256(tensor) != expected:
                raise RuntimeError("router forward state changed before backward")
        return grad, None


@dataclass(frozen=True)
class RecordedRoute:
    logits: torch.Tensor
    weights: torch.Tensor
    expert_ids: torch.Tensor
    record: RouteRecord
    logical_ids: tuple


def _finish(
    logits,
    logical_ids,
    result,
    *,
    kind,
    global_ids,
    columns,
    keep,
    positions,
    renormalize,
    scale,
    versions,
    record_version,
    rank,
    world_size,
    protected=(),
    mode="normal",
    source=None,
):
    ids = _logical_ids(logical_ids)
    if len(ids) != logits.shape[0]:
        raise ValueError("one logical identity required per route row")
    rows = []
    hashes = _row_hashes(logits, global_ids)
    experts_cpu, columns_cpu = result.expert_ids.cpu().tolist(), columns.cpu().tolist()
    mask_cpu, positions_cpu, weights_cpu = (
        keep.cpu().tolist(),
        positions.cpu().tolist(),
        _bits(result.weights),
    )
    for i, token in enumerate(ids):
        rows.append(
            {
                "logical_id": token,
                "experts": experts_cpu[i],
                "columns": columns_cpu[i],
                "keep": mask_cpu[i],
                "positions": positions_cpu[i],
                "weight_bits": weights_cpu[i],
                "logit_sha256": hashes[i],
            }
        )
    record = _record(
        {
            "schema": RECORD_SCHEMA,
            "kind": kind,
            "mode": mode,
            "absolute_layer": result.execution["absolute_layer"],
            "versions": versions,
            "record_version": record_version,
            "rank": rank,
            "world_size": world_size,
            "global_ids": list(global_ids),
            "tokens": sorted(rows, key=lambda row: row["logical_id"]),
            "renormalize": renormalize,
            "scale": scale,
            "execution": result.execution,
            "source_record_sha256": source,
        }
    )
    weights = _BackwardGuard.apply(
        result.weights, (logits, result.expert_ids, columns, keep, positions, *protected)
    )
    return RecordedRoute(logits, weights, result.expert_ids, record, ids)


def normal_learned_route(
    logits,
    logical_ids,
    bias,
    plan,
    *,
    record_version,
    rank=0,
    world_size=1,
    protected=(),
    projection_execution=None,
    **versions,
):
    """Real normal selection; existing AITER selected-weight VJP stays in autograd."""
    implementation()
    result = learned_route(logits, logical_ids, bias, plan, **versions)
    if projection_execution is not None:
        result.execution["gate_projection"] = projection_execution
    return _finish(
        logits,
        logical_ids,
        result,
        kind="learned",
        global_ids=plan.global_ids,
        columns=result.column_indices,
        keep=result.surviving_mask,
        positions=result.dispatch_positions,
        renormalize=True,
        scale=plan.scale,
        versions={k: v for k, v in versions.items() if k != "absolute_layer"},
        record_version=record_version,
        rank=rank,
        world_size=world_size,
        protected=(bias, *protected),
    )


def normal_hash_route(
    logits,
    logical_ids,
    vocabulary_ids,
    table,
    *,
    record_version,
    rank=0,
    world_size=1,
    protected=(),
    projection_execution=None,
    renormalize=True,
    routed_scaling_factor=1.5,
    **versions,
):
    implementation()
    result = hash_route(
        logits,
        vocabulary_ids,
        table,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        **versions,
    )
    if projection_execution is not None:
        result.execution["gate_projection"] = projection_execution
    result.execution["vocabulary_ids_sha256"] = tensor_sha256(vocabulary_ids)
    return _finish(
        logits,
        logical_ids,
        result,
        kind="hash",
        global_ids=tuple(range(256)),
        columns=result.expert_ids,
        keep=torch.ones_like(result.expert_ids, dtype=torch.bool),
        positions=torch.full_like(result.expert_ids, -1),
        renormalize=renormalize,
        scale=routed_scaling_factor,
        versions={k: v for k, v in versions.items() if k != "absolute_layer"},
        record_version=record_version,
        rank=rank,
        world_size=world_size,
        protected=(table._prepared, *protected),
    )


class _ReplayFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, columns, keep, kind, renormalize, scale):
        api, _ = implementation()
        with torch.cuda.device(logits.device):
            weights, scores = api.router_replay_fwd(logits, columns, keep, kind, renormalize, scale)
        if not torch.isfinite(weights).all().item():
            raise ValueError("replayed weights overflowed FP32")
        ctx.save_for_backward(logits, scores, columns, keep)
        ctx.options = kind, renormalize, scale
        return weights

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        if not torch.isfinite(grad).all().item():
            raise ValueError("finite replay weight gradients required")
        logits, scores, columns, keep = ctx.saved_tensors
        api, _ = implementation()
        with torch.cuda.device(logits.device):
            dx = api.router_replay_bwd(logits, scores, columns, keep, grad, *ctx.options)
        if not torch.isfinite(dx).all().item():
            raise ValueError("replay VJP overflowed FP32")
        return dx, None, None, None, None, None


def diagnostic_fixed_route(
    logits, logical_ids, record, *, absolute_layer, versions, allow_changed_logits=False
):
    """No normal selection. Only explicit diagnostics may reuse these saved IDs."""
    data = record.payload()
    if data["mode"] != "normal":
        raise ValueError("diagnostic replay requires an original normal-route record")
    if type(allow_changed_logits) is not bool:
        raise TypeError("allow_changed_logits must be boolean")
    if (
        type(absolute_layer) is not int
        or absolute_layer != data["absolute_layer"]
        or versions != data["versions"]
    ):
        raise ValueError("replay absolute-layer/model/route version barrier mismatch")
    # Reuse the production shape/device/finite/Graph checks before any replay copy.
    if not isinstance(logits, torch.Tensor):
        raise TypeError("replay requires FP32 logits")
    if logits.is_cuda:
        with torch.cuda.device(logits.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("router replay does not support Graph capture")
    _validate(logits, logical_ids, torch.zeros(256, device=logits.device))
    tokens = _logical_ids(logical_ids)
    by_id = {row["logical_id"]: row for row in data["tokens"]}
    if any(token not in by_id for token in tokens):
        raise ValueError("replay token outside the recorded logical domain")
    rows = [by_id[t] for t in tokens]
    hashes = _row_hashes(logits, data["global_ids"])
    if not allow_changed_logits and any(
        h != r["logit_sha256"] for h, r in zip(hashes, rows, strict=True)
    ):
        raise ValueError("replay logits changed; enable diagnostic change explicitly")
    device = logits.device
    columns = torch.tensor([r["columns"] for r in rows], device=device, dtype=torch.int64).reshape(
        -1, 6
    )
    ids = torch.tensor([r["experts"] for r in rows], device=device, dtype=torch.int64).reshape(
        -1, 6
    )
    keep = torch.tensor([r["keep"] for r in rows], device=device, dtype=torch.bool).reshape(-1, 6)
    positions = torch.tensor(
        [r["positions"] for r in rows], device=device, dtype=torch.int64
    ).reshape(-1, 6)
    weights = _ReplayFn.apply(
        logits.contiguous(), columns, keep, data["kind"], data["renormalize"], data["scale"]
    )
    from types import SimpleNamespace

    _, manifest = implementation()
    execution = {
        "absolute_layer": absolute_layer,
        "actual_backend": "aiter.router_replay",
        "source_sha256": manifest["source_sha256"],
        "normal_selection_recomputed": False,
        "diagnostic_only": True,
        "changed_logits_allowed": allow_changed_logits,
    }
    result = SimpleNamespace(weights=weights, expert_ids=ids, execution=execution)
    return _finish(
        logits,
        logical_ids,
        result,
        kind=data["kind"],
        global_ids=data["global_ids"],
        columns=columns,
        keep=keep,
        positions=positions,
        renormalize=data["renormalize"],
        scale=data["scale"],
        versions=versions,
        record_version=data["record_version"],
        rank=data["rank"],
        world_size=data["world_size"],
        mode="diagnostic_fixed",
        source=record.sha256,
    )


def compare_route_records(actual, expected, *, require_normal=True):
    """Compare discrete global choices first; diagnostics cannot earn acceptance."""
    a, e = actual.payload(), expected.payload()
    diagnostic = a["mode"] != "normal" or e["mode"] != "normal"
    if require_normal and diagnostic:
        raise ValueError("fixed-route replay is diagnostic only, never normal-route acceptance")
    for field in (
        "kind",
        "absolute_layer",
        "versions",
        "rank",
        "world_size",
        "record_version",
        "renormalize",
        "scale",
    ):
        if a[field] != e[field]:
            raise ValueError(f"route comparison {field} identity mismatch")
    if [r["logical_id"] for r in a["tokens"]] != [r["logical_id"] for r in e["tokens"]]:
        raise ValueError("route comparison logical domain mismatch")
    if not diagnostic:
        for field in ("actual_backend", "contract_id", "source_sha256", "config"):
            if a["execution"].get(field) != e["execution"].get(field):
                raise ValueError(f"route comparison execution {field} identity mismatch")
        a_gate, e_gate = (
            a["execution"].get("gate_projection"),
            e["execution"].get("gate_projection"),
        )
        if (a_gate is None) != (e_gate is None):
            raise ValueError("route comparison Gate provenance mismatch")
        if a_gate is not None:
            for field in ("actual_backend", "source_sha256", "config", "backward_backend"):
                if a_gate.get(field) != e_gate.get(field):
                    raise ValueError(f"route comparison Gate {field} identity mismatch")
    for ar, er in zip(a["tokens"], e["tokens"], strict=True):
        for field in ("experts", "keep", "positions"):
            for slot, (av, ev) in enumerate(zip(ar[field], er[field], strict=True)):
                if av != ev:
                    return {
                        "status": "discrete_mismatch",
                        "numeric_compared": False,
                        "acceptance": False,
                        "first_difference": {
                            "logical_id": ar["logical_id"],
                            "slot": slot,
                            "field": field,
                            "actual": av,
                            "expected": ev,
                        },
                    }
    for ar, er in zip(a["tokens"], e["tokens"], strict=True):
        if ar["logit_sha256"] != er["logit_sha256"]:
            return {
                "status": "input_mismatch",
                "numeric_compared": False,
                "acceptance": False,
                "first_difference": {"logical_id": ar["logical_id"], "field": "logits"},
            }
        for slot, (av, ev) in enumerate(zip(ar["weight_bits"], er["weight_bits"], strict=True)):
            if av != ev:
                return {
                    "status": "value_mismatch",
                    "numeric_compared": True,
                    "acceptance": False,
                    "first_difference": {
                        "logical_id": ar["logical_id"],
                        "slot": slot,
                        "field": "weight_bits",
                        "actual": av,
                        "expected": ev,
                    },
                }
    return {
        "status": "match" if a["tokens"] else "empty",
        "numeric_compared": bool(a["tokens"]),
        "acceptance": bool(a["tokens"]) and not diagnostic,
        "diagnostic_only": diagnostic,
    }


@dataclass(frozen=True)
class RouteVJPRecord:
    payload_json: str
    sha256: str

    def payload(self):
        data = json.loads(self.payload_json)
        if _digest(data) != self.sha256 or data.get("schema") != VJP_SCHEMA:
            raise ValueError("invalid route VJP record")
        _validate_record(data.get("forward"))
        if _digest(data["forward"]) != data.get("forward_sha256"):
            raise ValueError("VJP forward record identity mismatch")
        _version(data.get("loss_version"), "loss_version")
        for field, width in (("upstream_bits", 6), ("dlogits_bits", 256)):
            rows = data.get(field)
            if not isinstance(rows, list) or len(rows) != len(data["forward"]["tokens"]):
                raise ValueError("VJP payload row count mismatch")
            for row in rows:
                if (
                    not isinstance(row, list)
                    or len(row) != width
                    or any(type(v) is not int or not -(2**31) <= v < 2**31 for v in row)
                ):
                    raise ValueError("invalid VJP FP32 payload")
                if not torch.isfinite(
                    torch.tensor(row, dtype=torch.int32).view(torch.float32)
                ).all():
                    raise ValueError("nonfinite VJP payload")
        return data


def save_vjp_record(record, path):
    data = record.payload()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"payload": data, "sha256": record.sha256}, indent=2) + "\n")


def load_vjp_record(path):
    data = json.loads(Path(path).read_text())
    record = RouteVJPRecord(
        json.dumps(data["payload"], sort_keys=True, separators=(",", ":")), data["sha256"]
    )
    record.payload()
    return record


def record_route_vjp(result, grad_weights, *, loss_version, retain_graph=False):
    """Record the local VJP; training through earlier parameters uses weights.backward()."""
    _version(loss_version, "loss_version")
    if not isinstance(grad_weights, torch.Tensor) or grad_weights.dtype != torch.float32:
        raise TypeError("route VJP requires FP32 upstream weights")
    if grad_weights.shape != result.weights.shape or grad_weights.device != result.weights.device:
        raise ValueError("route VJP upstream shape/device mismatch")
    dx = torch.autograd.grad(
        result.weights, result.logits, grad_weights, retain_graph=retain_graph
    )[0]
    data = result.record.payload()
    gids = data["global_ids"]
    order = sorted(range(len(result.logical_ids)), key=lambda i: result.logical_ids[i])
    canonical_columns = sorted(range(256), key=lambda i: gids[i])
    upstream_bits = _bits(grad_weights)
    payload = {
        "schema": VJP_SCHEMA,
        "forward": data,
        "forward_sha256": result.record.sha256,
        "loss_version": loss_version,
        "upstream_bits": [upstream_bits[i] for i in order],
        "dlogits_bits": _bits(dx.detach().cpu()[order][:, canonical_columns]),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return dx, RouteVJPRecord(text, _digest(payload))


def compare_vjp_records(actual, expected, *, require_normal=True):
    a, e = actual.payload(), expected.payload()
    if a["loss_version"] != e["loss_version"]:
        raise ValueError("VJP loss normalization/version mismatch")
    forward = compare_route_records(
        _record(a["forward"]), _record(e["forward"]), require_normal=require_normal
    )
    if forward["status"] != "match":
        return forward
    for field in ("upstream_bits", "dlogits_bits"):
        for row, (actual_row, expected_row) in enumerate(zip(a[field], e[field], strict=True)):
            for column, (actual_bits, expected_bits) in enumerate(
                zip(actual_row, expected_row, strict=True)
            ):
                if actual_bits != expected_bits:
                    return {
                        "status": "gradient_mismatch"
                        if field == "dlogits_bits"
                        else "upstream_mismatch",
                        "field": field,
                        "acceptance": False,
                        "first_difference": {
                            "logical_id": a["forward"]["tokens"][row]["logical_id"],
                            "slot" if field == "upstream_bits" else "global_expert_id": column,
                            "actual_bits": actual_bits,
                            "expected_bits": expected_bits,
                        },
                    }
    return forward


class RouterTrainingOp:
    def __call__(self, kind, *args, **kwargs):
        if kind == "learned":
            return normal_learned_route(*args, **kwargs)
        if kind == "hash":
            return normal_hash_route(*args, **kwargs)
        raise ValueError("router training kind must be learned or hash")
