# SPDX-License-Identifier: Apache-2.0
"""Saved-forward-plan expert gradient round trip for DSV4."""

from dataclasses import dataclass
import hashlib
import json

import torch
import torch.distributed as dist


class ExpertBackwardError(RuntimeError):
    pass


def _require(ok, message):
    if not ok:
        raise ExpertBackwardError(message)


@dataclass(frozen=True)
class ForwardSlot:
    token: int
    topk_slot: int
    expert: int
    destination_rank: int


@dataclass(frozen=True)
class SavedForwardRoute:
    """Immutable receipt supplied by the forward expert-dispatch owner.

    Slots are in destination-rank, then token/Top-K order. The expert rank
    receives them in source-rank order; no backward rerouting is permitted.
    """

    run_id: str
    model_version: str
    global_ranks: tuple[int, ...]
    local_tokens: int
    hidden_size: int
    topk_width: int
    slots: tuple[ForwardSlot, ...]
    forward_artifact_fingerprint: str = ""
    placement_version: str = ""

    def __post_init__(self):
        _require(bool(self.run_id) and bool(self.model_version), "run/model version required")
        _require(
            self.global_ranks and tuple(sorted(set(self.global_ranks))) == self.global_ranks,
            "sorted unique global ranks required",
        )
        _require(
            all(type(rank) is int and rank >= 0 for rank in self.global_ranks),
            "invalid global rank",
        )
        _require(
            type(self.local_tokens) is int
            and self.local_tokens >= 0
            and type(self.hidden_size) is int
            and self.hidden_size > 0
            and type(self.topk_width) is int
            and 1 <= self.topk_width <= 6,
            "invalid route dimensions",
        )
        _require(
            isinstance(self.slots, tuple)
            and all(isinstance(slot, ForwardSlot) for slot in self.slots),
            "immutable forward slots required",
        )
        keys = []
        for slot in self.slots:
            _require(
                type(slot.token) is int
                and 0 <= slot.token < self.local_tokens
                and type(slot.topk_slot) is int
                and 0 <= slot.topk_slot < self.topk_width
                and type(slot.expert) is int
                and slot.expert >= 0
                and slot.destination_rank in self.global_ranks,
                "invalid forward slot",
            )
            keys.append((slot.destination_rank, slot.token, slot.topk_slot))
        _require(keys == sorted(keys), "forward slots must follow fixed rank/token/slot order")
        _require(
            len({(s.token, s.topk_slot) for s in self.slots}) == len(self.slots),
            "duplicate forward token/slot",
        )
        _require(
            isinstance(self.forward_artifact_fingerprint, str)
            and isinstance(self.placement_version, str),
            "invalid forward provenance",
        )

    @classmethod
    def from_p3_forward(
        cls,
        bundle,
        saved_route,
        forward_ctx,
        *,
        expert_owners,
        global_ranks,
        source_rank,
        local_tokens,
        hidden_size,
        expected_token_set,
        placement_version,
    ):
        """Consume P3 T05's sealed forward, never recompute Top-K in backward."""
        try:
            from rl_engine.p3.checker import validate_plan
            from rl_engine.p3.contract import SAVED_ROUTE_SCHEMA
            from rl_engine.p3.provider import validate_saved
        except ImportError as exc:
            raise ExpertBackwardError("P3 RoutePlan provider is not installed") from exc
        _require(
            isinstance(expert_owners, (tuple, list))
            and len(expert_owners) == 256
            and all(rank in global_ranks for rank in expert_owners),
            "complete physical expert placement required",
        )
        _require(
            isinstance(expected_token_set, (tuple, list))
            and list(expected_token_set) == sorted(set(expected_token_set)),
            "explicit sorted source token set required",
        )
        try:
            validate_plan(
                bundle,
                expected_placement_version=placement_version,
                expected_runtime={
                    "run_id": forward_ctx.run_id,
                    "rank": source_rank,
                    "expected_token_set": list(expected_token_set),
                },
            )
            validate_saved(saved_route, SAVED_ROUTE_SCHEMA, forward_ctx)
        except ValueError as exc:
            raise ExpertBackwardError("P3 forward plan/saved state is invalid") from exc
        plan = bundle["RoutePlan"]
        envelopes = plan["envelope"]
        _require(
            bool(envelopes) and envelopes[0]["envelope_schema_version"] == "RoutePlanEnvelope.v2",
            "P3 ownership envelope v2 required",
        )
        artifact = plan["route_artifact_fingerprint"]
        _require(
            forward_ctx.rank == source_rank
            and forward_ctx.route_artifact_fingerprint == artifact
            and saved_route.header["forward_invocation_id"] == forward_ctx.invocation_id
            and saved_route.header["route_artifact_fingerprint"] == artifact,
            "saved forward invocation/artifact mismatch",
        )
        _require(
            type(local_tokens) is int and local_tokens == len(plan["core"]) // 6,
            "P3 source row count differs from local tokens",
        )
        slots = []
        for index, (core, envelope) in enumerate(zip(plan["core"], envelopes, strict=True)):
            _require(
                envelope["rank"] == source_rank and envelope["source_row"] == index // 6,
                "P3 source rank/row mismatch",
            )
            if core["valid"]:
                physical = envelope["physical_expert_id"]
                _require(
                    type(physical) is int and 0 <= physical < 256,
                    "P3 physical expert outside placement",
                )
                slots.append(
                    ForwardSlot(
                        envelope["source_row"],
                        core["topk_index"],
                        physical,
                        expert_owners[physical],
                    )
                )
        identity = plan["identity"]
        version = (
            f"{identity['checkpoint_id']}:{identity['weight_id']}:{identity['weight_fingerprint']}"
        )
        return cls(
            forward_ctx.run_id,
            version,
            tuple(global_ranks),
            local_tokens,
            hidden_size,
            6,
            tuple(
                sorted(slots, key=lambda slot: (slot.destination_rank, slot.token, slot.topk_slot))
            ),
            artifact,
            placement_version,
        )

    @property
    def fingerprint(self):
        payload = (
            self.run_id,
            self.model_version,
            self.global_ranks,
            self.local_tokens,
            self.hidden_size,
            self.topk_width,
            tuple((s.token, s.topk_slot, s.expert, s.destination_rank) for s in self.slots),
            self.forward_artifact_fingerprint,
            self.placement_version,
        )
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


class ExpertBackwardRoundTrip:
    """One-shot dy dispatch and dX return, bound to the saved forward route."""

    def __init__(self, receipt: SavedForwardRoute, *, group, allow_cpu_reference=False):
        _require(isinstance(receipt, SavedForwardRoute), "saved forward receipt required")
        _require(dist.is_available() and dist.is_initialized(), "process group required")
        _require(
            tuple(dist.get_process_group_ranks(group)) == receipt.global_ranks,
            "saved route group differs from process group",
        )
        backend = str(dist.get_backend(group)).lower()
        rocm = "nccl" in backend and bool(torch.version.hip)
        _require(
            rocm or ("gloo" in backend and allow_cpu_reference), "unvalidated expert transport"
        )
        self.device = (
            torch.device("cuda", torch.cuda.current_device()) if rocm else torch.device("cpu")
        )
        self.receipt = receipt
        self.group = group
        self.rank = dist.get_rank()
        self.world_size = len(receipt.global_ranks)
        _require(self.rank in receipt.global_ranks, "rank outside saved route")
        self.local_index = receipt.global_ranks.index(self.rank)
        peers = [None] * self.world_size
        dist.all_gather_object(peers, receipt, group=group)
        _require(
            all(
                peer.global_ranks == receipt.global_ranks
                and peer.run_id == receipt.run_id
                and peer.model_version == receipt.model_version
                and peer.hidden_size == receipt.hidden_size
                and peer.topk_width == receipt.topk_width
                for peer in peers
            ),
            "forward route metadata mismatch",
        )
        self.send_counts = [
            sum(s.destination_rank == rank for s in receipt.slots) for rank in receipt.global_ranks
        ]
        self.recv_counts = [
            sum(s.destination_rank == self.rank for s in peer.slots) for peer in peers
        ]
        self.inbound_slots = tuple(
            (source, slot.token, slot.topk_slot, slot.expert)
            for source, peer in zip(receipt.global_ranks, peers, strict=True)
            for slot in peer.slots
            if slot.destination_rank == self.rank
        )
        self._phase = "ready"
        self._dtype = None

    def _agree(self, signature, valid):
        peers = [None] * self.world_size
        dist.all_gather_object(peers, (signature, valid), group=self.group)
        _require(all(peer[0] == signature for peer in peers), f"gradient phase mismatch: {peers}")
        _require(all(peer[1] for peer in peers), "stale run/plan or invalid gradient layout")

    def _preflight(
        self, phase, run_id, model_version, fingerprint, tensor, rows, recomputed_route=None
    ):
        try:
            dtype = str(tensor.dtype) if isinstance(tensor, torch.Tensor) else None
            valid = (
                self._phase == phase
                and run_id == self.receipt.run_id
                and model_version == self.receipt.model_version
                and fingerprint == self.receipt.fingerprint
                and (
                    recomputed_route is None
                    or (
                        isinstance(recomputed_route, SavedForwardRoute)
                        and recomputed_route.fingerprint == self.receipt.fingerprint
                    )
                )
                and isinstance(tensor, torch.Tensor)
                and tuple(tensor.shape) == (rows, self.receipt.hidden_size)
                and tensor.device == self.device
                and tensor.is_contiguous()
                and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)
                and (self._dtype is None or tensor.dtype == self._dtype)
            )
            self._agree((phase, run_id, model_version, dtype), valid)
            return tensor.dtype
        except Exception:
            self._phase = "poisoned"
            raise

    def dispatch_dy(self, dy, *, run_id, model_version, forward_fingerprint, recomputed_route=None):
        dtype = self._preflight(
            "ready",
            run_id,
            model_version,
            forward_fingerprint,
            dy,
            self.receipt.local_tokens,
            recomputed_route,
        )
        try:
            indices = torch.tensor(
                [slot.token for slot in self.receipt.slots], device=self.device, dtype=torch.long
            )
            send = dy.index_select(0, indices).contiguous()
            received = dy.new_empty((len(self.inbound_slots), self.receipt.hidden_size))
            dist.all_to_all_single(
                received, send, self.recv_counts, self.send_counts, group=self.group
            )
        except Exception:
            self._phase = "poisoned"
            raise
        self._phase = "return"
        self._dtype = dtype
        return received

    def return_dx(self, expert_dx, *, run_id, model_version, forward_fingerprint):
        self._preflight(
            "return", run_id, model_version, forward_fingerprint, expert_dx, len(self.inbound_slots)
        )
        try:
            slot_dx = expert_dx.new_empty((len(self.receipt.slots), self.receipt.hidden_size))
            dist.all_to_all_single(
                slot_dx, expert_dx, self.send_counts, self.recv_counts, group=self.group
            )
            flat = slot_dx.new_zeros(
                (self.receipt.local_tokens * self.receipt.topk_width, self.receipt.hidden_size)
            )
            indices = torch.tensor(
                [s.token * self.receipt.topk_width + s.topk_slot for s in self.receipt.slots],
                device=self.device,
                dtype=torch.long,
            )
            flat.index_copy_(0, indices, slot_dx)
            lanes = flat.view(
                self.receipt.local_tokens, self.receipt.topk_width, self.receipt.hidden_size
            )
            total = lanes[:, 0].float()
            for slot in range(1, self.receipt.topk_width):
                total = total + lanes[:, slot].float()
        except Exception:
            self._phase = "poisoned"
            raise
        self._phase = "used"
        return total.to(expert_dx.dtype)

    def route_weight_gradient(self, dy, expert_outputs):
        """FP32 dp_s for local slot outputs in saved destination order."""
        _require(self._phase == "used", "dX return required before route gradient")
        _require(
            tuple(dy.shape) == (self.receipt.local_tokens, self.receipt.hidden_size)
            and tuple(expert_outputs.shape) == (len(self.receipt.slots), self.receipt.hidden_size)
            and dy.device == self.device
            and expert_outputs.device == self.device,
            "route gradient layout mismatch",
        )
        indices = torch.tensor(
            [s.token for s in self.receipt.slots], device=self.device, dtype=torch.long
        )
        scalars = (dy.index_select(0, indices).float() * expert_outputs.float()).sum(-1)
        output = torch.zeros(
            (self.receipt.local_tokens * self.receipt.topk_width,),
            dtype=torch.float32,
            device=self.device,
        )
        positions = torch.tensor(
            [s.token * self.receipt.topk_width + s.topk_slot for s in self.receipt.slots],
            device=self.device,
            dtype=torch.long,
        )
        output.index_copy_(0, positions, scalars)
        return output.view(self.receipt.local_tokens, self.receipt.topk_width)
