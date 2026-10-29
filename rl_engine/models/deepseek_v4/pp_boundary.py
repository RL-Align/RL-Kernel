# SPDX-License-Identifier: Apache-2.0
"""DSV4 PP2 stage/version boundary around Megatron's P2P communicator."""

from dataclasses import dataclass
import hashlib
import json

import torch
import torch.distributed as dist


class PpBoundaryError(RuntimeError):
    pass


def _require(ok, message):
    if not ok:
        raise PpBoundaryError(message)


@dataclass(frozen=True)
class Dsv4Pp2Plan:
    ranks: tuple[int, int]
    layer_ranges: tuple[tuple[int, int], tuple[int, int]]
    activation_shape: tuple[int, ...]
    dtype: torch.dtype
    model_version: str

    def __post_init__(self):
        _require(
            len(self.ranks) == 2
            and len(set(self.ranks)) == 2
            and all(type(rank) is int and rank >= 0 for rank in self.ranks),
            "PP2 requires two unique global ranks",
        )
        first, second = self.layer_ranges
        _require(
            first[0] == 0
            and 0 < first[1] == second[0] < second[1]
            and all(type(n) is int for segment in self.layer_ranges for n in segment),
            "contiguous global layer stages required",
        )
        _require(
            isinstance(self.activation_shape, tuple)
            and bool(self.activation_shape)
            and all(type(dim) is int and dim > 0 for dim in self.activation_shape),
            "fixed positive activation shape required",
        )
        _require(
            self.dtype in (torch.bfloat16, torch.float16, torch.float32),
            "unsupported activation dtype",
        )
        _require(
            isinstance(self.model_version, str) and bool(self.model_version),
            "model version required",
        )

    def stage_for_layer(self, layer):
        _require(
            type(layer) is int and 0 <= layer < self.layer_ranges[-1][1], "layer outside PP plan"
        )
        return int(layer >= self.layer_ranges[0][1])

    @property
    def fingerprint(self):
        payload = (
            self.ranks,
            self.layer_ranges,
            self.activation_shape,
            str(self.dtype),
            self.model_version,
        )
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


class Dsv4Pp2Boundary:
    """One microbatch's forward activation and backward gradient crossing.

    For production, ``communicator`` is Megatron's P2PCommunicator. The
    caller owns its process group and config; this adapter fixes stage layout,
    version and call order but does not replace the P2P transport.
    """

    def __init__(
        self, plan: Dsv4Pp2Plan, *, group, communicator=None, config=None, allow_cpu_reference=False
    ):
        _require(isinstance(plan, Dsv4Pp2Plan), "PP2 plan required")
        _require(dist.is_available() and dist.is_initialized(), "process group required")
        _require(
            tuple(dist.get_process_group_ranks(group)) == plan.ranks,
            "PP group differs from layer map",
        )
        backend = str(dist.get_backend(group)).lower()
        rocm = "nccl" in backend and bool(torch.version.hip)
        _require(rocm or ("gloo" in backend and allow_cpu_reference), "unvalidated PP transport")
        self.device = (
            torch.device("cuda", torch.cuda.current_device()) if rocm else torch.device("cpu")
        )
        self.plan, self.group = plan, group
        self.stage = plan.ranks.index(dist.get_rank())
        self._phase = "forward"
        self._microbatch_id = None
        if communicator is None:
            _require(
                rocm and config is not None and config.pipeline_dtype == plan.dtype,
                "Megatron pipeline config and matching dtype required",
            )
            from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

            communicator = P2PCommunicator(group, config)
        else:
            _require(allow_cpu_reference and not rocm, "injected P2P communicator is test-only")
        self.communicator = communicator
        self._agree(("construct", plan.fingerprint), True)

    def _agree(self, signature, valid):
        peers = [None, None]
        dist.all_gather_object(peers, (signature, valid), group=self.group)
        _require(all(peer[0] == signature for peer in peers), f"PP stage/version mismatch: {peers}")
        _require(all(peer[1] for peer in peers), "invalid PP stage activation/gradient layout")

    def _layout(self, value):
        return (
            isinstance(value, torch.Tensor)
            and tuple(value.shape) == self.plan.activation_shape
            and value.dtype == self.plan.dtype
            and value.device == self.device
            and value.is_contiguous()
        )

    def forward(self, activation, *, microbatch_id, model_version):
        try:
            valid = (
                self._phase == "forward"
                and model_version == self.plan.model_version
                and type(microbatch_id) is int
                and microbatch_id >= 0
                and (
                    (self.stage == 0 and self._layout(activation))
                    or (self.stage == 1 and activation is None)
                )
            )
            self._agree(("forward", microbatch_id, model_version, self.plan.fingerprint), valid)
            if self.stage == 0:
                self.communicator.send_forward(activation, is_last_stage=False)
                result = None
            else:
                result = self.communicator.recv_forward(
                    self.plan.activation_shape, is_first_stage=False
                )
                _require(self._layout(result), "received forward activation layout mismatch")
        except Exception:
            self._phase = "poisoned"
            raise
        self._phase = "backward"
        self._microbatch_id = microbatch_id
        return result

    def backward(self, gradient, *, microbatch_id, model_version):
        try:
            valid = (
                self._phase == "backward"
                and model_version == self.plan.model_version
                and microbatch_id == self._microbatch_id
                and (
                    (self.stage == 1 and self._layout(gradient))
                    or (self.stage == 0 and gradient is None)
                )
            )
            self._agree(("backward", microbatch_id, model_version, self.plan.fingerprint), valid)
            if self.stage == 1:
                self.communicator.send_backward(gradient, is_first_stage=False)
                result = None
            else:
                result = self.communicator.recv_backward(
                    self.plan.activation_shape, is_last_stage=False
                )
                _require(self._layout(result), "received backward gradient layout mismatch")
        except Exception:
            self._phase = "poisoned"
            raise
        self._phase = "used"
        return result
