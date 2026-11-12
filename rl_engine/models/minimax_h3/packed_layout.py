# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU identity, validation, permutations, and exact snapshots of H3 packed rows.

This module owns metadata, not feature tensors or arithmetic kernels. It does
not register a backend or establish full-model/GPU support. See the public
contract in ``docs/contracts/h3-packed-layout.md`` for the supported v1 domain.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

import torch

from rl_engine.reference.minimax_h3.adaln_row_gather import h3_adaln_indices

# --- Basic definitions: constants, dictionary types, and errors -------------------------------

_SCHEMA_VERSION = "h3-packed-layout.v1"
_PROFILE = "cpu.single_source"
_DTYPES = {
    "text_indices": torch.int64,
    "video_indices": torch.int64,
    "audio_indices": torch.int64,
    "token_tags": torch.int64,
    "timestep_indices": torch.int64,
    "timestep": torch.float32,
    "position_ids": torch.float64,
}
_Stream = Literal["text", "video", "audio"]
# The text feature stream can contain visual conditioning rows from the encoder.
_SOURCE_TAGS: dict[_Stream, tuple[int, ...]] = {
    "text": (0, 1),
    "video": (0,),
    "audio": (2,),
}
_ROW_FIELDS = ("token_tags", "timestep_indices", "position_ids")


class H3RowIdentity(TypedDict):
    """The dictionary returned by ``row_identity``."""

    batch_index: int
    packed_row: int
    source_stream: _Stream
    source_index: int
    modality: int
    timestep_table_row: int
    timestep: float
    position: tuple[float, float, float]
    output_stream: Literal["video", "audio"] | None
    output_selection_index: int | None
    role: Literal["condition", "target"]


class H3PackedLayoutError(ValueError):
    """A field does not satisfy the declared CPU layout or snapshot contract."""


# --- Helpers: metadata validation, provenance, and byte order ---------------------------------

# Metadata validation.


def _validate_int(name: str, value: int, low: int, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        domain = f">= {low}" if high is None else f"in [{low}, {high}]"
        raise H3PackedLayoutError(f"{name}: expected an integer {domain}, got {value!r}")
    return value


def _copy_metadata_tensor(name: str, value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise H3PackedLayoutError(f"{name}: expected a torch.Tensor, got {type(value).__name__}")
    if value.device.type != "cpu":
        raise H3PackedLayoutError(f"{name}: expected CPU metadata, got device {value.device}")
    if value.dtype != dtype:
        raise H3PackedLayoutError(f"{name}: expected {dtype}, got {value.dtype}")
    if value.layout != torch.strided or value.is_nested or value.requires_grad:
        raise H3PackedLayoutError(f"{name}: expected dense metadata without gradients")
    names = getattr(value, "names", None)
    if names is not None and any(dim_name is not None for dim_name in names):
        raise H3PackedLayoutError(f"{name}: expected unnamed metadata, got names {names}")
    return value.detach().clone(memory_format=torch.contiguous_format)


def _validate_index_range(name: str, values: torch.Tensor, upper: int) -> None:
    bad = ((values < 0) | (values >= upper)).nonzero()
    if bad.numel():
        index = int(bad[0, 0])
        raise H3PackedLayoutError(
            f"{name}[{index}]: packed/table index {int(values[index])} outside [0, {upper})"
        )


def _timestep_values(times: torch.Tensor) -> tuple[float, ...]:
    """
    Validate/decode IEEE-754 binary32 without ambient floating-point comparisons.
    FP32 subnormals become normal FP64 values. Integer bit checks and FP64
    ldexp preserve them even when the caller enables flush-to-zero. Signed
    zeros compare equal for table uniqueness but retain their stored sign.

    """
    values = []
    first_rows: dict[int, int] = {}
    for row, signed_bits in enumerate(times.view(torch.int32).tolist()):
        # Binary32 stores a sign, an 8-bit exponent (bias 127), and 23 fraction bits.
        bits = signed_bits & 0xFFFFFFFF
        magnitude = bits & 0x7FFFFFFF
        exponent = (magnitude >> 23) & 0xFF
        fraction = magnitude & 0x7FFFFF
        if exponent == 0xFF:
            raise H3PackedLayoutError(f"timestep[{row}]: expected a finite value")
        if magnitude > 0x3F800000 or (bits >> 31 and magnitude != 0):
            raise H3PackedLayoutError(f"timestep[{row}]: expected a value in [0, 1]")
        if magnitude == 0:
            value = -0.0 if bits >> 31 else 0.0
        elif exponent == 0:
            value = math.ldexp(fraction, -149)
        else:
            value = math.ldexp((1 << 23) | fraction, exponent - 150)
        if magnitude in first_rows:
            raise H3PackedLayoutError(
                f"timestep: expected distinct values; timestep[{row}] duplicates "
                f"timestep[{first_rows[magnitude]}] (value {value!r})"
            )
        first_rows[magnitude] = row
        values.append(value)
    return tuple(values)


# Provenance validation and copying.


def _copy_json_value(value: Any, active_containers: set[int]) -> Any:
    """Copy JSON-compatible values without converting floats to decimal text."""
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, int):
        int.__repr__(value)  # Apply the same integer digit limit as the JSON encoder.
        return int.__int__(value)
    if isinstance(value, float):
        number = float.__float__(value)
        if not math.isfinite(number):
            raise ValueError("expected finite JSON data")
        return number
    if not isinstance(value, (dict, list, tuple)):
        raise TypeError(f"expected JSON data, got {type(value).__name__}")

    container_id = id(value)
    if container_id in active_containers:
        raise ValueError("circular reference in JSON data")
    active_containers.add(container_id)
    try:
        if isinstance(value, dict):
            copied = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                copied[str.__str__(key)] = _copy_json_value(item, active_containers)
            return dict(sorted(copied.items()))
        copied_items = []
        for item in value:
            copied_items.append(_copy_json_value(item, active_containers))
        return copied_items
    finally:
        active_containers.remove(container_id)


def _copy_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise H3PackedLayoutError("provenance: expected a JSON-compatible mapping")
    try:
        context = _copy_json_value(dict(value), set())
    except (TypeError, ValueError) as exc:
        raise H3PackedLayoutError(f"provenance: {exc}") from exc
    for name in ("checkpoint_revision", "runtime_revision"):
        revision = context.get(name)
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise H3PackedLayoutError(
                f"provenance.{name}: expected a full lowercase 40-hex revision"
            )
    if context.get("task_family") not in ("FL2VA", "Ref2VA"):
        raise H3PackedLayoutError("provenance.task_family: expected 'FL2VA' or 'Ref2VA'")
    return context


# Byte order for exact tensor snapshots.


def _swap_if_big_endian(data: bytes, width: int) -> bytes:
    if sys.byteorder == "little":
        return data
    return b"".join(data[index : index + width][::-1] for index in range(0, len(data), width))


# --- Core layout lifecycle --------------------------------------------------------------------


class H3PackedLayout:
    """
    CPU packed-row metadata with validated ownership and row mappings.
    Owns independent metadata copies; tensor properties return copies.
    Supports row identity queries, safe permutations, and exact snapshots.

    """

    __slots__ = (
        "_fields",
        "_batch_size",
        "_condition_counts",
        "_provenance",
        "_owners",
        "_time_values",
    )

    # Initialization and validation.

    def __init__(
        self,
        *,
        text_indices: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        token_tags: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        position_ids: torch.Tensor,
        provenance: Mapping[str, Any],
        num_condition_video_rows: int,
        num_condition_audio_rows: int,
        batch_size: int = 1,
    ) -> None:
        """Validate and copy layout metadata.

        Pass the producer's condition row counts, using ``0`` for streams without condition rows.
        Required provenance: checkpoint_revision and runtime_revision (40 lowercase hex),
        and task_family ("FL2VA" or "Ref2VA"). Optional: JSON-compatible preparation.
        """
        fields: dict[str, torch.Tensor] = {
            "text_indices": text_indices,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "token_tags": token_tags,
            "timestep_indices": timestep_indices,
            "timestep": timestep,
            "position_ids": position_ids,
        }

        for name, dtype in _DTYPES.items():
            fields[name] = _copy_metadata_tensor(name, fields[name], dtype)

        positions = fields["position_ids"]
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise H3PackedLayoutError(
                f"position_ids: expected shape (S, 3) with S > 0, got {tuple(positions.shape)}"
            )

        num_rows = positions.shape[0]
        for name, values in fields.items():
            if name == "position_ids":
                continue
            if values.ndim != 1:
                raise H3PackedLayoutError(
                    f"{name}: expected a 1-D tensor, got {tuple(values.shape)}"
                )
            if name in _ROW_FIELDS and values.shape != (num_rows,):
                raise H3PackedLayoutError(
                    f"{name}: expected shape ({num_rows},), got {tuple(values.shape)}"
                )

        times = fields["timestep"]
        if times.numel() == 0:
            raise H3PackedLayoutError("timestep: expected at least one distinct time")

        bad = (~torch.isfinite(positions)).nonzero()
        if bad.numel():
            location = ",".join(str(int(item)) for item in bad[0])
            raise H3PackedLayoutError(f"position_ids[{location}]: expected a finite value")

        self._time_values = _timestep_values(times)

        _validate_index_range("timestep_indices", fields["timestep_indices"], times.numel())

        # Build the packed row ownership map with exclusive ownership and complete coverage.
        owners: list[tuple[_Stream, int] | None] = [None] * num_rows
        tags = fields["token_tags"].tolist()
        for stream, allowed_tags in _SOURCE_TAGS.items():
            name = f"{stream}_indices"
            _validate_index_range(name, fields[name], num_rows)

            for source_row, packed_row in enumerate(fields[name].tolist()):
                owner = owners[packed_row]
                if owner is not None:
                    raise H3PackedLayoutError(
                        f"{name}[{source_row}]: packed row {packed_row} already owned by "
                        f"{owner[0]}_indices[{owner[1]}]"
                    )
                tag = tags[packed_row]
                if tag not in allowed_tags:
                    expected = ", ".join(str(value) for value in allowed_tags)
                    raise H3PackedLayoutError(
                        f"token_tags[{packed_row}]: expected one of {{{expected}}} for "
                        f"{name}[{source_row}], got {tag}"
                    )
                owners[packed_row] = (stream, source_row)

        if None in owners:
            raise H3PackedLayoutError(
                f"source_indices: packed row {owners.index(None)} has no source"
            )

        self._batch_size = _validate_int("batch_size", batch_size, 1)
        self._condition_counts = {
            "video": _validate_int(
                "num_condition_video_rows",
                num_condition_video_rows,
                0,
                fields["video_indices"].numel(),
            ),
            "audio": _validate_int(
                "num_condition_audio_rows",
                num_condition_audio_rows,
                0,
                fields["audio_indices"].numel(),
            ),
        }

        self._provenance = _copy_provenance(provenance)

        self._fields = fields
        self._owners = tuple(owner for owner in owners if owner is not None)

    # Properties: context, source selectors, row metadata, and derived indices.

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def sequence_length(self) -> int:
        return self._fields["token_tags"].numel()

    @property
    def source_lengths(self) -> dict[str, int]:
        return {stream: self._fields[f"{stream}_indices"].numel() for stream in _SOURCE_TAGS}

    @property
    def provenance(self) -> Mapping[str, Any]:
        return _copy_json_value(self._provenance, set())

    @property
    def num_condition_video_rows(self) -> int:
        return self._condition_counts["video"]

    @property
    def num_condition_audio_rows(self) -> int:
        return self._condition_counts["audio"]

    @property
    def text_indices(self) -> torch.Tensor:
        return self._fields["text_indices"].clone()

    @property
    def video_indices(self) -> torch.Tensor:
        return self._fields["video_indices"].clone()

    @property
    def audio_indices(self) -> torch.Tensor:
        return self._fields["audio_indices"].clone()

    @property
    def token_tags(self) -> torch.Tensor:
        return self._fields["token_tags"].clone()

    @property
    def timestep(self) -> torch.Tensor:
        return self._fields["timestep"].clone()

    @property
    def timestep_indices(self) -> torch.Tensor:
        return self._fields["timestep_indices"].clone()

    @property
    def position_ids(self) -> torch.Tensor:
        return self._fields["position_ids"].clone()

    @property
    def adaln_indices(self) -> torch.Tensor:
        return h3_adaln_indices(self._fields["timestep_indices"], self._fields["token_tags"])

    # Row identity lookup.

    def row_identity(self, packed_row: int, *, batch_index: int = 0) -> H3RowIdentity:
        """Describe one row; the selected output slot is in source order."""
        _validate_int("packed_row", packed_row, 0, self.sequence_length - 1)
        _validate_int("batch_index", batch_index, 0, self.batch_size - 1)
        stream, source_row = self._owners[packed_row]
        time_row = int(self._fields["timestep_indices"][packed_row])
        position = self._fields["position_ids"][packed_row].tolist()

        output_stream: Literal["video", "audio"] | None
        output_selection_index: int | None
        role: Literal["condition", "target"]
        if stream == "text":
            output_stream = None
            output_selection_index = None
            role = "condition"
        else:
            output_stream = stream
            output_selection_index = source_row
            role = "condition" if source_row < self._condition_counts[stream] else "target"

        return {
            "batch_index": batch_index,
            "packed_row": packed_row,
            "source_stream": stream,
            "source_index": source_row,
            "modality": int(self._fields["token_tags"][packed_row]),
            "timestep_table_row": time_row,
            "timestep": self._time_values[time_row],
            "position": (position[0], position[1], position[2]),
            "output_stream": output_stream,
            "output_selection_index": output_selection_index,
            "role": role,
        }

    # Safe packed-row permutation.

    def permute_rows(self, perm: torch.Tensor) -> H3PackedLayout:
        """Return a new layout for ``perm[new_row] = old_row``.

        Move already-packed feature tensors separately with ``index_select``
        on their row axis. Source feature streams retain their original order.
        """
        perm = _copy_metadata_tensor("perm", perm, torch.int64)
        num_rows = self.sequence_length
        if perm.shape != (num_rows,):
            raise H3PackedLayoutError(
                f"perm: expected shape ({num_rows},), got {tuple(perm.shape)}"
            )
        _validate_index_range("perm", perm, num_rows)
        if torch.unique(perm).numel() != num_rows:
            raise H3PackedLayoutError(
                f"perm: expected each packed row in [0, {num_rows}) exactly once"
            )
        old_to_new = torch.empty(num_rows, dtype=torch.int64, device="cpu")
        old_to_new[perm] = torch.arange(num_rows, dtype=torch.int64, device="cpu")
        fields = {name: self._fields[name].index_select(0, perm) for name in _ROW_FIELDS}
        for stream in _SOURCE_TAGS:
            selector_name = f"{stream}_indices"
            fields[selector_name] = old_to_new.index_select(0, self._fields[selector_name])
        fields["timestep"] = self._fields["timestep"]
        return H3PackedLayout(
            **fields,
            provenance=self._provenance,
            batch_size=self.batch_size,
            num_condition_video_rows=self.num_condition_video_rows,
            num_condition_audio_rows=self.num_condition_audio_rows,
        )

    # Exact snapshot export and restoration.

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot with exact tensor bytes."""
        tensors = {}
        for name, value in self._fields.items():
            data = value.view(torch.uint8).numpy().tobytes()
            tensors[name] = {
                "dtype": str(value.dtype).removeprefix("torch."),
                "shape": list(value.shape),
                "data_le_hex": _swap_if_big_endian(data, value.element_size()).hex(),
            }
        return {
            "schema_version": _SCHEMA_VERSION,
            "profile": _PROFILE,
            "batch_size": self.batch_size,
            "num_condition_video_rows": self.num_condition_video_rows,
            "num_condition_audio_rows": self.num_condition_audio_rows,
            "provenance": self.provenance,
            "tensors": tensors,
        }

    @classmethod
    def from_dict(cls, snapshot: Mapping[str, Any]) -> H3PackedLayout:
        """Restore the exact supported version, then perform all constructor checks."""
        keys = {
            "schema_version",
            "profile",
            "batch_size",
            "num_condition_video_rows",
            "num_condition_audio_rows",
            "provenance",
            "tensors",
        }
        if not isinstance(snapshot, Mapping) or set(snapshot) != keys:
            raise H3PackedLayoutError("snapshot: unexpected or missing top-level fields")
        if snapshot["schema_version"] != _SCHEMA_VERSION or snapshot["profile"] != _PROFILE:
            raise H3PackedLayoutError("snapshot: unsupported schema_version or profile")
        tensor_specs = snapshot["tensors"]
        if not isinstance(tensor_specs, Mapping) or set(tensor_specs) != set(_DTYPES):
            raise H3PackedLayoutError("snapshot.tensors: unexpected or missing layout fields")
        fields = {}
        for name, dtype in _DTYPES.items():
            tensor_spec = tensor_specs[name]
            prefix = f"snapshot.tensors.{name}"
            if not isinstance(tensor_spec, Mapping) or set(tensor_spec) != {
                "dtype",
                "shape",
                "data_le_hex",
            }:
                raise H3PackedLayoutError(f"{prefix}: expected dtype, shape, data_le_hex")
            if tensor_spec["dtype"] != str(dtype).removeprefix("torch."):
                raise H3PackedLayoutError(f"{prefix}.dtype: expected {dtype}")
            shape = tensor_spec["shape"]
            ndim = 2 if name == "position_ids" else 1
            if not isinstance(shape, list) or len(shape) != ndim:
                raise H3PackedLayoutError(f"{prefix}.shape: expected {ndim} dimensions")
            for index, dim in enumerate(shape):
                _validate_int(f"{prefix}.shape[{index}]", dim, 0, (1 << 63) - 1)
            data_le_hex = tensor_spec["data_le_hex"]
            if not isinstance(data_le_hex, str):
                raise H3PackedLayoutError(f"{prefix}.data_le_hex: expected a hex string")
            try:
                data = bytes.fromhex(data_le_hex)
            except ValueError as exc:
                raise H3PackedLayoutError(f"{prefix}.data_le_hex: invalid hex bytes") from exc
            count = math.prod(shape)
            width = dtype.itemsize
            if len(data) != count * width:
                raise H3PackedLayoutError(f"{prefix}.data_le_hex: byte count disagrees with shape")
            native = _swap_if_big_endian(data, width)
            if count:
                fields[name] = torch.frombuffer(bytearray(native), dtype=dtype).reshape(shape)
            else:
                fields[name] = torch.empty(shape, dtype=dtype, device="cpu")
        return cls(
            **fields,
            provenance=snapshot["provenance"],
            batch_size=snapshot["batch_size"],
            num_condition_video_rows=snapshot["num_condition_video_rows"],
            num_condition_audio_rows=snapshot["num_condition_audio_rows"],
        )


# --- Public interface -------------------------------------------------------------------------

__all__ = ["H3PackedLayout", "H3PackedLayoutError", "H3RowIdentity"]
