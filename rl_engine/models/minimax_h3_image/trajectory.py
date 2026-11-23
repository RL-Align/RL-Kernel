# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Sealed six-step trajectory artifact for MiniMax-H3-Image (RFC #738 ``h3_image_evidence_schema``).

It extends the MiniMax-H3 trajectory artifact (RFC #420 ``h3_trajectory_artifact``) with
what an image request adds. The hashing, the FP32 hex encoding, the per-step hash chain and
the tensor comparison are the H3 ones; the record layout is new.

``inputs`` holds everything a run is *given*, hashed field by field. Two runs are
comparable only when it matches, and ``first_drift`` reports the first differing field
before looking at any step:

* ``task`` (``t2i`` / ``edit``) and seed;
* ``canvas``: requested size, resolved canvas, latent grid, minimum generation area and
  the post-decode resize rule;
* ``presentation``: prompt, token ids, token tags, text-row count, truncation;
* ``references`` (edit only), in packed order: source bytes, VAE input shape, latent grid,
  the FP16-rounded posterior sample and the conditioned rows at ``t = 0.999``;
* ``vae``: latent-statistics digest and the tiling plan (size, overlap, plan digests);
* ``layout``: the ``image_position_grid`` artifact digests, sequence length, the
  image/text index tensors and the reference-row count;
* ``noise``: target noise and per-reference noise digests (draw order is list order);
* ``schedule``: the sigmas the sampler steps on, the six timesteps and the reference
  timestep, all as FP32 hex, and the six PDD head digests from the architecture fingerprint.

Each of the six ``step_records`` holds the PDD head index (must equal the step index), the
sigma pair, the per-row timestep table, the velocity of every image row and of the
generated rows, the generated latent after the update, and the reference rows. For an edit
the reference rows must hash to the conditioned value at every step (RFC #738 I-C6); a T2I
trajectory has none. ``decoded`` optionally carries the output bytes: secondary evidence,
reported by ``first_drift`` only after every step matched.

The artifact is bound to the H3-Image architecture fingerprint and optionally to an
operator trace. torch is imported only to hash or compare tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_engine.models.minimax_h3 import trajectory as h3_traj
from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image.position_grid import MAX_REFERENCES

SCHEMA_VERSION = "rlkernel.h3_image.trajectory.v1"
TASKS = ("t2i", "edit")
# Order in which one step's records are compared: the timestep table feeds the model, the
# velocity feeds the update, the update must leave the references alone.
STEP_KINDS = ("timestep_table", "velocity", "generated_velocity", "reference_rows", "latent")
INPUT_FIELDS = (
    "task",
    "seed",
    "canvas",
    "presentation",
    "references",
    "vae",
    "layout",
    "noise",
    "schedule",
)
REQUIRED = {
    "canvas": ("requested", "canvas", "latent", "min_generation_area", "resize"),
    "presentation": ("prompt_sha256", "token_ids_sha256", "token_tags_sha256", "num_text_tokens"),
    "vae": ("latent_stats_sha256", "tile_size", "tile_overlap_min", "encode_tiled", "decode_tiled"),
    "layout": (
        "position_sha256_fp64",
        "position_sha256_fp32",
        "sequence_length",
        "image_indices_sha256",
        "text_indices_sha256",
        "num_reference_rows",
    ),
    "reference": (
        "source_sha256",
        "vae_input_shape",
        "latent",
        "posterior_fp16_sha256",
        "conditioned_rows_sha256",
    ),
}

tensor_digest = h3_traj.tensor_digest
f32_hex = h3_traj.f32_hex
f32_from_hex = h3_traj.f32_from_hex
_sha = h3_traj._sha
GENESIS = h3_traj.GENESIS


def released_schedule() -> dict[str, Any]:
    """The schedule block every strict trajectory must carry, from the fingerprint."""

    grid = fp.sigma_grid()
    return {
        "step_sigmas": [f32_hex(s) for s in grid["step_sigmas"]],
        "timesteps": [f32_hex(t) for t in grid["timesteps"]],
        "reference_timestep": f32_hex(fp.REFERENCE_TIMESTEP),
        "euler_update_form": fp.SCHEDULER["euler_update_form"],
        "pdd_head_sha256": [h["weight"] for h in fp.pdd_head_digests()],
    }


def _check_fields(name: str, block: Mapping[str, Any], required: Sequence[str]) -> None:
    missing = [key for key in required if key not in block]
    if missing:
        raise ValueError(f"{name} is missing {missing}")


def _check_inputs(inputs: Mapping[str, Any]) -> None:
    if set(inputs) != set(INPUT_FIELDS):
        raise ValueError(f"inputs must hold exactly {INPUT_FIELDS}")
    if inputs["task"] not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {inputs['task']!r}")
    for name in ("canvas", "presentation", "vae", "layout"):
        _check_fields(name, inputs[name], REQUIRED[name])
    references = inputs["references"]
    if inputs["task"] == "t2i" and references:
        raise ValueError("a t2i request has no references")
    if inputs["task"] == "edit" and not references:
        raise ValueError("an edit request needs at least one reference")
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"more than {MAX_REFERENCES} references")
    if inputs["task"] == "edit" and not inputs["layout"].get("conditioned_reference_rows_sha256"):
        raise ValueError("an edit layout needs conditioned_reference_rows_sha256")
    for index, reference in enumerate(references):
        _check_fields(f"reference {index}", reference, REQUIRED["reference"])
    if len(inputs["noise"].get("references", [])) != len(references):
        raise ValueError("one reference noise digest per reference is required")
    if "target_sha256" not in inputs["noise"]:
        raise ValueError("noise is missing target_sha256")
    if (inputs["layout"]["num_reference_rows"] == 0) != (inputs["task"] == "t2i"):
        raise ValueError("num_reference_rows must be 0 exactly for t2i")
    if inputs["schedule"] != released_schedule():
        raise ValueError("schedule differs from the released PDD schedule")
    canvas = inputs["canvas"]
    multiple = fp.CANVAS_MULTIPLE
    if canvas["canvas"][0] % multiple or canvas["canvas"][1] % multiple:
        raise ValueError(f"canvas must be a multiple of {multiple}")
    ratio = fp.VAE_SPATIAL_COMPRESSION
    if list(canvas["latent"]) != [canvas["canvas"][0] // ratio, canvas["canvas"][1] // ratio]:
        raise ValueError("latent grid is not canvas / 16")


@dataclass(frozen=True)
class ImageStepRecord:
    index: int
    pdd_head: int
    sigma: str
    sigma_next: str
    timestep_table: Mapping[str, Any]  # {"values": [f32 hex], "indices_sha256": str}
    velocity_sha256: str  # every image row the head projected (references + generated)
    generated_velocity_sha256: str
    reference_rows_sha256: str  # "" for t2i
    latent_sha256: str  # generated rows after the update
    chain_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "pdd_head": self.pdd_head,
            "sigma": self.sigma,
            "sigma_next": self.sigma_next,
            "timestep_table": dict(self.timestep_table),
            "velocity_sha256": self.velocity_sha256,
            "generated_velocity_sha256": self.generated_velocity_sha256,
            "reference_rows_sha256": self.reference_rows_sha256,
            "latent_sha256": self.latent_sha256,
            "chain_sha256": self.chain_sha256,
        }


def tensor_key(step: int, kind: str) -> str:
    return f"{step}.{kind}"


class ImageTrajectoryRecorder:
    """Builds an H3-Image artifact step by step; ``seal()`` returns the immutable dict.

    ``record_step`` takes the velocity of every image row and the image-row latents after
    the scheduler update, both ``[batch, image_rows, 96]`` in ``image_indices`` order, and
    splits them at ``num_reference_rows`` itself, so the reference/generated boundary is
    the one sealed in ``inputs.layout``.
    """

    def __init__(
        self,
        *,
        label: str,
        inputs: Mapping[str, Any],
        operator_trace_sha256: str = "",
        provenance: Mapping[str, Any] | None = None,
        keep_tensors: bool = False,
    ) -> None:
        _check_inputs(inputs)
        self.label = label
        self.inputs = {key: inputs[key] for key in INPUT_FIELDS}
        self.num_reference_rows = int(inputs["layout"]["num_reference_rows"])
        self.schedule = inputs["schedule"]
        self.steps = fp.NUM_PDD_HEADS
        self.operator_trace_sha256 = operator_trace_sha256
        self.provenance = dict(provenance or {})
        self.records: list[ImageStepRecord] = []
        self.keep_tensors = keep_tensors
        self.tensors: dict[str, Any] = {}
        self.decoded: dict[str, Any] | None = None

    def _reference_digest(self, rows: Any) -> str:
        return tensor_digest(rows) if self.num_reference_rows else ""

    def record_step(
        self,
        index: int,
        *,
        pdd_head: int,
        timestep_values: Any,
        timestep_indices: Any,
        velocity: Any,
        image_latents: Any,
    ) -> ImageStepRecord:
        """Record denoising step ``index`` (consecutive from 0)."""

        if index != len(self.records):
            raise ValueError(f"expected step {len(self.records)}, got {index}")
        if index >= self.steps:
            raise ValueError(f"the released schedule has only {self.steps} steps")
        if isinstance(pdd_head, bool) or int(pdd_head) != pdd_head or pdd_head != index:
            raise ValueError(f"step {index} must run PDD head {index}, got {pdd_head!r}")
        if velocity.shape != image_latents.shape or velocity.ndim != 3:
            raise ValueError("velocity and image_latents must both be [batch, image_rows, width]")
        rows = int(image_latents.shape[1])
        if rows <= self.num_reference_rows:
            raise ValueError("image rows must include generated rows after the references")
        reference_rows = image_latents[:, : self.num_reference_rows]
        generated = image_latents[:, self.num_reference_rows :]
        reference_digest = self._reference_digest(reference_rows)
        if reference_digest != _conditioned_reference_digest(self.inputs):
            raise ValueError(f"step {index}: reference rows differ from the conditioned rows")
        table = {
            "values": [f32_hex(float(v)) for v in timestep_values.reshape(-1).tolist()],
            "indices_sha256": tensor_digest(timestep_indices),
        }
        body: dict[str, Any] = {
            "index": index,
            "pdd_head": int(pdd_head),
            "sigma": self.schedule["step_sigmas"][index],
            "sigma_next": self.schedule["step_sigmas"][index + 1],
            "timestep_table": table,
            "velocity_sha256": tensor_digest(velocity),
            "generated_velocity_sha256": tensor_digest(velocity[:, self.num_reference_rows :]),
            "reference_rows_sha256": reference_digest,
            "latent_sha256": tensor_digest(generated),
        }
        previous = self.records[-1].chain_sha256 if self.records else GENESIS
        record = ImageStepRecord(chain_sha256=h3_traj._chain(previous, body), **body)
        self.records.append(record)
        if self.keep_tensors:
            kept = {
                "velocity": velocity,
                "generated_velocity": velocity[:, self.num_reference_rows :],
                "reference_rows": reference_rows,
                "latent": generated,
                "timestep_table": timestep_indices,
            }
            for kind, tensor in kept.items():
                self.tensors[tensor_key(index, kind)] = tensor.detach().cpu().clone()
        return record

    def record_decoded(self, image_uint8: Any, *, output_size: Sequence[int]) -> None:
        """Attach the decoded 8-bit image (secondary evidence) after the last step."""

        if len(self.records) != self.steps:
            raise ValueError("decode is recorded after all six steps")
        if str(image_uint8.dtype) != "torch.uint8":
            raise ValueError("decoded output must be uint8")
        self.decoded = {"sha256": tensor_digest(image_uint8), "output_size": list(output_size)}

    def seal(self) -> dict[str, Any]:
        if len(self.records) != self.steps:
            raise ValueError(f"recorded {len(self.records)} of {self.steps} steps")
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "architecture_fingerprint_sha256": fp.build_fingerprint()["fingerprint_sha256"],
            "label": self.label,
            "steps": self.steps,
            "inputs": self.inputs,
            "inputs_sha256": {key: _sha(self.inputs[key]) for key in INPUT_FIELDS},
            "operator_trace_sha256": self.operator_trace_sha256,
            "provenance": self.provenance,
            "step_records": [record.to_dict() for record in self.records],
            "final_latent_sha256": self.records[-1].latent_sha256,
            "decoded": self.decoded,
        }
        artifact["artifact_sha256"] = _sha(artifact)
        return artifact

    def save_tensors(self, path: Path) -> None:
        """Write kept tensors as safetensors (keys ``step.kind``)."""

        if not self.keep_tensors:
            raise ValueError("recorder was created with keep_tensors=False")
        from safetensors.torch import save_file

        save_file({key: value.contiguous() for key, value in self.tensors.items()}, str(path))


# --- validation ---------------------------------------------------------------------------


def verify(artifact: Mapping[str, Any], *, check_fingerprint: bool = True) -> None:
    """Fail closed unless the artifact is consistent, sealed and on the released schedule."""

    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported trajectory schema {artifact.get('schema_version')!r}")
    body = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    if artifact.get("artifact_sha256") != _sha(body):
        raise ValueError("artifact_sha256 does not match the content")
    inputs = artifact["inputs"]
    _check_inputs(inputs)
    if artifact["inputs_sha256"] != {key: _sha(inputs[key]) for key in INPUT_FIELDS}:
        raise ValueError("inputs_sha256 does not match the inputs")
    if artifact["steps"] != fp.NUM_PDD_HEADS or len(artifact["step_records"]) != fp.NUM_PDD_HEADS:
        raise ValueError(f"a released trajectory has exactly {fp.NUM_PDD_HEADS} steps")
    schedule = inputs["schedule"]
    reference_t = schedule["reference_timestep"]
    conditioned = _conditioned_reference_digest(inputs)
    previous = GENESIS
    for index, record in enumerate(artifact["step_records"]):
        if record["index"] != index:
            raise ValueError(f"step {index}: out-of-order record {record['index']}")
        if record["pdd_head"] != index:
            raise ValueError(f"step {index}: ran PDD head {record['pdd_head']}")
        if record["sigma"] != schedule["step_sigmas"][index]:
            raise ValueError(f"step {index}: sigma disagrees with the schedule")
        if record["sigma_next"] != schedule["step_sigmas"][index + 1]:
            raise ValueError(f"step {index}: sigma_next disagrees with the schedule")
        values = record["timestep_table"]["values"]
        if schedule["timesteps"][index] not in values:
            raise ValueError(f"step {index}: timestep table lacks the schedule timestep")
        expected_values = {schedule["timesteps"][index]}
        if inputs["task"] == "edit":
            expected_values.add(reference_t)
        if set(values) != expected_values:
            raise ValueError(f"step {index}: timestep table holds {values}")
        if sorted(values, key=f32_from_hex) != values:
            raise ValueError(f"step {index}: timestep table is not sorted")
        if record["reference_rows_sha256"] != conditioned:
            raise ValueError(f"step {index}: reference rows changed")
        step_body = {k: v for k, v in record.items() if k != "chain_sha256"}
        if record["chain_sha256"] != h3_traj._chain(previous, step_body):
            raise ValueError(f"step {index}: hash chain broken")
        previous = record["chain_sha256"]
    if artifact["final_latent_sha256"] != artifact["step_records"][-1]["latent_sha256"]:
        raise ValueError("final_latent_sha256 differs from the last step")
    if check_fingerprint and (
        artifact["architecture_fingerprint_sha256"] != fp.build_fingerprint()["fingerprint_sha256"]
    ):
        raise ValueError("artifact was recorded against a different H3-Image fingerprint")


def _conditioned_reference_digest(inputs: Mapping[str, Any]) -> str:
    """Digest the reference rows must keep: the packed conditioned rows, or "" for t2i."""

    if inputs["task"] == "t2i":
        return ""
    return str(inputs["layout"]["conditioned_reference_rows_sha256"])


def dumps(artifact: Mapping[str, Any]) -> str:
    import json

    verify(artifact)
    return json.dumps(artifact, indent=1, sort_keys=True) + "\n"


def loads(text: str, *, check_fingerprint: bool = True) -> dict[str, Any]:
    import json

    artifact = json.loads(text)
    verify(artifact, check_fingerprint=check_fingerprint)
    return artifact


# --- comparison ---------------------------------------------------------------------------


def input_mismatches(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> list[str]:
    """Input fields that differ, in ``INPUT_FIELDS`` order, with the first differing key."""

    problems: list[str] = []
    for field in INPUT_FIELDS:
        if lhs["inputs_sha256"][field] == rhs["inputs_sha256"][field]:
            continue
        a, b = lhs["inputs"][field], rhs["inputs"][field]
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            problems.append(f"{field}.{keys[0]}" if keys else field)
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            index = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
            problems.append(f"{field}[{index}]")
        else:
            problems.append(field)
    return problems


def first_drift(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
    *,
    tensors_lhs: Path | None = None,
    tensors_rhs: Path | None = None,
) -> dict[str, Any] | None:
    """First point where two image trajectories differ, or ``None`` if bitwise equal.

    Inputs are compared first (a differing input explains everything after it), then each
    step in ``STEP_KINDS`` order, then the decoded bytes. Step entries name the PDD head
    and whether the drift is in reference or generated rows.
    """

    verify(lhs)
    verify(rhs)
    problems = input_mismatches(lhs, rhs)
    if problems:
        return {"stage": "inputs", "field": problems[0], "fields": problems}
    for a, b in zip(lhs["step_records"], rhs["step_records"]):
        for kind in STEP_KINDS:
            key = f"{kind}_sha256" if kind != "timestep_table" else kind
            if a[key] != b[key]:
                drift: dict[str, Any] = {
                    "stage": "step",
                    "step": a["index"],
                    "pdd_head": a["pdd_head"],
                    "kind": kind,
                    "rows": {
                        "reference_rows": "reference",
                        "velocity": "all_image",
                        "timestep_table": "all",
                    }.get(kind, "generated"),
                }
                if tensors_lhs is not None and tensors_rhs is not None:
                    drift.update(
                        h3_traj._tensor_difference(
                            tensors_lhs, tensors_rhs, tensor_key(a["index"], kind)
                        )
                    )
                return drift
    if (lhs["decoded"] or {}).get("sha256") != (rhs["decoded"] or {}).get("sha256"):
        return {"stage": "decoded", "field": "sha256"}
    return None
