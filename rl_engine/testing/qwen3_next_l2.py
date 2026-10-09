# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Frozen full-model Qwen3-Next inputs and original-dtype L2 artifacts.

Native workers receive only JSON requests (tokens, weights and provider identity).
They write a weights_only-safe torch artifact with metadata, prompt_tokens,
response_tokens, prediction_positions, BF16 logits [B,16,151936], FP32 logprobs
[B,16], and layer_digests for all 48 layers. Recurrent debug states describe the
same prediction positions: P-1 through P+14, before consuming response token 16.
No rollout hidden tensor or recurrent state is a recompute input.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from rl_engine.integrations.qwen3_next_subnode import SUBNODES, subnode_layout
from rl_engine.testing.tensor_identity import tensor_bitwise_equal

VOCAB_SIZE = 151936
RESPONSE_LENGTH = 16
PROMPT_LENGTHS = (8, 64, 256, 1024)
FROZEN_TEXTS = (
    "The capital of France is Paris. Explain the route from the library to the river. ",
    "Compute seven plus five, then describe a careful way to check the answer. ",
    "A robot observes a red cup beside a blue plate on a wooden kitchen table. ",
    "Write a precise explanation of how rain reaches a mountain stream in autumn. ",
)
NATIVE_WORKER_MODULE = "rl_engine.integrations.qwen3_next_native_workers"


# The native rollout cache budget every certified profile pins (4 GiB per GPU).
NATIVE_KV_CACHE_BYTES = 4 << 30


def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class L2Case:
    prompt_length: int
    batch_size: int
    prefill: str

    @property
    def case_id(self):
        return f"p{self.prompt_length:04d}-b{self.batch_size}-{self.prefill}"

    @property
    def chunks(self):
        if self.prefill == "full":
            return [self.prompt_length]
        remaining = self.prompt_length - 1
        chunks = [1, min(63, remaining)]
        remaining -= chunks[-1]
        while remaining:
            size = min(64, remaining)
            chunks.append(size)
            remaining -= size
        return chunks


def case_matrix():
    return [
        L2Case(length, batch, prefill)
        for length in PROMPT_LENGTHS
        for batch in (1, 4)
        for prefill in ("full", "chunked")
    ]


def prompt_fixture(tokenizer, tokenizer_identity):
    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in FROZEN_TEXTS]
    if any(
        not row or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in row)
        for row in encoded
    ):
        raise ValueError("Pinned tokenizer produced an empty or out-of-range prompt")
    prompts = {}
    for length in PROMPT_LENGTHS:
        prompts[str(length)] = torch.tensor(
            [(row * ((length + len(row) - 1) // len(row)))[:length] for row in encoded],
            dtype=torch.int64,
        )
        if torch.unique(prompts[str(length)], dim=0).shape[0] != 4:
            raise ValueError("The four pinned prompt samples must be distinct")
    identity = {
        "texts": list(FROZEN_TEXTS),
        "tokenizer": tokenizer_identity,
        "prompt_hashes": {key: tensor_digest(value) for key, value in prompts.items()},
    }
    return {
        "schema_version": 1,
        "prompts": prompts,
        "prompt_identity": identity,
        "prompt_sha256": json_digest(identity),
    }


def freeze_responses(prompts, baseline_artifacts, *, baseline_identity, baseline_records):
    responses, old_logprobs = {}, {}
    for length in PROMPT_LENGTHS:
        artifact = baseline_artifacts[str(length)]
        response = artifact["response_tokens"]
        selected = artifact["logprobs"]
        if response.shape != (4, RESPONSE_LENGTH) or response.dtype != torch.int64:
            raise ValueError(
                "Baseline generation must produce sixteen real tokens for four samples"
            )
        if (
            selected.shape != (4, RESPONSE_LENGTH)
            or selected.dtype != torch.float32
            or not torch.isfinite(selected).all()
        ):
            raise ValueError("Baseline logprobs must be finite FP32 [4,16]")
        if bool(((response < 0) | (response >= VOCAB_SIZE)).any()):
            raise ValueError("Baseline response token is outside the official vocabulary")
        responses[str(length)] = response.detach().cpu().clone()
        old_logprobs[str(length)] = selected.detach().cpu().clone()
    fixture = {
        **prompts,
        "responses": responses,
        "old_logprobs": old_logprobs,
        "baseline_identity": baseline_identity,
        "baseline_artifacts": baseline_records,
    }
    fixture["content_sha256"] = fixture_digest(fixture)
    return fixture


def fixture_digest(fixture):
    identity = {
        "prompt_sha256": fixture["prompt_sha256"],
        "baseline_identity": fixture["baseline_identity"],
        "baseline_artifacts": fixture["baseline_artifacts"],
        "responses": {key: tensor_digest(value) for key, value in fixture["responses"].items()},
        "old_logprobs": {
            key: tensor_digest(value) for key, value in fixture["old_logprobs"].items()
        },
    }
    return json_digest(identity)


def validate_fixture(fixture):
    expected_keys = {str(length) for length in PROMPT_LENGTHS}
    if fixture.get("schema_version") != 1 or any(
        set(fixture[name]) != expected_keys
        for name in ("prompts", "responses", "old_logprobs", "baseline_artifacts")
    ):
        raise ValueError("Incomplete frozen L2 input matrix")
    for record in fixture["baseline_artifacts"].values():
        if (
            set(record) != {"path", "sha256"}
            or not Path(record["path"]).is_absolute()
            or len(record["sha256"]) != 64
        ):
            raise ValueError("Baseline generation artifact identity is incomplete")
    for length in PROMPT_LENGTHS:
        key = str(length)
        prompt, response, logprobs = (
            fixture[name][key] for name in ("prompts", "responses", "old_logprobs")
        )
        if (
            prompt.shape != (4, length)
            or prompt.dtype != torch.int64
            or tensor_digest(prompt) != fixture["prompt_identity"]["prompt_hashes"][key]
        ):
            raise ValueError("Frozen prompt was changed")
        if (
            response.shape != (4, RESPONSE_LENGTH)
            or response.dtype != torch.int64
            or bool(((response < 0) | (response >= VOCAB_SIZE)).any())
        ):
            raise ValueError("Invalid frozen response")
        if (
            logprobs.shape != (4, RESPONSE_LENGTH)
            or logprobs.dtype != torch.float32
            or not torch.isfinite(logprobs).all()
        ):
            raise ValueError("Invalid frozen baseline logprobs")
    if (
        json_digest(fixture["prompt_identity"]) != fixture["prompt_sha256"]
        or fixture_digest(fixture) != fixture["content_sha256"]
    ):
        raise ValueError("Frozen L2 fixture content identity mismatch")


def save_tensor_artifact(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_tensor_artifact(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_weight_identity(identity):
    """Bind mutable export bytes separately from the fixed official model identity.

    This is the JSON boundary check. The native loaders and profile activation also
    call validate_training_export to rehash and inspect the actual exported files.
    """
    if identity.get("provider_config", {}).get("profile") != "qwen3_next_tp4_shared_eager_v1":
        return
    version = identity.get("weight_version")
    if not isinstance(version, str) or not version:
        raise ValueError("Official shared model identity requires an explicit weight version")
    actual = identity.get("checkpoint_identity", {})
    if version == "baseline":
        if actual.get("kind") == "qwen3_next_native_hf_export":
            raise ValueError("An updated native export cannot masquerade as the official baseline")
        if any(key in identity for key in ("base_checkpoint", "base_checkpoint_identity")) and any(
            (
                identity.get("base_checkpoint") != identity.get("checkpoint"),
                identity.get("base_checkpoint_identity") != actual,
            )
        ):
            raise ValueError("Baseline identity cannot redirect its official base checkpoint")
        return
    if any(key not in identity for key in ("base_checkpoint", "base_checkpoint_identity")):
        raise ValueError("Updated weights require a separately pinned official base identity")
    if any(
        (
            not Path(identity["checkpoint"]).is_absolute(),
            not Path(identity["base_checkpoint"]).is_absolute(),
            identity["checkpoint"] == identity["base_checkpoint"],
            actual.get("schema_version") != 1,
            actual.get("kind") != "qwen3_next_native_hf_export",
            actual.get("status") != "complete",
            actual.get("weight_version") != version,
            actual.get("base_checkpoint") != identity["base_checkpoint"],
            actual.get("base_checkpoint_identity") != identity["base_checkpoint_identity"],
            actual.get("architecture_layers") != 48,
            actual.get("weights_dtype") != "bfloat16",
            type(actual.get("step")) is not int,
            actual.get("step", 0) < 1,
        )
    ):
        raise ValueError("Updated checkpoint identity differs from its published native export")
    training = actual.get("training_identity", {})
    fields = ("source_commit", "source_tree", "environment_sha256", "provider_config", "topology")
    if any(training.get(key) != identity.get(key) for key in fields) or any(
        (
            training.get("checkpoint") != identity["base_checkpoint"],
            training.get("checkpoint_identity") != identity["base_checkpoint_identity"],
        )
    ):
        raise ValueError("Native export training provenance differs from the model/base identity")


def worker_request(case, fixture, *, role, identity, generate=False, trace_layers=None):
    if role not in ("rollout", "recompute"):
        raise ValueError("Unknown native model role")
    validate_weight_identity(identity)
    if generate and (role != "rollout" or case.batch_size != 4 or case.prefill != "full"):
        raise ValueError("Only initial B4 full-prefill rollout may define frozen responses")
    if not generate:
        validate_fixture(fixture)
    prompt = fixture["prompts"][str(case.prompt_length)][: case.batch_size]
    request = {
        "schema_version": 1,
        "case_id": case.case_id,
        "role": role,
        "identity": identity,
        "sample_ids": list(range(case.batch_size)),
        "prompt_tokens": prompt.tolist(),
        "response_tokens": (
            None
            if generate
            else fixture["responses"][str(case.prompt_length)][: case.batch_size].tolist()
        ),
        "response_length": RESPONSE_LENGTH,
        "prediction_positions": list(
            range(case.prompt_length - 1, case.prompt_length + RESPONSE_LENGTH - 1)
        ),
        "consumed_response_tokens": RESPONSE_LENGTH - 1,
        "rollout_prefill_chunks": case.chunks,
        "prefill_chunks": case.chunks if role == "rollout" else None,
        "consumed_input_tokens": (
            None
            if generate
            else [
                left + right[:-1]
                for left, right in zip(
                    prompt.tolist(),
                    fixture["responses"][str(case.prompt_length)][: case.batch_size].tolist(),
                )
            ]
        ),
        "state_position": case.prompt_length + RESPONSE_LENGTH - 2,
        "execution": (
            {
                "rollout_mode": "autoregressive",
                "prefill_chunks": case.chunks,
                "response_mode": "generate" if generate else "teacher_forced",
            }
            if role == "rollout"
            else {
                "recompute_mode": "full_from_tokens",
                "input_sequence_length": case.prompt_length + RESPONSE_LENGTH - 1,
            }
        ),
        "fixture_sha256": fixture["prompt_sha256"] if generate else fixture["content_sha256"],
        "logits_dtype": "bfloat16",
        "logprobs_dtype": "float32",
        "vocab_size": VOCAB_SIZE,
        "trace_layers": list(trace_layers or []),
        "required_layer_digests": list(range(48)),
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 1234,
            "ignore_eos": True,
        },
    }
    request["request_sha256"] = json_digest(request)
    return request


def layer_state_names(layer):
    kinds = ("convolution", "recurrent_state") if (layer + 1) % 4 else ("key", "value")
    return sorted(f"tp{rank}.{kind}" for rank in range(4) for kind in kinds)


def layer_digest(layer, output, states, *, request=None, subnodes=None):
    """Hash each sample's real layer outputs, sub-nodes and all TP-owned state tensors."""
    if (
        layer not in range(48)
        or output.ndim != 3
        or output.shape[1:] != (16, 2048)
        or output.dtype != torch.bfloat16
    ):
        raise ValueError(
            "Layer diagnostics require original BF16 response hidden states [B,16,2048]"
        )
    batch = output.shape[0]
    if sorted(states) != layer_state_names(layer) or any(
        not isinstance(value, torch.Tensor) or value.shape[0] != batch or not value.numel()
        for value in states.values()
    ):
        raise ValueError("Actual convolution/recurrent or KV state tensors are required")
    shapes = (
        native_state_shapes(request)
        if request is not None
        and request["identity"].get("provider_config", {}).get("profile")
        == "qwen3_next_tp4_shared_eager_v1"
        else None
    )
    for name, value in states.items():
        if shapes is not None:
            validate_native_state_storage(value, name.rsplit(".", 1)[-1], shapes)
        if "recurrent_state" in name and value.dtype != torch.float32:
            raise ValueError("Recurrent diagnostic state must preserve FP32")
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite recurrent/KV diagnostic state")
    if not torch.isfinite(output).all():
        raise ValueError("Nonfinite layer diagnostic output")
    if subnodes is None or tuple(subnodes) != SUBNODES:
        raise ValueError(f"Layer diagnostics require the sub-nodes {SUBNODES} in order")
    for name, value in subnodes.items():
        width, dtype = subnode_layout(name)
        if value.dtype != dtype or tuple(value.shape) != (batch, 16, width):
            raise ValueError(f"Sub-node {name} must be original {dtype} [B,16,{width}]")
    subnode_hashes = {
        name: [tensor_digest(value[index]) for index in range(batch)]
        for name, value in subnodes.items()
    }
    outputs = [tensor_digest(output[index]) for index in range(batch)]
    state_hashes = [
        json_digest({name: tensor_digest(value[index]) for name, value in sorted(states.items())})
        for index in range(batch)
    ]
    return {
        "layer": layer,
        "sample_output_sha256": outputs,
        "sample_state_sha256": state_hashes,
        "output_sha256": json_digest({"layer": layer, "samples": outputs}),
        "state_sha256": json_digest({"layer": layer, "samples": state_hashes}),
        "state_names": sorted(states),
        "sample_subnode_sha256": subnode_hashes,
        "subnode_sha256": json_digest({"layer": layer, "subnodes": subnode_hashes}),
    }


def embedding_digest(values):
    """Per-sample digests of the [B, 16, 2048] BF16 embedding rows at the response positions."""
    if values.dtype != torch.bfloat16 or values.ndim != 3 or values.shape[1:] != (16, 2048):
        raise ValueError("Embedding diagnostics require original BF16 [B,16,2048] rows")
    samples = [tensor_digest(values[index]) for index in range(values.shape[0])]
    return {"sample_sha256": samples, "sha256": json_digest({"embedding": samples})}


def first_divergent_node(reference, candidate):
    """The first (layer, sub-node) whose sample digests differ, in execution order.

    Within a layer the sub-nodes are compared in SUBNODES order, then the layer
    output and the recurrent/KV state. Returns ``{"layer", "node"}`` or None.
    """
    batch = candidate["logits"].shape[0]
    left, right = reference["embedding_digest"], candidate["embedding_digest"]
    if left["sample_sha256"][:batch] != right["sample_sha256"]:
        return {"layer": 0, "node": "embedding"}
    for left, right in zip(reference["layer_digests"], candidate["layer_digests"]):
        for node in SUBNODES:
            if left["sample_subnode_sha256"][node][:batch] != right["sample_subnode_sha256"][node]:
                return {"layer": right["layer"], "node": node}
        for node, name in (("output", "sample_output_sha256"), ("state", "sample_state_sha256")):
            if left[name][:batch] != right[name]:
                return {"layer": right["layer"], "node": node}
    return None


def first_divergent_layer(reference, candidate):
    batch = candidate["logits"].shape[0]
    for left, right in zip(reference["layer_digests"], candidate["layer_digests"]):
        for name in ("sample_output_sha256", "sample_state_sha256"):
            if left[name][:batch] != right[name]:
                return right["layer"]
    return None


def validate_model_artifact(artifact, request):
    validate_weight_identity(request["identity"])
    batch = len(request["prompt_tokens"])
    metadata = artifact.get("metadata", {})
    if metadata != {
        key: request[key]
        for key in ("schema_version", "case_id", "role", "identity", "request_sha256")
    }:
        raise ValueError("Model artifact provenance differs from its independent worker request")
    if artifact.get("execution") != request["execution"]:
        raise ValueError("Artifact execution mode does not match its rollout/recompute contract")
    profile = request["identity"].get("provider_config", {})
    if profile.get("profile") == "qwen3_next_tp4_shared_eager_v1" and request["role"] == "rollout":
        validate_native_scheduler_execution(artifact, request)
    expected = {
        "logits": ((batch, RESPONSE_LENGTH, VOCAB_SIZE), torch.bfloat16),
        "logprobs": ((batch, RESPONSE_LENGTH), torch.float32),
        "response_tokens": ((batch, RESPONSE_LENGTH), torch.int64),
        "prompt_tokens": ((batch, len(request["prompt_tokens"][0])), torch.int64),
        "prediction_positions": ((RESPONSE_LENGTH,), torch.int64),
    }
    for name, (shape, dtype) in expected.items():
        tensor = artifact.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(f"{name}: expected CPU {dtype} {shape}, preserving original storage")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{name}: nonfinite full-model artifact")
    if not torch.equal(
        artifact["prompt_tokens"], torch.tensor(request["prompt_tokens"], dtype=torch.int64)
    ) or not torch.equal(
        artifact["prediction_positions"],
        torch.tensor(request["prediction_positions"], dtype=torch.int64),
    ):
        raise ValueError("Prompt or response prediction offsets do not match")
    if request["response_tokens"] is not None and not torch.equal(
        artifact["response_tokens"], torch.tensor(request["response_tokens"], dtype=torch.int64)
    ):
        raise ValueError("Worker changed frozen response tokens")
    if bool(
        ((artifact["response_tokens"] < 0) | (artifact["response_tokens"] >= VOCAB_SIZE)).any()
    ):
        raise ValueError("Generated token is outside the official vocabulary")
    embedding = artifact.get("embedding_digest", {})
    samples = embedding.get("sample_sha256", [])
    if len(samples) != batch or embedding.get("sha256") != json_digest({"embedding": samples}):
        raise ValueError("The artifact must identify every sample's embedding rows")
    layers = artifact.get("layer_digests", [])
    if [layer.get("layer") for layer in layers] != list(range(48)):
        raise ValueError("Full 48-layer diagnostic identities are required")
    for layer in layers:
        if layer.get("state_names") != layer_state_names(layer["layer"]):
            raise ValueError("Per-layer state identities must cover every TP rank")
        for name in ("output", "state"):
            values = layer.get(f"sample_{name}_sha256", [])
            if len(values) != batch:
                raise ValueError("Every layer must identify every sample's output and state")
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in values
            ):
                raise ValueError("Invalid per-sample layer tensor digest")
            if layer.get(f"{name}_sha256") != json_digest(
                {"layer": layer["layer"], "samples": values}
            ):
                raise ValueError("Layer digest is not bound to its sample tensor identities")
        subnodes = layer.get("sample_subnode_sha256", {})
        if tuple(subnodes) != SUBNODES or any(
            len(values) != batch or any(not isinstance(v, str) or len(v) != 64 for v in values)
            for values in subnodes.values()
        ):
            raise ValueError("Every layer must identify every sample's sub-node outputs")
        if layer.get("subnode_sha256") != json_digest(
            {"layer": layer["layer"], "subnodes": subnodes}
        ):
            raise ValueError("Layer sub-node digest is not bound to its sample identities")
    for name, value in artifact.get("debug_tensors", {}).items():
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Debug layer/state tensors must be finite CPU tensors")
        if "recurrent_state" in name and value.dtype != torch.float32:
            raise ValueError("Recurrent state diagnostics must preserve FP32 storage")


def validate_debug_capture(artifact, request):
    tensors = artifact.get("debug_tensors", {})
    for layer in request["trace_layers"]:
        record = artifact["layer_digests"][layer]
        prefix = f"layer.{layer}."
        names = [prefix + "output"] + [prefix + name for name in record["state_names"]]
        names += [f"{prefix}subnode.{name}" for name in SUBNODES]
        if any(name not in tensors for name in names):
            raise ValueError("First-divergence capture is missing layer output or state tensors")
        actual = layer_digest(
            layer,
            tensors[prefix + "output"],
            {name: tensors[prefix + name] for name in record["state_names"]},
            request=request,
            subnodes={name: tensors[f"{prefix}subnode.{name}"] for name in SUBNODES},
        )
        if actual != record:
            raise ValueError("Saved first-divergence tensors differ from their layer identities")


def validate_native_scheduler_execution(artifact, request):
    expected = {
        "kv_cache_memory_bytes_per_gpu": NATIVE_KV_CACHE_BYTES,
        "max_model_len": 1040,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 16384,
        "scheduler": "rl_engine.integrations.qwen3_next_native_rollout.ExactPrefillScheduler",
        "async_scheduling": False,
        "prefix_caching": False,
        "weights_dtype": "bfloat16",
        "recurrent_state_dtype": "float32",
        "model_runner": "v1",
    }
    if artifact.get("native_runtime_profile") != expected:
        raise ValueError("Native rollout runtime differs from its fixed scheduler/cache profile")
    mode = (
        "certified" if "scheduler_profile_certificate" in request["identity"] else "qualification"
    )
    if artifact.get("provider_execution_mode") != mode:
        raise ValueError("Native rollout execution mode differs from its certificate identity")
    if (
        request["identity"]["provider_config"].get("native_rollout_kv_cache_memory_bytes")
        != NATIVE_KV_CACHE_BYTES
    ):
        raise ValueError("The submitted profile did not pin the native cache budget")
    sizes = [*request["prefill_chunks"], *([1] * request["consumed_response_tokens"])]
    steps = artifact.get("native_scheduler_steps", [])
    if len(steps) != len(sizes):
        raise ValueError("Native scheduler did not execute every requested prefill/decode boundary")
    samples = request["sample_ids"]
    consumed = 0
    for step, size in zip(steps, sizes):
        if set(step) != {"samples", "computed_before", "token_counts"} or any(
            (
                sorted(step["samples"]) != sorted(samples),
                len(set(step["samples"])) != len(samples),
                step["computed_before"] != [consumed] * len(samples),
                step["token_counts"] != [size] * len(samples),
            )
        ):
            raise ValueError("Native scheduler executed a different sample/chunk/offset matrix")
        consumed += size


def compare_artifacts(rollout, recompute):
    comparisons = {
        name: tensor_bitwise_equal(rollout[name], recompute[name])
        for name in ("response_tokens", "logits", "logprobs")
    }
    first_layer = first_divergent_layer(rollout, recompute)
    first_node = first_divergent_node(rollout, recompute)
    comparisons["layer_states"] = first_node is None
    first_difference = None
    for name, passed in comparisons.items():
        if passed:
            continue
        if name == "layer_states":
            break
        left, right = rollout[name], recompute[name]
        if left.shape == right.shape and left.dtype == right.dtype:
            width = left.element_size()
            changed = (
                left.contiguous().reshape(-1).view(torch.uint8)
                != right.contiguous().reshape(-1).view(torch.uint8)
            ).nonzero()
            if changed.numel():
                flat = int(changed[0, 0]) // width
                first_difference = {
                    "tensor": name,
                    "flat_element": flat,
                    "byte_offset": int(changed[0, 0]),
                }
        break
    return {
        "passed": all(comparisons.values()),
        "comparisons": comparisons,
        "first_divergent_layer": first_layer,
        "first_divergent_node": first_node,
        "first_difference": first_difference,
    }


def native_state_shapes(request):
    """Complete canonical per-rank state after P prompt plus 15 response inputs.

    TP4GDN owns 2048 convolution channels with three prior samples and eight
    128x128 recurrent heads. Full attention owns one 256-wide KV head per rank;
    the paired replicas remain independently recorded and compared.
    """
    prompts = request["prompt_tokens"]
    batch = len(prompts)
    if batch not in (1, 4) or any(len(row) != len(prompts[0]) for row in prompts):
        raise ValueError("Native state evidence requires the fixed equal-length batch1/4 matrix")
    prompt_length = len(prompts[0])
    consumed = prompt_length + RESPONSE_LENGTH - 1
    # L2 requests are pinned to PROMPT_LENGTHS by their own validators; batch-invariance
    # cells use any prompt that fits the rollout profile (1024 + 16 = max_model_len).
    if (
        not 1 <= prompt_length <= 1024
        or request.get("consumed_response_tokens") != RESPONSE_LENGTH - 1
        or request.get("state_position") != consumed - 1
    ):
        raise ValueError("Native state position must consume the prompt and exactly 15 responses")
    return {
        "convolution": (batch, 2048, 3),
        "recurrent_state": (batch, 8, 128, 128),
        "key": (batch, 1, consumed, 256),
        "value": (batch, 1, consumed, 256),
    }


def validate_native_state_storage(value, name, expected_shapes):
    """One dtype/shape/finite boundary shared by assembly and stored trace audits."""
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shapes[name]:
        raise ValueError(
            f"Original worker must retain complete {name} state shape "
            f"{expected_shapes[name]} after the consumed prompt/response prefix"
        )
    dtype = torch.float32 if name == "recurrent_state" else torch.bfloat16
    if value.dtype != dtype or not torch.isfinite(value).all():
        raise ValueError("Original worker state violates BF16/FP32 finite storage")


def validate_native_worker_traces(artifact, request, *, artifact_root=None):
    """Rehash four actual traces and reconstruct every layer and state digest."""
    profile = request["identity"].get("provider_config", {}).get("profile")
    if profile != "qwen3_next_tp4_shared_eager_v1":
        return
    records = artifact.get("worker_traces", [])
    if len(records) != 4 or sorted(record.get("rank", -1) for record in records) != [0, 1, 2, 3]:
        raise ValueError("Native model needs one original trace from each TP rank")
    binary = artifact.get("provider_binary", {})
    digest = binary.get("sha256", "")
    if (
        binary.get("module") != "rl_engine._C"
        or not isinstance(binary.get("path"), str)
        or not Path(binary["path"]).is_absolute()
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Native model must identify the actual imported CUDA provider binary")
    batch = len(request["sample_ids"])
    expected_state_shapes = native_state_shapes(request)
    state_hashes = [[{} for _ in range(batch)] for _ in range(48)]
    for record in records:
        rank = record["rank"]
        if record.get("request_sha256") != request["request_sha256"]:
            raise ValueError("Worker trace request identity mismatch")
        trace = load_tensor_artifact(checked_artifact_path(record, artifact_root=artifact_root))
        provenance = {
            "rank": rank,
            "request_sha256": request["request_sha256"],
            "provider_binary": binary,
        }
        if request["role"] == "rollout":
            provenance.update(
                native_runtime_profile=artifact["native_runtime_profile"],
                provider_execution_mode=artifact["provider_execution_mode"],
                schedule=artifact["native_scheduler_steps"],
            )
        else:
            provenance.update(role="recompute", execution=request["execution"])
        if any(trace.get(key) != value for key, value in provenance.items()):
            raise ValueError("Worker trace rank/runtime/scheduler provenance mismatch")
        tensors = {
            "logprobs": artifact["logprobs"],
            "responses": artifact["response_tokens"],
            "local_logits": artifact["logits"][..., rank * 37984 : (rank + 1) * 37984],
        }
        # Recompute keeps the gathered full logits on rank zero, and every rank
        # retains its original local shard and independently selected logprobs.
        if request["role"] == "rollout" or rank == 0:
            tensors["logits"] = artifact["logits"]
        for key, expected in tensors.items():
            if not isinstance(trace.get(key), torch.Tensor) or not tensor_bitwise_equal(
                trace[key], expected
            ):
                raise ValueError(f"Original worker {key} bytes contradict the assembled artifact")
        if request["role"] == "recompute":
            layers = trace.get("layers", {})
            if not isinstance(layers, dict) or set(layers) != {*range(48), "model"}:
                raise ValueError("Original recompute worker must retain all 48 layer traces")
            outputs = [layers[layer].get("output") for layer in range(48)]
            if any(
                not isinstance(value, torch.Tensor)
                or value.shape != (batch, 16, 2048)
                or value.dtype != torch.bfloat16
                for value in outputs
            ):
                raise ValueError("Original recompute worker must retain all 48 BF16 layer outputs")
            hidden = torch.stack(outputs)
            states = [layers[layer].get("states", {}) for layer in range(48)]
            subnodes = {
                name: torch.stack([layers[layer]["subnodes"][name] for layer in range(48)])
                for name in SUBNODES
            }
            embedding = layers["model"]["embedding"]
        else:
            hidden = trace.get("hidden")
            states = trace.get("states", [])
            subnodes, embedding = trace.get("subnodes", {}), trace.get("embedding")
        if embedding_digest(embedding) != artifact["embedding_digest"]:
            raise ValueError("Original worker embedding rows contradict the assembled artifact")
        for name in SUBNODES:
            for layer in range(48):
                if [
                    tensor_digest(subnodes[name][layer, sample]) for sample in range(batch)
                ] != artifact["layer_digests"][layer]["sample_subnode_sha256"][name]:
                    raise ValueError(
                        f"Original worker {name} rows contradict the assembled artifact"
                    )
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.shape != (48, batch, 16, 2048)
            or hidden.dtype != torch.bfloat16
        ):
            raise ValueError("Original worker must retain all 48 BF16 layer outputs")
        if not torch.isfinite(hidden).all() or len(states) != 48:
            raise ValueError("Original worker has nonfinite outputs or missing layer states")
        for layer in range(48):
            expected_names = (
                ("key", "value") if (layer + 1) % 4 == 0 else ("convolution", "recurrent_state")
            )
            if set(states[layer]) != set(expected_names):
                raise ValueError("Worker trace state types differ from the official layer layout")
            for sample in range(batch):
                if (
                    tensor_digest(hidden[layer, sample])
                    != artifact["layer_digests"][layer]["sample_output_sha256"][sample]
                ):
                    raise ValueError("Original worker layer output contradicts its declared digest")
            for name, value in states[layer].items():
                validate_native_state_storage(value, name, expected_state_shapes)
                for sample in range(batch):
                    state_hashes[layer][sample][f"tp{rank}.{name}"] = tensor_digest(value[sample])
    for layer in range(48):
        if [json_digest(values) for values in state_hashes[layer]] != artifact["layer_digests"][
            layer
        ]["sample_state_sha256"]:
            raise ValueError("Original TP state bytes contradict the assembled layer digests")


def compare_batch_prefill(reference, candidate):
    batch = candidate["logits"].shape[0]
    compared = {
        name: tensor_bitwise_equal(reference[name][:batch], candidate[name])
        for name in ("response_tokens", "logits", "logprobs")
    }
    compared["layer_states"] = first_divergent_node(reference, candidate) is None
    return compared


def validate_matrix_report(report, *, expected_identity=None, artifact_root=None):
    """A passed label alone cannot close the full-model matrix."""
    expected_ids = [case.case_id for case in case_matrix()]
    required = {
        "schema_version": 1,
        "status": "passed",
        "scope": "qwen3_next_full_model_l2",
        "raw_bit_comparison": True,
        "logits_dtype": "bfloat16",
        "logprobs_dtype": "float32",
        "response_length": 16,
        "vocab_size": 151936,
        "case_ids": expected_ids,
    }
    if any(report.get(key) != value for key, value in required.items()) or set(
        report.get("cases", {})
    ) != set(expected_ids):
        raise ValueError("Full-model L2 requires the complete sixteen-case raw-bit matrix")
    identity = report.get("identity", {})
    if identity.get("architecture_layers") != 48 or identity.get("topology") != {
        "tp": 4,
        "pp": 1,
        "cp": 1,
        "ep": 1,
    }:
        raise ValueError("Full-model L2 cannot change model depth or topology")
    processes = report.get("processes", {})
    if (
        set(processes) != {"rollout", "recompute"}
        or any(
            row.get("exit_code") != 0
            or row.get("gpu_release_verified") is not True
            or type(row.get("pid")) is not int
            or row["pid"] <= 0
            for row in processes.values()
        )
        or processes["rollout"]["pid"] == processes["recompute"]["pid"]
    ):
        raise ValueError("Independent process completion and GPU release evidence are required")
    checks = {"response_tokens", "logits", "logprobs", "layer_states"}
    for case in report["cases"].values():
        compared = case.get("comparisons", {})
        if (
            case.get("passed") is not True
            or set(compared) != checks
            or any(value is not True for value in compared.values())
        ):
            raise ValueError(
                "Every response token, full logits, logprob and layer/state identity must pass"
            )
        groups = [case.get("batch_prefill", {})]
        if identity.get("weight_version") == "baseline":
            groups.append(case.get("baseline_generation", {}))
        for group in groups:
            if set(group) != {"rollout", "recompute"} or any(
                set(row) != checks or any(value is not True for value in row.values())
                for row in group.values()
            ):
                raise ValueError(
                    "Batch, chunk and actual baseline generation comparisons are mandatory"
                )
        if set(case.get("artifacts", {})) != {"rollout", "recompute"}:
            raise ValueError("Both independent full-vocabulary tensor artifacts are required")
    _validate_report_artifacts(
        report,
        roles=("rollout", "recompute"),
        expected_identity=expected_identity,
        artifact_root=artifact_root,
    )


def checked_artifact_path(record, *, artifact_root=None):
    """Rehash bytes before parsing; artifact paths cannot escape this attempt."""
    if not isinstance(record, dict) or not {"path", "sha256"} <= record.keys():
        raise ValueError("Tensor evidence requires a path and SHA256")
    path = Path(record["path"])
    if not path.is_absolute():
        raise ValueError("Tensor artifact path must be absolute")
    path = path.resolve(strict=True)
    if artifact_root is not None and not path.is_relative_to(Path(artifact_root).resolve()):
        raise ValueError("Tensor artifact escaped its current attempt")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record["sha256"]:
        raise ValueError("Tensor artifact content digest changed")
    return path


def _validate_report_artifacts(report, *, roles, expected_identity, artifact_root):
    identity = report["identity"]
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Model matrix source/environment/provider identity mismatch")
    from rl_engine.testing.qwen3_next_binary import validate_binary_record

    binary = validate_binary_record(
        report.get("provider_binary"),
        source_tree=identity.get("source_tree"),
        artifact_root=artifact_root,
    )
    # Inputs may be the explicit frozen baseline shared by subsequent training
    # runs. Their bytes are pinned separately; outputs must be in this attempt.
    fixture = load_tensor_artifact(checked_artifact_path(report.get("fixture")))
    validate_fixture(fixture)
    if fixture["content_sha256"] != report["fixture"].get("content_sha256"):
        raise ValueError("Matrix frozen-input content identity mismatch")
    baselines = {}
    for length, record in fixture["baseline_artifacts"].items():
        baseline = load_tensor_artifact(checked_artifact_path(record))
        original_identity = baseline.get("metadata", {}).get("identity", {})
        if json_digest(original_identity) != fixture["baseline_identity"]:
            raise ValueError("Baseline generation provenance changed")
        if identity.get("weight_version") == "baseline" and original_identity != identity:
            raise ValueError("Baseline weights/provider differ from the matrix")
        baseline_request = worker_request(
            L2Case(int(length), 4, "full"),
            fixture,
            role="rollout",
            identity=original_identity,
            generate=True,
        )
        validate_model_artifact(baseline, baseline_request)
        validate_native_worker_traces(baseline, baseline_request)
        baselines[length] = baseline
    worker_records = report.get("worker_results", {})
    if set(worker_records) != set(roles):
        raise ValueError("Independent worker result artifacts are missing")
    workers = {}
    for role in roles:
        worker = json.loads(
            checked_artifact_path(
                worker_records[role],
                artifact_root=artifact_root,
            ).read_text()
        )
        if any(
            worker.get(key) != value
            for key, value in {
                "status": "passed",
                "role": role,
                "identity": identity,
                "fixture_sha256": fixture["content_sha256"],
                "pid": report["processes"][role]["pid"],
                "provider_binary": binary,
            }.items()
        ) or set(worker.get("cases", {})) != {case.case_id for case in case_matrix()}:
            raise ValueError("Independent worker provenance or case inventory mismatch")
        workers[role] = worker
    for case in case_matrix():
        row = report["cases"][case.case_id]
        outputs = {}
        for role in roles:
            record = row["artifacts"][role]
            path = checked_artifact_path(record, artifact_root=artifact_root)
            request = worker_request(case, fixture, role=role, identity=identity)
            worker_record = workers[role]["cases"][case.case_id]
            if worker_record != {**record, "request_sha256": request["request_sha256"]}:
                raise ValueError("Worker and matrix refer to different model output artifacts")
            outputs[role] = load_tensor_artifact(path)
            if outputs[role].get("provider_binary") != binary:
                raise ValueError("Model output used a different native CUDA provider")
            validate_model_artifact(outputs[role], request)
            validate_native_worker_traces(outputs[role], request, artifact_root=artifact_root)
            reference_record = report["cases"][L2Case(case.prompt_length, 4, "full").case_id][
                "artifacts"
            ][role]
            reference = load_tensor_artifact(
                checked_artifact_path(reference_record, artifact_root=artifact_root)
            )
            actual = compare_batch_prefill(reference, outputs[role])
            if not all(actual.values()) or actual != row["batch_prefill"][role]:
                raise ValueError("Batch/prefill tensor bytes contradict matrix acceptance")
            if identity.get("weight_version") == "baseline":
                actual = compare_batch_prefill(baselines[str(case.prompt_length)], outputs[role])
                if not all(actual.values()) or actual != row["baseline_generation"][role]:
                    raise ValueError("Generated baseline tensor bytes contradict matrix acceptance")
        if len(roles) == 2:
            actual = compare_artifacts(outputs["rollout"], outputs["recompute"])
            if not actual["passed"] or actual["comparisons"] != row["comparisons"]:
                raise ValueError("Independent full-model tensor bytes contradict matrix acceptance")


def validate_scheduler_report(report, *, expected_identity=None, artifact_root=None):
    expected_ids = [case.case_id for case in case_matrix()]
    required = {
        "schema_version": 1,
        "status": "passed",
        "raw_bit_comparison": True,
        "scope": "qwen3_next_native_scheduler_only",
        "case_ids": expected_ids,
        "full_model_l2": "not_evaluated",
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("Actual scheduler requires its complete native sixteen-case matrix")
    if set(report.get("cases", {})) != set(expected_ids):
        raise ValueError("Scheduler matrix omitted mandatory cases")
    identity = report.get("identity", {})
    if identity.get("architecture_layers") != 48 or identity.get("topology") != {
        "tp": 4,
        "pp": 1,
        "cp": 1,
        "ep": 1,
    }:
        raise ValueError("Scheduler gate must execute the complete TP4 model")
    processes = report.get("processes", {})
    process = processes.get("rollout", {})
    if set(processes) != {"rollout"} or any(
        (
            process.get("exit_code") != 0,
            process.get("gpu_release_verified") is not True,
            type(process.get("pid")) is not int,
            process.get("pid", 0) <= 0,
        )
    ):
        raise ValueError("Scheduler process completion and memory release evidence missing")
    checks = {"response_tokens", "logits", "logprobs", "layer_states"}
    for row in report["cases"].values():
        if row.get("passed") is not True or set(row.get("artifacts", {})) != {"rollout"}:
            raise ValueError("Scheduler case failed or has no native tensor artifact")
        for field in ("batch_prefill", "baseline_generation"):
            group = row.get(field, {})
            values = group.get("rollout", {})
            if any(
                (
                    set(group) != {"rollout"},
                    set(values) != checks,
                    not all(value is True for value in values.values()),
                )
            ):
                raise ValueError("Scheduler batch/prefill/raw-bit comparisons are incomplete")
    _validate_report_artifacts(
        report,
        roles=("rollout",),
        expected_identity=expected_identity,
        artifact_root=artifact_root,
    )
