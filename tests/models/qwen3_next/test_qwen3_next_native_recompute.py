# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from pathlib import Path

import pytest
import torch

from rl_engine.integrations.orchestrators.vime.experiments.qwen3_next.native_recompute import (
    response_rows,
)
from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime_runtime import runtime_argv
from rl_engine.models.qwen3_next.qwen3_next_subnode import SUBNODES, subnode_layout


@pytest.mark.parametrize("batch", [1, 4])
def test_response_rows_start_at_last_prompt_and_do_not_consume_last_target(batch):
    request = {
        "prompt_tokens": [[1] * 8 for _ in range(batch)],
        "consumed_input_tokens": [[1] * 23 for _ in range(batch)],
        "prediction_positions": list(range(7, 23)),
    }
    hidden = torch.arange(batch * 23 * 2).reshape(batch * 23, 1, 2).to(torch.bfloat16)
    selected = response_rows(hidden, request)
    assert selected.dtype == torch.bfloat16 and selected.shape == (batch, 16, 2)
    for index in range(batch):
        assert torch.equal(selected[index], hidden[index * 23 + 7 : (index + 1) * 23, 0])
    with pytest.raises(ValueError, match="BF16"):
        response_rows(hidden.float(), request)
    request["prediction_positions"] = list(range(8, 24))
    with pytest.raises(ValueError, match="last target"):
        response_rows(hidden, request)


def test_native_cli_has_full_model_vocabulary_topology_and_no_precision_override():
    argv = runtime_argv("/checkpoint", training=True)
    for name, value in {
        "num-layers": 48,
        "tensor-model-parallel-size": 4,
        "expert-tensor-parallel-size": 4,
        "pipeline-model-parallel-size": 1,
        "context-parallel-size": 1,
        "padded-vocab-size": 151936,
    }.items():
        assert argv[argv.index("--" + name) + 1] == str(value)
    assert "--optimization-dtype" not in argv and "--reduce-dtype" not in argv
    assert "--sequence-parallel" not in argv
    assert "--use-rollout-logprobs" in argv


def test_recompute_batch_loads_once_and_keeps_verifiable_original_rank_traces(
    tmp_path, monkeypatch
):
    import copy
    import hashlib

    from rl_engine.integrations.orchestrators.vime.experiments.qwen3_next import (
        native_recompute as recompute,
    )
    from rl_engine.validation.models.qwen3_next.qwen3_next_l2 import (
        load_tensor_artifact,
        save_tensor_artifact,
        validate_native_worker_traces,
    )

    request = {
        "schema_version": 1,
        "case_id": "cpu-contract-only",
        "role": "recompute",
        "identity": {
            "provider_config": {"profile": "qwen3_next_tp4_shared_eager_v1"},
            "weight_version": "baseline",
        },
        "request_sha256": "a" * 64,
        "execution": {"recompute_mode": "full_from_tokens", "input_sequence_length": 23},
        "prompt_tokens": [[1] * 8],
        "response_tokens": [[2] * 16],
        "prediction_positions": list(range(7, 23)),
        "sample_ids": [0],
        "consumed_response_tokens": 15,
        "state_position": 22,
        "trace_layers": [],
    }
    second = {**request, "case_id": "second-cpu-case", "request_sha256": "b" * 64}
    calls = []

    def spawn(target, *, args, nprocs, join):
        cases, staging = args
        calls.append((target, nprocs, join))
        assert len(cases) == 2
        for index, (current, _output) in enumerate(cases):
            for rank in range(4):
                layers = {}
                for layer in range(48):
                    names = (
                        ("key", "value")
                        if (layer + 1) % 4 == 0
                        else ("convolution", "recurrent_state")
                    )
                    layers[layer] = {
                        "output": torch.zeros((1, 16, 2048), dtype=torch.bfloat16),
                        "subnodes": {
                            name: torch.zeros(
                                (1, 16, subnode_layout(name)[0]), dtype=subnode_layout(name)[1]
                            )
                            for name in SUBNODES
                        },
                        "states": {
                            name: torch.zeros(
                                (),
                                dtype=(
                                    torch.float32 if name == "recurrent_state" else torch.bfloat16
                                ),
                            ).expand(
                                {
                                    "convolution": (1, 2048, 3),
                                    "recurrent_state": (1, 8, 128, 128),
                                    "key": (1, 1, 23, 256),
                                    "value": (1, 1, 23, 256),
                                }[name]
                            )
                            for name in names
                        },
                    }
                layers["model"] = {"embedding": torch.zeros((1, 16, 2048), dtype=torch.bfloat16)}
                trace = {
                    "rank": rank,
                    "request_sha256": current["request_sha256"],
                    "role": "recompute",
                    "execution": current["execution"],
                    "provider_binary": {
                        "module": "rl_engine._C",
                        "path": "/cpu-test-only/_C.so",
                        "sha256": "c" * 64,
                    },
                    "layers": layers,
                    "local_logits": torch.zeros((), dtype=torch.bfloat16).expand(1, 16, 37984),
                    "logprobs": torch.zeros((1, 16)),
                    "responses": torch.tensor(current["response_tokens"], dtype=torch.int64),
                }
                if rank == 0:
                    trace["logits"] = torch.zeros((), dtype=torch.bfloat16).expand(1, 16, 151936)
                save_tensor_artifact(Path(staging) / f"case-{index}-rank-{rank}.pt", trace)

    monkeypatch.setattr(recompute.mp, "spawn", spawn)
    outputs = [(request, tmp_path / "first.pt"), (second, tmp_path / "second.pt")]
    recompute.run_recompute_cases(iter(outputs))
    assert calls == [(recompute._rank_main, 4, True)]
    for current, path in outputs:
        artifact = load_tensor_artifact(path)
        assert len(artifact["worker_traces"]) == 4
        validate_native_worker_traces(artifact, current, artifact_root=tmp_path)
    artifact = load_tensor_artifact(outputs[0][1])
    for record in artifact["worker_traces"]:
        assert Path(record["path"]).is_file()

    missing_binary = copy.deepcopy(artifact)
    missing_binary.pop("provider_binary")
    with pytest.raises(ValueError, match="imported CUDA provider binary"):
        validate_native_worker_traces(missing_binary, request, artifact_root=tmp_path)

    # Merely retaining files is insufficient: even after rehashing the record,
    # original hidden/state/local-logit bytes must reproduce the declared result.
    record = artifact["worker_traces"][1]
    original = load_tensor_artifact(record["path"])
    for field, message in (
        ("state", "state bytes"),
        ("hidden", "layer output"),
        ("local_logits", "local_logits bytes"),
        ("binary", "provenance mismatch"),
        ("convolution_truncated", "complete convolution state shape"),
        ("convolution_transposed", "complete convolution state shape"),
        ("recurrent_truncated", "complete recurrent_state state shape"),
        ("key_short", "complete key state shape"),
        ("key_long", "complete key state shape"),
        ("value_truncated", "complete value state shape"),
    ):
        trace = copy.deepcopy(original)
        if field == "state":
            value = trace["layers"][0]["states"]["recurrent_state"].clone()
            value[0, 0, 0, 0] = -0.0
            trace["layers"][0]["states"]["recurrent_state"] = value
        elif field == "convolution_truncated":
            trace["layers"][0]["states"]["convolution"] = torch.zeros((1, 1), dtype=torch.bfloat16)
        elif field == "convolution_transposed":
            value = trace["layers"][0]["states"]["convolution"]
            trace["layers"][0]["states"]["convolution"] = value.transpose(1, 2)
        elif field == "recurrent_truncated":
            value = trace["layers"][0]["states"]["recurrent_state"]
            trace["layers"][0]["states"]["recurrent_state"] = value[:, :7]
        elif field in ("key_short", "key_long"):
            length = 22 if field == "key_short" else 24
            trace["layers"][3]["states"]["key"] = torch.zeros(
                (1, 1, length, 256), dtype=torch.bfloat16
            )
        elif field == "value_truncated":
            value = trace["layers"][3]["states"]["value"]
            trace["layers"][3]["states"]["value"] = value[..., :255]
        elif field == "hidden":
            trace["layers"][0]["output"][0, 0, 0] = -0.0
        elif field == "binary":
            trace["provider_binary"]["sha256"] = "d" * 64
        else:
            trace["local_logits"] = trace["local_logits"].clone()
            trace["local_logits"][0, 0, 0] = -0.0
        save_tensor_artifact(record["path"], trace)
        with open(record["path"], "rb") as stream:
            record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        with pytest.raises(ValueError, match=message):
            validate_native_worker_traces(artifact, request, artifact_root=tmp_path)
        if "complete " in message:
            ranks = [load_tensor_artifact(item["path"]) for item in artifact["worker_traces"]]
            with pytest.raises(ValueError, match=message):
                recompute._assemble(request, ranks, artifact["worker_traces"])


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("prompt_length", [8, 64, 256, 1024])
def test_complete_native_state_shapes_bind_every_consumed_token(batch, prompt_length):
    from rl_engine.validation.models.qwen3_next.qwen3_next_l2 import native_state_shapes

    request = {
        "prompt_tokens": [[1] * prompt_length for _ in range(batch)],
        "consumed_response_tokens": 15,
        "state_position": prompt_length + 14,
    }
    assert native_state_shapes(request) == {
        "convolution": (batch, 2048, 3),
        "recurrent_state": (batch, 8, 128, 128),
        "key": (batch, 1, prompt_length + 15, 256),
        "value": (batch, 1, prompt_length + 15, 256),
    }
    with pytest.raises(ValueError, match="exactly 15 responses"):
        native_state_shapes({**request, "state_position": prompt_length + 15})
    with pytest.raises(ValueError, match="exactly 15 responses"):
        native_state_shapes({**request, "consumed_response_tokens": 16})
