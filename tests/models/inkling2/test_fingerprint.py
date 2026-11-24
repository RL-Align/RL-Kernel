# SPDX-License-Identifier: Apache-2.0
import copy
import json

import pytest

from rl_engine.models.inkling2.fingerprint import (
    HF_REVISION,
    LOOP,
    TEXT,
    snapshot_fingerprint,
    validate_config,
)


def config():
    return {
        "dtype": "bfloat16",
        "model_type": "inkling2_loop",
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "text_config": {**copy.deepcopy(TEXT), "loop_config": dict(LOOP)},
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }


def sidecar(converted=False):
    groups = {}
    for layer in range(24):
        dtype = "bfloat16" if converted else ("nvfp4" if layer < 20 else "mxfp8")
        for suffix in ("w13_weight", "w2_weight"):
            groups.setdefault(dtype, []).append(f"model.llm.layers.{layer}.mlp.experts.{suffix}")
    return {
        "model": {
            "parameter_groups": [
                {"compute_dtype": dtype, "tensors": tensors} for dtype, tensors in groups.items()
            ]
        }
    }


def test_released_config_and_hash(tmp_path):
    c = config()
    (tmp_path / "config.json").write_text(json.dumps(c))
    q = tmp_path / "hf_quant_config.json"
    q.write_text(json.dumps(sidecar()))
    a = snapshot_fingerprint(tmp_path, revision=HF_REVISION)
    q.write_text(json.dumps(sidecar(), indent=2))
    b = snapshot_fingerprint(tmp_path, revision=HF_REVISION)
    assert a["logical_visits"] == 48 and a["physical_layers"] == 24
    assert a["quant_metadata_sha256"] != b["quant_metadata_sha256"]


def test_reject_shrunk_or_variable_loop():
    c = config()
    c["text_config"]["loop_config"]["num_passes"] = 2
    with pytest.raises(ValueError, match="fixed-loop"):
        validate_config(c, revision=HF_REVISION)
    with pytest.raises(ValueError, match="revision"):
        validate_config(config(), revision="main")
    c = config()
    c["text_config"]["hidden_size"] = 64
    with pytest.raises(ValueError, match="hidden_size"):
        validate_config(c, revision=HF_REVISION)


def test_bf16_profile_is_distinct(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(config()))
    q = tmp_path / "hf_quant_config.json"
    q.write_text(json.dumps(sidecar()))
    with pytest.raises(ValueError, match="expert precision"):
        snapshot_fingerprint(tmp_path, revision=HF_REVISION, converted=True)
    q.write_text(json.dumps(sidecar(converted=True)))
    assert snapshot_fingerprint(tmp_path, revision=HF_REVISION, converted=True)["profile"] == (
        "bf16-dequant"
    )
