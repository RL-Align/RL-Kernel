# SPDX-License-Identifier: Apache-2.0
"""Resource isolation must reject forbidden GPUs before probing a GPU runtime."""

import copy

import pytest
from examples.operators.dsv41_4l.gpu_visibility import host_mapping, validate_mapping


def example_mapping():
    return {
        "physical_gpus": [4, 5, 6, 7],
        "rocr_visible_devices": "4,5,6,7",
        "mapping": [
            {"logical_gpu": i, "physical_gpu": i + 4, "pci_bus_id": pci}
            for i, pci in enumerate(
                ("0000:a3:00.0", "0000:ab:00.0", "0000:b3:00.0", "0000:bb:00.0")
            )
        ],
    }


@pytest.mark.parametrize("selection", ["0", "0,1,2,3", "0,1,2,3,4,5,6,7", "4,4", "5,4"])
def test_invalid_gpu_selection_never_queries_devices(selection, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid GPU policy reached subprocess/device query")

    monkeypatch.setattr("subprocess.check_output", forbidden)
    with pytest.raises(ValueError):
        host_mapping(selection)


def test_pci_mapping_detects_reordered_or_incomplete_visibility():
    expected = example_mapping()
    observed = [
        {"logical_gpu": d["logical_gpu"], "pci_bus_id": d["pci_bus_id"]}
        for d in expected["mapping"]
    ]
    env = {"ROCR_VISIBLE_DEVICES": "4,5,6,7"}
    validate_mapping(expected, observed, env)
    swapped = copy.deepcopy(observed)
    swapped[0]["pci_bus_id"], swapped[1]["pci_bus_id"] = (
        swapped[1]["pci_bus_id"],
        swapped[0]["pci_bus_id"],
    )
    with pytest.raises(RuntimeError, match="PCI mapping mismatch"):
        validate_mapping(expected, swapped, env)
    with pytest.raises(RuntimeError, match="count differs"):
        validate_mapping(expected, observed[:3], env)


@pytest.mark.parametrize(
    "name",
    ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "HSA_VISIBLE_DEVICES"],
)
def test_second_stage_filters_are_rejected_including_empty_values(name):
    expected = example_mapping()
    observed = [
        {"logical_gpu": d["logical_gpu"], "pci_bus_id": d["pci_bus_id"]}
        for d in expected["mapping"]
    ]
    with pytest.raises(RuntimeError, match="must be unset"):
        validate_mapping(expected, observed, {"ROCR_VISIBLE_DEVICES": "4,5,6,7", name: ""})
