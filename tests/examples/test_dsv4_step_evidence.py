# SPDX-License-Identifier: Apache-2.0
import pytest
from tensordict import NonTensorData, TensorDict
from examples.operators.dsv4_4l.step_evidence import training_metrics


def test_real_tensordict_unwraps_training_metrics_to_dict():
    result = TensorDict(
        {"metrics": NonTensorData({"grad_norm": 279.73, "loss": 1.2})}, batch_size=[]
    )
    assert isinstance(result["metrics"], dict)
    assert training_metrics(result)["grad_norm"] == 279.73


@pytest.mark.parametrize("norm", [0, float("nan"), float("inf")])
def test_missing_or_invalid_gradient_norm_is_rejected(norm):
    with pytest.raises(RuntimeError):
        training_metrics({"metrics": {"grad_norm": norm}})
    with pytest.raises(KeyError):
        training_metrics({"metrics": {}})
