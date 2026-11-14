# SPDX-License-Identifier: Apache-2.0
"""Read the pinned TrainingWorker metric ABI without substituting defaults."""

import math


def training_metrics(output):
    metrics = output["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError(f"locked TrainingWorker metrics must be dict, got {type(metrics)}")
    norm = metrics["grad_norm"]
    if not math.isfinite(float(norm)) or float(norm) <= 0:
        raise RuntimeError("actual optimizer gradient norm must be finite and nonzero")
    return metrics
