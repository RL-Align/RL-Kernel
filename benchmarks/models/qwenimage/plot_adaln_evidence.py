#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot public-operator medians from an adaln_evidence JSON report."""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("output")
    args = parser.parse_args()
    plt.switch_backend("Agg")
    with open(args.report) as handle:
        report = json.load(handle)
    cases = [
        c
        for c in report["cases"]
        if c["dtype"] == "torch.bfloat16" and c["batch"] == 1 and c["seq"] in (1, 6889)
    ]
    labels = [f'{c["stream"]}\nB={c["batch"]}, S={c["seq"]}' for c in cases]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    positions = np.arange(len(cases))
    for axis, mode, title in zip(
        axes,
        ("forward", "backward", "forward_backward"),
        ("Forward", "Backward only", "Forward + backward"),
    ):
        for offset, name, label in [
            (-0.18, "native", "Native PyTorch"),
            (0.18, "candidate", "Strict CUDA"),
        ]:
            values = [c["timing"][mode]["median_ms"][name] for c in cases]
            bars = axis.bar(positions + offset, values, width=0.36, label=label)
            axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        axis.set_xticks(positions, labels)
        axis.set_title(title)
        axis.set_ylabel("Median latency (ms)")
        axis.margins(y=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Qwen-Image AdaLN • B200 • BF16 • real checkpoint weights")
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
