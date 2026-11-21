# SPDX-License-Identifier: Apache-2.0
"""Synthetic dense reward for the training run (T2) on a layer-truncated model.

A DeepSeek-V4-Flash cut to a few layers cannot solve DAPO math, so the math reward is 0 for
every sample, every GRPO group has identical rewards, the advantages are 0 and the policy
gradient vanishes: 200 "training" steps would keep the weights fixed and prove nothing about
bitwise alignment after updates. This reward is the fraction of ASCII digits in the response:
it differs between the samples of a group, so advantages and gradients are nonzero and the
weights really move every step. It measures nothing about quality.
"""


def compute_score(data_source=None, solution_str="", ground_truth=None, extra_info=None, **kwargs):
    text = solution_str or ""
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)
