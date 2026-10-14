# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Independent CPU FP32 Linear reference for the H3 video input head."""

import torch.nn.functional as F

from rl_engine.ops.gemm.video_input_projection_fp32 import validate_projection_inputs


class NativeVideoInputProjectionFP32Op:
    def forward_fp32(self, x, weight, bias=None, *, logical_keys=None, parameter_id=None):
        validate_projection_inputs(x, weight, bias)
        # CPU avoids a GPU TF32 or batch-dependent cuBLAS reference path.
        return F.linear(x.cpu(), weight.cpu(), None if bias is None else bias.cpu()).to(x.device)

    __call__ = forward_fp32
