# SPDX-License-Identifier: Apache-2.0
"""Preserve the native DSV4 W4A16 recipe when loading a gfx950 rollout.

The pinned vLLM auto selector prefers AITER on gfx950. AITER can quantize
activations depending on batch size, whereas the native actor retains BF16
activations. Select the upstream unfused W4A16 implementation for this explicit
loading scope; leave model configuration and checkpoint tensors unchanged.
"""

from contextlib import contextmanager

import torch


@contextmanager
def native_w4a16_loading():
    if not torch.version.hip or not torch.cuda.is_available():
        yield
        return
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    if arch != "gfx950":
        yield
        return

    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        Mxfp4MoEKernelOracle,
    )
    from vllm.model_executor.layers.quantization import mxfp4
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp4Static

    original = mxfp4.select_deepseek_v4_mxfp4_moe_backend

    def select(config):
        if config.routing_method != RoutingMethodType.DeepseekV4 or config.moe_backend != "auto":
            return original(config)
        # Use vLLM's support checks and packed-weight conversion. Fail if the
        # pinned implementation cannot satisfy this deployment; no fallback.
        return Mxfp4MoEKernelOracle(config, kMxfp4Static, None).return_or_raise(
            Mxfp4MoeBackend.TRITON_UNFUSED, activation_key=None
        )

    mxfp4.select_deepseek_v4_mxfp4_moe_backend = select
    try:
        yield
    finally:
        mxfp4.select_deepseek_v4_mxfp4_moe_backend = original
