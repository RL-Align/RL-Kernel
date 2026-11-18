# SPDX-License-Identifier: Apache-2.0
"""Native FP8 encoding for the supported CDNA checkpoint execution paths.

MI300 (gfx942) uses FNUZ; MI350/MI355 (gfx950) uses OCP E4M3FN.
See https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
"""

import torch


def fp8_dtype(device):
    if not torch.version.hip or torch.device(device).type != "cuda":
        raise RuntimeError("native FP8 execution requires a ROCm GPU")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch == "gfx942":
        return torch.float8_e4m3fnuz
    if arch == "gfx950":
        return torch.float8_e4m3fn
    raise ValueError(f"native FP8 execution requires gfx942/gfx950, got {arch}")
