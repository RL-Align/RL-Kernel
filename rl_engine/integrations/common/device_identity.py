# SPDX-License-Identifier: Apache-2.0
"""Read the actual selected HIP device identity, independent of visibility names."""

import ctypes

import torch


def hip_device_identity():
    """HIP ordinals can be remapped; PCI identity proves worker device separation."""
    logical = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(logical)
    hip = ctypes.CDLL("libamdhip64.so")
    query = hip.hipDeviceGetPCIBusId
    query.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(64)
    status = query(buffer, len(buffer), logical)
    if status:
        raise RuntimeError(f"hipDeviceGetPCIBusId failed: {status}")
    return dict(
        logical_gpu=logical,
        pci_bus_id=buffer.value.decode().lower(),
        architecture=props.gcnArchName,
        device_name=props.name,
    )
