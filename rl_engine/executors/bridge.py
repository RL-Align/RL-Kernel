# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch
from typing import Dict, Any, cast
from rl_engine.utils.logger import logger


class IPCWeightBridge:
    """
    This implementation bridges the memory sharing between training and inference processes.
    It leverages the CUDA IPC mechanism to achieve zero-copy weight synchronization,
    significantly reducing communication overhead.
       - Training process: Exports IPC handles for model weights.
       - Inference process: Reconstructs Tensors based on these handles,
       directly accessing the training process's memory.
    Suitable for RL inference scenarios such as vLLM that require frequent weight updates,
    ensuring efficient collaboration between training and inference.
    """

    def __init__(self):
        # Mapping of storage parameter names to IPC handles
        self.handle_registry: Dict[str, Any] = {}

    def export_model_handles(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        The training process calls:
        Iterates through the model parameters and generates a cross-process
        readable IPC handle for each Tensor.
        """
        logger.info("Exporting model weights via CUDA IPC...")
        ipc_handles = {}

        for name, param in model.named_parameters():
            data = param.data.detach()
            # Calling PyTorch's shared memory interface
            shared_storage = data.storage()._share_cuda_()
            ipc_handles[name] = {
                "handle": shared_storage,
                "shape": data.shape,
                "dtype": data.dtype,
                "stride": data.stride(),
            }

        self.handle_registry = ipc_handles
        return ipc_handles

    def import_model_weights(self, ipc_handles: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        The inference process calls:
        Reconstructs Tensors based on the received handles,
        directly accessing the training process's memory.
        """
        logger.info("Importing model weights from IPC handles...")
        remote_weights = {}

        for name, info in ipc_handles.items():
            cuda_float_tensor = cast(Any, torch.cuda).FloatTensor
            storage = cuda_float_tensor._new_shared_cuda(info["handle"])
            tensor = cuda_float_tensor(storage).view(info["shape"])
            remote_weights[name] = tensor

        return remote_weights
