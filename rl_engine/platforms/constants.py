# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from enum import Enum


class DeviceType(Enum):
    CUDA = "cuda"  # NVIDIA GPU
    ROCM = "rocm"  # AMD GPU
    NPU = "npu"  # Huawei Ascend
    TPU = "tpu"  # Google TPU
    XPU = "xpu"  # Intel GPU
    CPU = "cpu"


class BackendLib(Enum):
    FLASHINFER = "flashinfer"
    AITER = "aiter"
    TRITON = "triton"
    CANN = "cann"  # Huawei NPU
    NATIVE = "native"  # PyTorch Default


class PrecisionType(Enum):
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"


class SamplingMethod(Enum):
    TOP_K = "top_k"
    TOP_P = "top_p"
    TOP_K_TOP_P = "top_k_top_p"
    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    NUCLEUS = "nucleus"


class MemoryFormat(Enum):
    CONTIGUOUS = "contiguous"
    CHANNELS_LAST = "channels_last"
    PRESERVE = "preserve"


class OperatorFusionLevel(Enum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class KernelOptimizationLevel(Enum):
    DEFAULT = "default"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class LoggingLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProfilingMode(Enum):
    OFF = "off"
    BASIC = "basic"
    DETAILED = "detailed"
    CUSTOM = "custom"


class DistributedStrategy(Enum):
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"


class CheckpointFormat(Enum):
    PT = "pt"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    CUSTOM = "custom"


class ActivationFunction(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    TANH = "tanh"
    SIGMOID = "sigmoid"


class Constants:
    def __init__(self):
        self.DeviceType = DeviceType
        self.BackendLib = BackendLib
        self.PrecisionType = PrecisionType
        self.SamplingMethod = SamplingMethod
        self.MemoryFormat = MemoryFormat
        self.OperatorFusionLevel = OperatorFusionLevel
        self.KernelOptimizationLevel = KernelOptimizationLevel
        self.LoggingLevel = LoggingLevel
        self.ProfilingMode = ProfilingMode
        self.DistributedStrategy = DistributedStrategy
        self.CheckpointFormat = CheckpointFormat
        self.ActivationFunction = ActivationFunction


constants = Constants()
