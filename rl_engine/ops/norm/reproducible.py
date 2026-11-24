# SPDX-License-Identifier: Apache-2.0
"""Partition-independent L2 norm using an exact integer sum of FP32 squares.

Each FP32 significand squared has at most 48 bits. Three 16-bit limbs are
summed in exponent bins with integer atomics, then reduced across ranks.
No floating-point reduction order enters the result. This also controls
gradient clipping, rather than changing the printed metric.
"""
import ctypes as C
import glob
import math
from pathlib import Path

import torch

_KERNELS: dict[int | None, tuple[C.CDLL, C.c_void_p, C.c_void_p, str]] = {}
_SOURCE = r"""
extern "C" __global__ void bin_squares(const float* x, long long n,
                                      unsigned long long* out) {
    __shared__ unsigned long long bins[768];
    for (int j=threadIdx.x; j<768; j+=blockDim.x) bins[j]=0;
    __syncthreads();
    for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
         i<n; i+=(long long)gridDim.x*blockDim.x) {
        unsigned int bits=__float_as_uint(x[i]) & 0x7fffffffU;
        unsigned int e=bits>>23, m=bits & 0x7fffffU;
        if (e==255) { atomicOr(out+768+(m==0),1ULL); continue; }
        if (e) m|=0x800000U;
        unsigned long long square=(unsigned long long)m*m;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // Integer limb sums are exact. Aggregate equal exponent bins within
        // each active warp before the shared atomic, avoiding contention.
        const unsigned int peers = __match_any_sync(__activemask(), e);
        const bool leader = (threadIdx.x & 31) == __ffs(peers) - 1;
        for (int limb=0;limb<3;++limb) {
            const unsigned int v=(square>>(16*limb)) & 65535ULL;
            if (__popc(peers) > 2) {
                const unsigned int sum=__reduce_add_sync(peers,v);
                if (leader && sum) atomicAdd(bins+e*3+limb,(unsigned long long)sum);
            } else if (v) {
                atomicAdd(bins+e*3+limb,(unsigned long long)v);
            }
        }
#else
        for(int limb=0;limb<3;++limb) {
            unsigned long long v=(square>>(16*limb)) & 65535ULL;
            if(v) atomicAdd(bins+e*3+limb,v);
        }
#endif
    }
    __syncthreads();
    for (int j=threadIdx.x;j<768;j+=blockDim.x)
        if(bins[j]) atomicAdd(out+j,bins[j]);
}
"""


def _checked(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation} failed with status {code}")


def _nvrtc_candidates(site=None, cuda_version=None):
    """NVRTC libraries to try, best match for torch's CUDA build first.

    CUDA 12 wheels ship ``nvidia/cuda_nvrtc/lib/libnvrtc.so.12``; CUDA 13
    wheels ship ``nvidia/cu13/lib/libnvrtc.so.13``. After the wheel copies
    come the system sonames for torch's CUDA major, then the unversioned name.
    """
    if site is None:
        site = Path(torch.__file__).parent.parent / "nvidia"
    if cuda_version is None:
        cuda_version = torch.version.cuda
    major = (cuda_version or "").split(".")[0]
    wheel_dirs = [site / "cuda_nvrtc" / "lib"]
    if major:
        wheel_dirs.insert(0, site / f"cu{major}" / "lib")
    wheel_dirs += sorted(Path(p) for p in glob.glob(str(site / "cu[0-9]*" / "lib")))
    candidates = []
    for directory in wheel_dirs:
        candidates += sorted(glob.glob(str(directory / "libnvrtc.so*")), key=len)
    if major:
        candidates.append(f"libnvrtc.so.{major}")
    candidates += ["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"]
    return list(dict.fromkeys(candidates))


def _load_nvrtc():
    errors = []
    for name in _nvrtc_candidates():
        try:
            return C.CDLL(name)
        except OSError as error:
            errors.append(f"{name}: {error}")
    raise OSError("could not load NVRTC; tried:\n  " + "\n  ".join(errors))


def _kernel(device):
    device = torch.device(device)
    key = device.index
    if key in _KERNELS:
        return _KERNELS[key]
    hip = torch.version.hip is not None
    if hip:
        rtc = C.CDLL("libhiprtc.so")
        driver = C.CDLL("libamdhip64.so")
        prefix = "hiprtc"
        module_prefix = "hip"
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        source = "#include <hip/hip_runtime.h>\n" + _SOURCE
    else:
        rtc = _load_nvrtc()
        driver = C.CDLL("libcuda.so.1")
        prefix = "nvrtc"
        module_prefix = "cu"
        major, minor = torch.cuda.get_device_capability(device)
        arch = f"compute_{major}{minor}"
        source = _SOURCE
    program = C.c_void_p()
    _checked(
        getattr(rtc, prefix + "CreateProgram")(
            C.byref(program), source.encode(), b"rlk_norm.cu", 0, None, None
        ),
        "create norm program",
    )
    options = (C.c_char_p * 1)(f"--gpu-architecture={arch}".encode())
    status = getattr(rtc, prefix + "CompileProgram")(program, 1, options)
    if status:
        size = C.c_size_t()
        getattr(rtc, prefix + "GetProgramLogSize")(program, C.byref(size))
        log = C.create_string_buffer(size.value)
        getattr(rtc, prefix + "GetProgramLog")(program, log)
        raise RuntimeError("exact norm compile failed: " + log.value.decode())
    size = C.c_size_t()
    getter = "Code" if hip else "PTX"
    _checked(
        getattr(rtc, prefix + "Get" + getter + "Size")(program, C.byref(size)), "get norm code size"
    )
    code = C.create_string_buffer(size.value)
    _checked(getattr(rtc, prefix + "Get" + getter)(program, code), "get norm code")
    getattr(rtc, prefix + "DestroyProgram")(C.byref(program))
    module = C.c_void_p()
    function = C.c_void_p()
    _checked(
        getattr(driver, module_prefix + "ModuleLoadData")(C.byref(module), code), "load norm module"
    )
    _checked(
        getattr(driver, module_prefix + "ModuleGetFunction")(
            C.byref(function), module, b"bin_squares"
        ),
        "load norm kernel",
    )
    result = (driver, module, function, module_prefix)
    _KERNELS[key] = result
    return result


def integer_square_bins(values):
    """Accumulate exact FP32 squares on the tensors' accelerator device."""
    if not values:
        return torch.zeros(770, dtype=torch.int64, device="cuda")
    if any(value.device != values[0].device for value in values):
        raise ValueError("exact gradient norm tensors must share a device")
    with torch.cuda.device(values[0].device):
        return _integer_square_bins(values)


def _integer_square_bins(values):
    device = values[0].device
    result = torch.zeros(770, dtype=torch.int64, device=device)
    driver, module, function, prefix = _kernel(device)
    # A limb contributes at most 65535 per value; the global caller checks
    # the count before integer reduction, ruling out signed int64 overflow.
    for value in values:
        if not value.numel():
            continue
        value = value.detach().float().contiguous()
        x = C.c_void_p(value.data_ptr())
        n = C.c_longlong(value.numel())
        out = C.c_void_p(result.data_ptr())
        params = (C.c_void_p * 3)(
            C.cast(C.byref(x), C.c_void_p),
            C.cast(C.byref(n), C.c_void_p),
            C.cast(C.byref(out), C.c_void_p),
        )
        grid = min(1024, (value.numel() + 1023) // 1024)
        stream = C.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        _checked(
            getattr(driver, prefix + "LaunchKernel" if prefix == "cu" else "hipModuleLaunchKernel")(
                function, grid, 1, 1, 256, 1, 1, 0, stream, params, None
            ),
            "exact norm launch",
        )
    return result


def norm_from_bins(bins):
    if bins[768]:
        return float("nan")
    if bins[769]:
        return float("inf")
    total = 0
    for e in range(255):
        mantissa = bins[e * 3] + (bins[e * 3 + 1] << 16) + (bins[e * 3 + 2] << 32)
        total += mantissa << (2 * (max(e, 1) - 1))
    return math.sqrt(math.ldexp(float(total), -298))


def exact_norm(values, group=None) -> float:
    """L2 norm of all elements of ``values`` over the ranks of ``group``, bit-identical on each.

    Each rank passes the tensors it *counts*: a tensor held by several ranks (a replicated
    parameter, or a gradient every context-parallel rank holds in full) must be passed by
    exactly one of them. The squares are binned as integers and the bins are summed with an
    integer all-reduce, so neither the partition of the tensors nor the reduction order of the
    collective changes the result. Without an initialised process group, or with a group of
    one rank, this is the norm of the local tensors.
    """

    values = [values] if isinstance(values, torch.Tensor) else list(values)
    if any(not isinstance(v, torch.Tensor) or not v.is_cuda for v in values):
        raise TypeError("exact gradient norm requires accelerator tensors")
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world = torch.distributed.get_world_size(group) if distributed else 1
    if sum(v.numel() for v in values) * world >= (1 << 63) // 65535:
        raise OverflowError("exact gradient norm integer accumulator capacity exceeded")
    bins = integer_square_bins(values)
    if world > 1:
        torch.distributed.all_reduce(bins, group=group)
    return norm_from_bins(bins.cpu().tolist())


def clip_grad_norm_exact_(parameters, max_norm: float, *, counted=None, group=None) -> float:
    """``torch.nn.utils.clip_grad_norm_`` with the exact norm of :func:`exact_norm`.

    ``parameters`` is a Tensor or an iterable of them. ``counted(p)`` selects the parameters whose
    gradients enter the norm on this rank (default: all of them); use it so that each replicated
    parameter is counted once across ``group``. Every gradient is then scaled by
    ``clamp(max_norm / (norm + 1e-6), max=1)`` computed in FP32, as PyTorch does, and multiplied at
    FP32 whatever the gradients' dtype. A non-finite norm leaves the gradients unchanged and is
    returned, so that the caller can skip the step (verl's FSDP2 clip does the same).
    """

    if isinstance(parameters, torch.Tensor):  # one parameter, as torch.nn.utils.clip_grad_norm_
        parameters = [parameters]
    params = [p for p in parameters if p.grad is not None]
    keep = params if counted is None else [p for p in params if counted(p)]
    total = exact_norm([p.grad for p in keep], group)
    if not math.isfinite(total):
        return total
    coef = torch.clamp(
        torch.tensor(max_norm, dtype=torch.float32)
        / (torch.tensor(total, dtype=torch.float32) + 1e-6),
        max=1.0,
    )
    # The FP32 coefficient is passed as a Python scalar, so the kernels multiply at FP32 for FP16
    # and BF16 gradients too. PyTorch's clip passes a 0-d tensor, which type promotion casts to
    # the gradients' dtype: a coefficient below FP16's range then zeroes every gradient. For FP32
    # gradients the two agree. One foreach multiply per (device, dtype) group, no device copy.
    scale = float(coef)
    groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for p in params:
        groups.setdefault((p.grad.device, p.grad.dtype), []).append(p.grad)
    for grads in groups.values():
        torch._foreach_mul_(grads, scale)
    return total


def install():
    from megatron.core.optimizer import clip_grads, optimizer

    if getattr(clip_grads, "_rlk_exact_norm", False):
        return
    original = clip_grads.get_grad_norm_fp32

    def get_norm(grads_for_norm, norm_type=2, grad_stats_parallel_group=None):
        if float(norm_type) != 2:
            return original(grads_for_norm, norm_type, grad_stats_parallel_group)
        return exact_norm(grads_for_norm, grad_stats_parallel_group)

    clip_grads.get_grad_norm_fp32 = get_norm
    optimizer.get_grad_norm_fp32 = get_norm
    clip_grads._rlk_exact_norm = True
