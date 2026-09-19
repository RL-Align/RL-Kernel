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

_KERNELS = {}
_SOURCE = r'''
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
        for(int limb=0;limb<3;++limb) {
            unsigned long long v=(square>>(16*limb)) & 65535ULL;
            if(v) atomicAdd(bins+e*3+limb,v);
        }
    }
    __syncthreads();
    for (int j=threadIdx.x;j<768;j+=blockDim.x)
        if(bins[j]) atomicAdd(out+j,bins[j]);
}
'''


def _checked(code, operation):
    if code != 0:
        raise RuntimeError(f'{operation} failed with status {code}')


def _kernel(device):
    device=torch.device(device)
    key=device.index
    if key in _KERNELS:
        return _KERNELS[key]
    hip=torch.version.hip is not None
    if hip:
        rtc=C.CDLL('libhiprtc.so'); driver=C.CDLL('libamdhip64.so')
        prefix='hiprtc'; module_prefix='hip'
        arch=torch.cuda.get_device_properties(device).gcnArchName.split(':')[0]
        source='#include <hip/hip_runtime.h>\n'+_SOURCE
    else:
        libs=glob.glob(str(Path(torch.__file__).parent.parent/'nvidia/cuda_nvrtc/lib/libnvrtc.so*'))
        rtc=C.CDLL(sorted(libs,key=len)[0] if libs else 'libnvrtc.so.12')
        driver=C.CDLL('libcuda.so.1');prefix='nvrtc';module_prefix='cu'
        major,minor=torch.cuda.get_device_capability(device)
        arch=f'compute_{major}{minor}';source=_SOURCE
    program=C.c_void_p()
    _checked(getattr(rtc,prefix+'CreateProgram')(C.byref(program),source.encode(),b'rlk_norm.cu',0,None,None),'create norm program')
    options=(C.c_char_p*1)(f'--gpu-architecture={arch}'.encode())
    status=getattr(rtc,prefix+'CompileProgram')(program,1,options)
    if status:
        size=C.c_size_t()
        getattr(rtc,prefix+'GetProgramLogSize')(program,C.byref(size))
        log=C.create_string_buffer(size.value)
        getattr(rtc,prefix+'GetProgramLog')(program,log)
        raise RuntimeError('exact norm compile failed: '+log.value.decode())
    size=C.c_size_t()
    getter='Code' if hip else 'PTX'
    _checked(getattr(rtc,prefix+'Get'+getter+'Size')(program,C.byref(size)),'get norm code size')
    code=C.create_string_buffer(size.value)
    _checked(getattr(rtc,prefix+'Get'+getter)(program,code),'get norm code')
    getattr(rtc,prefix+'DestroyProgram')(C.byref(program))
    module=C.c_void_p();function=C.c_void_p()
    _checked(getattr(driver,module_prefix+'ModuleLoadData')(C.byref(module),code),'load norm module')
    _checked(getattr(driver,module_prefix+'ModuleGetFunction')(C.byref(function),module,b'bin_squares'),'load norm kernel')
    result=(driver,module,function,module_prefix)
    _KERNELS[key]=result
    return result


def integer_square_bins(values):
    """Accumulate exact FP32 squares on the tensors' accelerator device."""
    if not values:
        return torch.zeros(770,dtype=torch.int64,device='cuda')
    if any(value.device != values[0].device for value in values):
        raise ValueError('exact gradient norm tensors must share a device')
    with torch.cuda.device(values[0].device):
        return _integer_square_bins(values)


def _integer_square_bins(values):
    device=values[0].device
    result=torch.zeros(770,dtype=torch.int64,device=device)
    driver,module,function,prefix=_kernel(device)
    # A limb contributes at most 65535 per value; the global caller checks
    # the count before integer reduction, ruling out signed int64 overflow.
    for value in values:
        if not value.numel():continue
        value=value.detach().float().contiguous()
        x=C.c_void_p(value.data_ptr());n=C.c_longlong(value.numel());out=C.c_void_p(result.data_ptr())
        params=(C.c_void_p*3)(C.cast(C.byref(x),C.c_void_p),C.cast(C.byref(n),C.c_void_p),C.cast(C.byref(out),C.c_void_p))
        grid=min(1024,(value.numel()+1023)//1024)
        stream=C.c_void_p(torch.cuda.current_stream(device).cuda_stream)
        _checked(getattr(driver,prefix+'LaunchKernel' if prefix=='cu' else 'hipModuleLaunchKernel')(
            function,grid,1,1,256,1,1,0,stream,params,None),'exact norm launch')
    return result


def norm_from_bins(bins):
    if bins[768]:return float('nan')
    if bins[769]:return float('inf')
    total=0
    for e in range(255):
        mantissa=bins[e*3]+(bins[e*3+1]<<16)+(bins[e*3+2]<<32)
        total+=mantissa << (2*(max(e,1)-1))
    return math.sqrt(math.ldexp(float(total),-298))


def install():
    from megatron.core.optimizer import clip_grads,optimizer
    if getattr(clip_grads,'_rlk_exact_norm',False):return
    original=clip_grads.get_grad_norm_fp32
    def get_norm(grads_for_norm,norm_type=2,grad_stats_parallel_group=None):
        if float(norm_type)!=2:
            return original(grads_for_norm,norm_type,grad_stats_parallel_group)
        values=[grads_for_norm] if isinstance(grads_for_norm,torch.Tensor) else list(grads_for_norm)
        if any(not isinstance(v,torch.Tensor) or not v.is_cuda for v in values):
            raise TypeError('exact gradient norm requires accelerator tensors')
        group=grad_stats_parallel_group
        world=torch.distributed.get_world_size(group)
        if sum(v.numel() for v in values)*world >= (1<<63)//65535:
            raise OverflowError('exact gradient norm integer accumulator capacity exceeded')
        bins=integer_square_bins(values)
        torch.distributed.all_reduce(bins,group=group)
        return norm_from_bins(bins.cpu().tolist())
    clip_grads.get_grad_norm_fp32=get_norm
    optimizer.get_grad_norm_fp32=get_norm
    clip_grads._rlk_exact_norm=True
