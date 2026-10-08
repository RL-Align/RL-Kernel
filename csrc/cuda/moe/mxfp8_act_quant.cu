// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// MXFP8 activation quantization (P5-1) — CUDA backend.
//
// Bit-exact with rl_engine.moe.mx_format.mx_quantize(x, "e4m3") and therefore
// with the P5 oracle (numeric profile "oracle-fp32-serial-v1"):
//
//   block  = 32 elements along the last dim (row-local; never crosses a row)
//   amax   = max(|x|) over the block
//   code   = clamp(floor(log2(max(amax, FLT_MIN))) - 8, -127, 127) + 127,
//            with amax == 0 mapped to code 127 (scale 1.0)
//   scale  = 2^(code - 127)                     (exact, built from bit patterns)
//   elem   = cvt.rn.satfinite.e4m3(clamp(x / scale, -448, 448))
//
// floor(log2(.)) is read off the FP32 exponent field after the FLT_MIN clamp,
// which is the exact integer equivalent of the oracle's frexp path. Every step
// that could see a subnormal (|x| max, the FLT_MIN clamp, the scale, the
// division) is done on integer bit patterns or through explicit non-.ftz PTX,
// because --use_fast_math (KERNEL_ALIGN_USE_FAST_MATH=1 in setup.py) compiles
// fmaxf/fabsf/__fdiv_rn to their .ftz forms and would flush the subnormal
// scale 2^-127 and subnormal inputs. So neither fast-math nor libm rounding
// can move a single output byte.
//
// Determinism: max is order-independent over a fixed 32-element window and
// every element is transformed independently, so the emitted bytes do not
// depend on grid shape, block size, or how many rows are in flight (Axis-A
// batch invariance holds bitwise).

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <type_traits>

// fp16 / bf16 / fp32 only: the vectorized loader has no fp64 specialization and
// the P5 contract never feeds fp64 activations.
#define DISPATCH_ACT_QUANT_TYPES(TYPE, NAME, ...)              \
  AT_DISPATCH_SWITCH(                                          \
      TYPE,                                                    \
      NAME,                                                    \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)     \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)      \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

namespace {

constexpr int MX_BLOCK = 32;
constexpr int E8M0_BIAS = 127;
constexpr int EMAX_E4M3 = 8;
constexpr float E4M3_MAX = 448.0f;
// 2^-127 is subnormal in FP32; its bit pattern cannot be built from the
// exponent field alone.
constexpr unsigned int BITS_2POW_M127 = 0x00400000u;

template <typename T>
__device__ __forceinline__ float bits16_to_float(unsigned short bits);

template <>
__device__ __forceinline__ float bits16_to_float<at::BFloat16>(unsigned short bits) {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16);  // exact
}

template <>
__device__ __forceinline__ float bits16_to_float<at::Half>(unsigned short bits) {
  return __half2float(__ushort_as_half(bits));
}

// VEC == 1 is the unaligned scalar fallback; otherwise one 16-byte load
// (4 floats or 8 bf16/half).
template <typename scalar_t, int VEC>
__device__ __forceinline__ void load_vec(const scalar_t* __restrict__ p, float* out) {
  if constexpr (VEC == 1) {
    out[0] = static_cast<float>(p[0]);
  } else if constexpr (std::is_same_v<scalar_t, float>) {
    static_assert(VEC == 4, "16-byte chunk");
    const float4 v = *reinterpret_cast<const float4*>(p);
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
  } else {
    static_assert(VEC == 8, "16-byte chunk");
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    const unsigned int words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      out[2 * i] = bits16_to_float<scalar_t>(static_cast<unsigned short>(words[i] & 0xFFFFu));
      out[2 * i + 1] = bits16_to_float<scalar_t>(static_cast<unsigned short>(words[i] >> 16));
    }
  }
}

constexpr unsigned int ABS_MASK = 0x7FFFFFFFu;
constexpr unsigned int EXP_MASK = 0x7F800000u;
constexpr unsigned int FLT_MIN_NORMAL_BITS = 0x00800000u;  // 2^-126

// |x| as a bit pattern. For non-negative floats the unsigned bit pattern is
// monotone in the value, so the block amax is an integer max: no .ftz
// instruction is involved and subnormal inputs survive --use_fast_math.
__device__ __forceinline__ unsigned int abs_bits(float x) {
  return __float_as_uint(x) & ABS_MASK;
}

__device__ __forceinline__ bool is_nonfinite_bits(unsigned int abs) {
  return (abs & EXP_MASK) == EXP_MASK;
}

// IEEE round-to-nearest division with the non-.ftz opcode spelled out; the
// __fdiv_rn intrinsic turns into div.rn.ftz.f32 under --use_fast_math.
__device__ __forceinline__ float div_rn_no_ftz(float a, float b) {
  float r;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// E8M0 shared-scale code for one block, matching e8m0_scale_from_amax().
// Integer arithmetic on the amax bit pattern throughout.
__device__ __forceinline__ unsigned char e8m0_code_from_amax_bits(unsigned int amax_bits) {
  if (amax_bits == 0u) {
    return static_cast<unsigned char>(E8M0_BIAS);
  }
  // Clamp to FLT_MIN first (the oracle does the same), so the value is normal
  // and its biased exponent field is exactly floor(log2(amax)) + 127.
  const unsigned int a = max(amax_bits, FLT_MIN_NORMAL_BITS);
  const int floor_log2 = static_cast<int>(a >> 23) - E8M0_BIAS;
  int shared_exp = floor_log2 - EMAX_E4M3;
  shared_exp = max(-E8M0_BIAS, min(E8M0_BIAS, shared_exp));
  return static_cast<unsigned char>(shared_exp + E8M0_BIAS);
}

__device__ __forceinline__ float e8m0_decode(unsigned char code) {
  const unsigned int bits =
      (code > 0) ? (static_cast<unsigned int>(code) << 23) : BITS_2POW_M127;
  return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned char quantize_e4m3(float x, float scale) {
  float scaled = div_rn_no_ftz(x, scale);
  // fminf/fmaxf may flush a subnormal `scaled` under fast-math; that is
  // harmless here because E4M3 rounds anything below 2^-10 to zero anyway.
  scaled = fminf(fmaxf(scaled, -E4M3_MAX), E4M3_MAX);
  return static_cast<unsigned char>(
      __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3));
}

// One thread owns VEC contiguous elements and issues VEC / CHUNK 16-byte
// loads. When VEC == MX_BLOCK a thread owns a whole block and no cross-lane
// reduction is needed at all; smaller VEC values reduce across MX_BLOCK / VEC
// lanes with xor shuffles inside that lane group.
template <typename scalar_t, int VEC>
__global__ void mxfp8_act_quant_forward_kernel(
    const scalar_t* __restrict__ x,
    unsigned char* __restrict__ codes,
    unsigned char* __restrict__ scales,
    int* __restrict__ nonfinite_flag,
    const int64_t n_groups) {
  static_assert(VEC <= MX_BLOCK, "a thread must not span more than one MX block");
  constexpr int CHUNK = 16 / sizeof(scalar_t);  // elements per 16-byte load
  constexpr int LANES_PER_BLOCK = MX_BLOCK / VEC;
  const int64_t gid = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  const bool active = gid < n_groups;

  float v[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    v[i] = 0.0f;
  }
  if (active) {
    if constexpr (VEC >= CHUNK) {
#pragma unroll
      for (int c = 0; c < VEC / CHUNK; ++c) {
        load_vec<scalar_t, CHUNK>(x + gid * VEC + c * CHUNK, v + c * CHUNK);
      }
    } else {
      load_vec<scalar_t, VEC>(x + gid * VEC, v);
    }
  }

  unsigned int amax_bits = 0u;
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    amax_bits = max(amax_bits, abs_bits(v[i]));
  }

  // Every lane of the group ends up with the block amax; the whole warp takes
  // part in the shuffles, so inactive tail lanes stay well defined.
#pragma unroll
  for (int offset = 1; offset < LANES_PER_BLOCK; offset <<= 1) {
    amax_bits = max(amax_bits, __shfl_xor_sync(0xFFFFFFFFu, amax_bits, offset, LANES_PER_BLOCK));
  }

  if (!active) {
    return;
  }
  // inf/NaN patterns are >= 0x7F800000, so the block's largest |x| pattern
  // says whether any element is non-finite.
  if (is_nonfinite_bits(amax_bits)) {
    atomicOr(nonfinite_flag, 1);
  }

  const unsigned char code = e8m0_code_from_amax_bits(amax_bits);
  const float scale = e8m0_decode(code);

  unsigned char out[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    out[i] = quantize_e4m3(v[i], scale);
  }
  // Widest store the element count allows: 16 bytes, 8, or single bytes on
  // the unaligned fallback path.
  if constexpr (VEC % 16 == 0) {
#pragma unroll
    for (int c = 0; c < VEC / 16; ++c) {
      *reinterpret_cast<uint4*>(codes + gid * VEC + c * 16) =
          *reinterpret_cast<const uint4*>(out + c * 16);
    }
  } else if constexpr (VEC == 8) {
    *reinterpret_cast<uint2*>(codes + gid * VEC) = *reinterpret_cast<const uint2*>(out);
  } else {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      codes[gid * VEC + i] = out[i];
    }
  }
  if (gid % LANES_PER_BLOCK == 0) {
    scales[gid / LANES_PER_BLOCK] = code;
  }
}

template <typename scalar_t>
__global__ void ste_copy_kernel(
    const scalar_t* __restrict__ dy, scalar_t* __restrict__ dx, const int64_t n) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dx[idx] = dy[idx];
}

// 16 bytes per thread; the STE is a pure copy, so it should run at copy speed.
__global__ void ste_copy_vec16_kernel(
    const uint4* __restrict__ dy, uint4* __restrict__ dx, const int64_t n_vec) {
  const int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx >= n_vec) {
    return;
  }
  dx[idx] = dy[idx];
}

void check_act_quant_input(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(
      x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16 ||
          x.scalar_type() == at::kFloat,
      "x must be fp16, bf16, or fp32, got ",
      x.scalar_type());
  TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
  TORCH_CHECK(
      x.size(-1) % MX_BLOCK == 0,
      "last dim ",
      x.size(-1),
      " is not divisible by the MX block size ",
      MX_BLOCK);
}

bool is_aligned(const void* ptr, size_t bytes) {
  return (reinterpret_cast<uintptr_t>(ptr) % bytes) == 0;
}

}  // namespace

// Returns {codes uint8 [..., K], scales uint8 [..., K/32], nonfinite int32 [1]}.
// The flag is reported instead of raised so the caller decides when to pay for
// the device sync that the fail-closed contract needs; when the caller will not
// read it (check_finite == false) it is left uninitialized to skip the memset.
std::vector<torch::Tensor> mxfp8_act_quant_forward_cuda(torch::Tensor x, bool check_finite) {
  check_act_quant_input(x);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

  auto scale_sizes = x.sizes().vec();
  scale_sizes.back() = scale_sizes.back() / MX_BLOCK;
  auto codes = torch::empty(x.sizes(), x.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_sizes, x.options().dtype(torch::kUInt8));
  auto nonfinite = check_finite ? torch::zeros({1}, x.options().dtype(torch::kInt32))
                                : torch::empty({1}, x.options().dtype(torch::kInt32));

  const int64_t n = x.numel();
  if (n == 0) {
    return {codes, scales, nonfinite};
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;

  DISPATCH_ACT_QUANT_TYPES(x.scalar_type(), "mxfp8_act_quant_forward", [&] {
    const scalar_t* x_ptr = x.data_ptr<scalar_t>();
    unsigned char* codes_ptr = codes.data_ptr<unsigned char>();
    unsigned char* scales_ptr = scales.data_ptr<unsigned char>();
    int* flag_ptr = nonfinite.data_ptr<int>();
    // A thread owns one whole MX block for bf16/fp16 (4 x 16-byte loads, no
    // shuffle) and 8 elements for fp32 (2 x 16-byte loads). Every access is
    // at most 16 bytes wide, so 16-byte alignment is all the vector path
    // needs; a storage offset that breaks it falls back to the scalar path,
    // which emits identical bytes.
    constexpr int kVec = sizeof(scalar_t) == 2 ? 32 : 8;
    const bool vectorizable = is_aligned(x_ptr, 16) && is_aligned(codes_ptr, 16);
    if (vectorizable) {
      const int64_t n_groups = n / kVec;
      const int64_t blocks = (n_groups + threads - 1) / threads;
      mxfp8_act_quant_forward_kernel<scalar_t, kVec><<<blocks, threads, 0, stream>>>(
          x_ptr, codes_ptr, scales_ptr, flag_ptr, n_groups);
    } else {
      const int64_t blocks = (n + threads - 1) / threads;
      mxfp8_act_quant_forward_kernel<scalar_t, 1><<<blocks, threads, 0, stream>>>(
          x_ptr, codes_ptr, scales_ptr, flag_ptr, n);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes, scales, nonfinite};
}

// Straight-through estimator: dX = dY (dtype preserved, contiguous copy).
torch::Tensor mxfp8_act_quant_ste_backward_cuda(torch::Tensor dy) {
  TORCH_CHECK(dy.is_cuda(), "dy must be a CUDA tensor");
  TORCH_CHECK(dy.is_contiguous(), "dy must be contiguous");
  // The STE is an identity on any floating dtype (P5-1 spec contract table:
  // dy/dx "any float"); the oracle's backward hands it FP32 accumulators.
  TORCH_CHECK(dy.is_floating_point(), "dy must be a floating-point tensor, got ", dy.scalar_type());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(dy));
  auto dx = torch::empty_like(dy);
  const int64_t n = dy.numel();
  if (n == 0) {
    return dx;
  }
  const int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n_bytes = n * dy.element_size();
  if (n_bytes % 16 == 0 && is_aligned(dy.data_ptr(), 16) && is_aligned(dx.data_ptr(), 16)) {
    const int64_t n_vec = n_bytes / 16;
    const int64_t blocks = (n_vec + threads - 1) / threads;
    ste_copy_vec16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(dy.data_ptr()),
        reinterpret_cast<uint4*>(dx.data_ptr()),
        n_vec);
  } else {
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dy.scalar_type(),
        "mxfp8_act_quant_ste_backward",
        [&] {
          ste_copy_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              dy.data_ptr<scalar_t>(), dx.data_ptr<scalar_t>(), n);
        });
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dx;
}

// Standalone pybind module for the JIT path (see
// rl_engine/kernels/ops/cuda/moe/mxfp8_act_quant.py). The ahead-of-time build
// binds these symbols from csrc/ops.cpp instead, so the macro stays off there.
#ifdef RL_KERNEL_P5_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "P5-1 MXFP8 activation quantization (standalone JIT build)";
  m.def("mxfp8_act_quant_forward", &mxfp8_act_quant_forward_cuda,
        "MXFP8 (E4M3 + block-32 E8M0) activation quantization; "
        "returns {codes, scales, nonfinite_flag}",
        py::arg("x"), py::arg("check_finite") = true);
  m.def("mxfp8_act_quant_ste_backward", &mxfp8_act_quant_ste_backward_cuda,
        "Straight-through estimator backward for mxfp8_act_quant (dX = dY)");
}
#endif
