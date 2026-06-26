// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// det_gemm-local TMA primitives.

#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>

namespace det_gemm {

using nv_bf16 = __nv_bfloat16;

// CTA-scoped
__device__ __forceinline__ void mbar_init(uint32_t addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(addr),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, int phase) {
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "LAB_W:\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, 10000000;\n"
      "@!P bra.uni LAB_W;\n"
      "}" ::"r"(addr),
      "r"(phase));
}

// TMA 2D global -> shared (shared::cluster dst space)
__device__ __forceinline__ void tma_2d_g2s(uint32_t dst_smem, const void *tmap, int x, int y,
                                           uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];" ::"r"(dst_smem),
      "l"(tmap), "r"(x), "r"(y), "r"(mbar)
      : "memory");
}

// 2D bf16 tensor map, swizzle pinned to NONE
// The kernel reads tiles with plain row-major ldmatrix addressing, so TMA must
// write them unswizzled. (Auto-swizzle-by-stride would not match.)
inline void init_tmap_noswizzle(CUtensorMap *tmap, const nv_bf16 *gmem, uint64_t height,
                                uint64_t width, uint32_t box_h, uint32_t box_w) {
  uint64_t size[2] = {width, height};
  uint64_t stride[1] = {width * sizeof(nv_bf16)};
  uint32_t box[2] = {box_w, box_h};
  uint32_t estride[2] = {1, 1};
  CUresult res = cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void *)gmem, size, stride, box, estride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(res == CUDA_SUCCESS, "det_gemm: cuTensorMapEncodeTiled failed");
}

} // namespace det_gemm
