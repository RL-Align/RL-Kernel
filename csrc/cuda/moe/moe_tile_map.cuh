// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Token-block map for the SM90 grouped MoE GEMMs (forward and backward).
//
// Rows are sorted by expert (offsets[E + 1]); expert e owns ceil(count_e / BM)
// token blocks of BM rows. A GEMM CTA needs (expert, first row) of its token
// block. Scanning all experts in every CTA costs O(E) dependent loads per CTA,
// and the grid is sized from a host-side upper bound, so at small M most CTAs
// only scan and exit (E = 256, M = 6: ~8k CTAs, ~200 with work). One small
// kernel builds the map once per GEMM instead:
//
//   tile_expert[t], tile_row0[t] for t < count,   count = sum_e ceil(count_e / BM)
//
// and a CTA decodes its block with two loads. Scheduling only: every tile's
// bytes are independent of which CTA computes it, so the outputs are unchanged.
//
// Host-side bound without a device read (tiles <= M / BM + nonempty experts):
//   num_tb = ceil(M / BM) + min(E, M)
#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cub/block/block_scan.cuh>

namespace moe_tile_map {

constexpr int kThreads = 1024;

struct TileMap {
  const int32_t* expert = nullptr;   // [num_tb]
  const int32_t* row0 = nullptr;     // [num_tb]
  const int32_t* count = nullptr;    // [1]
};

inline int64_t token_block_bound(int64_t M, int64_t E, int bm) {
  return (M + bm - 1) / bm + std::min<int64_t>(E, M);
}

// One block. start[] (E + 1 ints) lives in dynamic shared memory.
template <int BM>
__global__ void __launch_bounds__(kThreads, 1)
build_kernel(const int32_t* __restrict__ offsets, int E, int num_tb, int32_t* __restrict__ out) {
  using Scan = cub::BlockScan<int, kThreads>;
  __shared__ typename Scan::TempStorage scan_tmp;
  __shared__ int carry;
  extern __shared__ int start[];   // [E + 1]
  const int tid = threadIdx.x;
  if (tid == 0) carry = 0;
  __syncthreads();
  for (int base = 0; base < E; base += kThreads) {
    const int e = base + tid;
    const int nb = e < E ? (offsets[e + 1] - offsets[e] + BM - 1) / BM : 0;
    int excl, total;
    Scan(scan_tmp).ExclusiveSum(nb, excl, total);
    if (e < E) start[e] = carry + excl;
    __syncthreads();
    if (tid == 0) carry += total;
    __syncthreads();
  }
  const int count = carry;
  if (tid == 0) start[E] = count;
  __syncthreads();
  int32_t* tile_expert = out;
  int32_t* tile_row0 = out + num_tb;
  for (int t = tid; t < count; t += kThreads) {
    // last expert e with start[e] <= t (empty experts share a start; take the last)
    int lo = 0, hi = E;   // invariant: start[lo] <= t < start[hi]
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (start[mid] <= t) lo = mid; else hi = mid;
    }
    tile_expert[t] = lo;
    tile_row0[t] = offsets[lo] + (t - start[lo]) * BM;
  }
  if (tid == 0) out[2 * num_tb] = count;
}

// Builds the map on `stream`. `storage` keeps the int32 [2 * num_tb + 1] buffer alive
// for the caller (the GEMM launch is stream-ordered after it).
template <int BM>
TileMap build(const int32_t* offsets, int64_t E, int64_t num_tb, const at::TensorOptions& opts,
              cudaStream_t stream, at::Tensor& storage) {
  storage = at::empty({2 * num_tb + 1}, opts.dtype(at::kInt));
  int32_t* out = storage.data_ptr<int32_t>();
  const size_t smem = static_cast<size_t>(E + 1) * sizeof(int);
  build_kernel<BM><<<1, kThreads, smem, stream>>>(offsets, static_cast<int>(E),
                                                  static_cast<int>(num_tb), out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return TileMap{out, out + num_tb, out + 2 * num_tb};
}

}  // namespace moe_tile_map
