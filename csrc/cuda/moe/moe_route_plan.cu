// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// DSv4 MoE route plan as a counting sort (rl_engine.kernels.ops.pytorch.moe.route_plan):
// the T * 6 (token, slot) pairs, stably sorted by expert. A stable sort's result is unique,
// so this equals the golden's torch.sort(stable=True) exactly; all integer work.
//
//   count:   one block per chunk of kChunk pairs: per-expert histogram -> counts[chunk][E].
//   scan:    one block: base[chunk][e] = (pairs of experts < e) + (pairs of expert e in
//            earlier chunks); offsets[e] for e in 0..E.
//   scatter: one block per chunk: each pair's position = base[chunk][e] + (pairs of expert e
//            earlier in the chunk), from per-warp counts and __match_any_sync ranks.
//            Writes perm[pos] = pair, token[pos] = pair / 6, inv[pair] = pos.
// No host synchronization: the launch sizes depend only on T and E.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kTopK = 6;
constexpr int kChunk = 512;  // pairs per block = threads per block
constexpr int kWarps = kChunk / 32;
constexpr int kMaxExperts = 1024;

template <typename Id>
__global__ void __launch_bounds__(kChunk)
    count_kernel(const Id* __restrict__ ids, int64_t pairs, int experts,
                 int32_t* __restrict__ counts) {
  extern __shared__ int32_t hist[];
  for (int e = threadIdx.x; e < experts; e += kChunk) hist[e] = 0;
  __syncthreads();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kChunk + threadIdx.x;
  if (i < pairs) atomicAdd(&hist[static_cast<int>(ids[i])], 1);
  __syncthreads();
  int32_t* row = counts + static_cast<int64_t>(blockIdx.x) * experts;
  for (int e = threadIdx.x; e < experts; e += kChunk) row[e] = hist[e];
}

// One block of kMaxExperts threads; thread e owns expert e.
__global__ void __launch_bounds__(kMaxExperts)
    scan_kernel(int32_t* __restrict__ base, int chunks, int experts,
                int32_t* __restrict__ offsets) {
  __shared__ int32_t warp_sums[kMaxExperts / 32];
  const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
  int32_t total = 0;
  if (e < experts) {
    for (int c = 0; c < chunks; ++c) {
      int32_t* p = base + static_cast<int64_t>(c) * experts + e;
      const int32_t n = *p;
      *p = total;
      total += n;
    }
  }
  // Exclusive scan of the per-expert totals.
  int32_t incl = total;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int32_t v = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) warp_sums[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int32_t w = warp_sums[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int32_t v = __shfl_up_sync(0xffffffffu, w, o);
      if (lane >= o) w += v;
    }
    warp_sums[lane] = w;  // inclusive over warps
  }
  __syncthreads();
  const int32_t start = incl - total + (warp > 0 ? warp_sums[warp - 1] : 0);
  if (e < experts) {
    offsets[e] = start;
    for (int c = 0; c < chunks; ++c) base[static_cast<int64_t>(c) * experts + e] += start;
    if (e == experts - 1) offsets[experts] = start + total;
  }
}

template <typename Id>
__global__ void __launch_bounds__(kChunk)
    scatter_kernel(const Id* __restrict__ ids, int64_t pairs, int experts,
                   const int32_t* __restrict__ base, int64_t* __restrict__ perm,
                   int64_t* __restrict__ token, int64_t* __restrict__ inv) {
  extern __shared__ int32_t wcount[];  // [kWarps][experts], then exclusive prefixes
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int k = threadIdx.x; k < kWarps * experts; k += kChunk) wcount[k] = 0;
  __syncthreads();
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kChunk + threadIdx.x;
  const bool valid = i < pairs;
  const int key = valid ? static_cast<int>(ids[i]) : experts + lane;  // unique if invalid
  const unsigned peers = __match_any_sync(0xffffffffu, key);
  const int rank = __popc(peers & ((1u << lane) - 1u));
  if (valid && rank == 0) wcount[warp * experts + key] = __popc(peers);
  __syncthreads();
  const int32_t* row = base + static_cast<int64_t>(blockIdx.x) * experts;
  for (int e = threadIdx.x; e < experts; e += kChunk) {
    int32_t run = row[e];
    for (int w = 0; w < kWarps; ++w) {
      const int32_t n = wcount[w * experts + e];
      wcount[w * experts + e] = run;
      run += n;
    }
  }
  __syncthreads();
  if (valid) {
    const int64_t pos = wcount[warp * experts + key] + rank;
    perm[pos] = i;
    token[pos] = i / kTopK;
    inv[i] = pos;
  }
}

}  // namespace

// (perm int64 [P], token int64 [P], inv int64 [T, 6], offsets int32 [E + 1]).
std::vector<torch::Tensor> moe_route_plan(torch::Tensor ids, int64_t num_experts) {
  TORCH_CHECK(ids.is_cuda() && ids.dim() == 2 && ids.size(1) == kTopK && ids.is_contiguous(),
              "moe_route_plan: ids must be contiguous CUDA [T, 6]");
  TORCH_CHECK(ids.scalar_type() == torch::kInt32 || ids.scalar_type() == torch::kInt64,
              "moe_route_plan: ids must be int32 or int64");
  TORCH_CHECK(num_experts > 0 && num_experts <= kMaxExperts,
              "moe_route_plan: 1..1024 experts supported");
  const c10::cuda::CUDAGuard guard(ids.device());
  const int64_t t = ids.size(0), pairs = t * kTopK;
  TORCH_CHECK(pairs < (int64_t{1} << 31), "moe_route_plan: too many pairs");
  const int experts = static_cast<int>(num_experts);
  auto i64 = ids.options().dtype(torch::kInt64);
  auto perm = torch::empty({pairs}, i64), token = torch::empty({pairs}, i64);
  auto inv = torch::empty({t, kTopK}, i64);
  auto offsets = torch::empty({experts + 1}, ids.options().dtype(torch::kInt32));
  if (pairs == 0) {
    offsets.zero_();
    return {perm, token, inv, offsets};
  }
  const int chunks = static_cast<int>((pairs + kChunk - 1) / kChunk);
  auto base = torch::empty({chunks, experts}, ids.options().dtype(torch::kInt32));
  auto stream = at::cuda::getCurrentCUDAStream();
  const size_t scatter_smem = sizeof(int32_t) * kWarps * experts;
  if (scatter_smem > 48 * 1024) {
    AT_DISPATCH_INDEX_TYPES(ids.scalar_type(), "moe_route_plan_attr", [&] {
      C10_CUDA_CHECK(cudaFuncSetAttribute(scatter_kernel<index_t>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          static_cast<int>(scatter_smem)));
    });
  }
  AT_DISPATCH_INDEX_TYPES(ids.scalar_type(), "moe_route_plan", [&] {
    const auto* p = ids.data_ptr<index_t>();
    count_kernel<index_t><<<chunks, kChunk, sizeof(int32_t) * experts, stream>>>(
        p, pairs, experts, base.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scan_kernel<<<1, kMaxExperts, 0, stream>>>(base.data_ptr<int32_t>(), chunks, experts,
                                               offsets.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    scatter_kernel<index_t><<<chunks, kChunk, scatter_smem, stream>>>(
        p, pairs, experts, base.data_ptr<int32_t>(), perm.data_ptr<int64_t>(),
        token.data_ptr<int64_t>(), inv.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {perm, token, inv, offsets};
}
