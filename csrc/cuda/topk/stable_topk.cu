// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Stable per-row top-k for the DSv4 sparse-attention indexer (SM90).
//
// Byte-equal to rl_engine.kernels.ops.pytorch.topk.stable_topk_per_row, which
// reproduces the selection of vLLM's batch-invariant DSv4 selector
// (ds4_bi.top_k_per_row_prefill): row r ranks the columns
// [row_starts[r], row_ends[r]) of logits[r] by the unique 64-bit key
//
//   E = (ordered_bits(score) << 32) | local_index,   descending,
//
// so equal score bits prefer the larger request-local index, and writes the
// local indices in key order, padded with -1.
//
// One block owns one row and nothing in it depends on other rows, the row's
// offset in the packed buffer, or the grid, so the output is batch-invariant.
// Shared-memory atomics only count or claim slots; every claimed set is either
// counted (order-free) or sorted before it is written.
//
// Selecting the k = min(top_k, length) largest keys:
//   0. A row with at most top_k candidates takes all of them.
//   1. Global pass 1: histogram of the top 11 score bits; find the bucket B
//      holding the k-th largest key, with r keys of B needed.
//   2. Global pass 2: keys above B go straight to the output set; keys in B go
//      to a shared-memory candidate list as full 64-bit keys E.
//   3. Radix-select the r-th largest E among the candidates (shared memory
//      only) and add the candidates >= it. E is unique, so exactly r are added
//      and score ties resolve by index without a separate pass.
//   If B holds more keys than the candidate list can take (huge tie groups),
//   steps 2-3 instead finish the radix select over the remaining score bits
//   with more global passes, find the tie cutoff index by counting ties down
//   from the row's end, and gather from global memory.
//   4. Sort the k keys descending and write their indices.
//
// Out-of-range rows are clamped to [0, width] instead of read out of bounds;
// the golden rejects them.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

#include <cstdint>

namespace {

constexpr int kMaxTopK = 2048;
constexpr int kDigitBits = 11;
constexpr int kBuckets = 1 << kDigitBits;  // 2048
constexpr int kCandidates = 2048;          // shared candidate list capacity
constexpr int kUnroll = 4;                 // global loads in flight per thread

// Block shapes. The selection is exact (keys are unique), so the output does
// not depend on which shape the host picks for a launch.
template <int kThreads>
struct Config {
  static constexpr int kWarps = kThreads / 32;
  static constexpr int kItems = kMaxTopK / kThreads;            // keys per thread in the sort
  static constexpr int kBucketsPerThread = kBuckets / kThreads;
  static constexpr int kMinBlocks = kThreads >= 1024 ? 1 : 2048 / kThreads / 2;
  using KeySort = cub::BlockRadixSort<uint64_t, kThreads, kItems>;
  using BucketScan = cub::BlockScan<uint32_t, kThreads>;
};

__device__ __forceinline__ uint32_t ordered_bits(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
}

__device__ __forceinline__ uint64_t compose(uint32_t key, int index) {
  return (static_cast<uint64_t>(key) << 32) | static_cast<uint32_t>(index);
}

template <int kThreads>
struct Shared {
  using C = Config<kThreads>;
  // The selection lives until its keys are in registers; the sort scratch
  // reuses the same storage afterwards.
  union {
    struct {
      uint32_t histogram[kBuckets];
      uint64_t candidates[kCandidates];
      uint64_t selected[kMaxTopK];
    } select;
    typename C::KeySort::TempStorage sort;
  } u;
  typename C::BucketScan::TempStorage scan;
  uint32_t warp_ties[C::kWarps];
  uint32_t selected_count, candidate_count;
  uint32_t bucket, rank, bucket_count;
  int cutoff;
};

// Visits keys [0, length) of a row with kUnroll independent loads per thread
// in flight. Every thread runs every iteration (so warp collectives are safe);
// `valid` marks the lanes that hold a key.
template <int kThreads, typename Fn>
__device__ __forceinline__ void for_each_key(const float* base, int64_t stride, int length,
                                             Fn fn) {
  for (int chunk = 0; chunk < length; chunk += kThreads * kUnroll) {
    uint32_t keys[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = chunk + u * kThreads + static_cast<int>(threadIdx.x);
      keys[u] = i < length ? ordered_bits(__ldg(base + static_cast<int64_t>(i) * stride)) : 0U;
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int i = chunk + u * kThreads + static_cast<int>(threadIdx.x);
      fn(keys[u], i, i < length);
    }
  }
}

// Warp-aggregated histogram increment: one shared atomic per distinct bucket
// per warp, so large tie groups do not serialize on one address.
__device__ __forceinline__ void histogram_add(uint32_t* histogram, bool hit, uint32_t bucket) {
  const uint32_t label = hit ? bucket : kBuckets + (threadIdx.x & 31);
  const uint32_t peers = __match_any_sync(0xFFFFFFFFU, label);
  if (hit && (threadIdx.x & 31) == __ffs(peers) - 1) {
    atomicAdd(&histogram[bucket], static_cast<uint32_t>(__popc(peers)));
  }
}

// Warp-aggregated slot claim on a shared counter.
__device__ __forceinline__ uint32_t claim_slot(uint32_t* counter, bool take) {
  const int lane = threadIdx.x & 31;
  const uint32_t ballot = __ballot_sync(0xFFFFFFFFU, take);
  uint32_t first = 0;
  if (lane == 0 && ballot) first = atomicAdd(counter, static_cast<uint32_t>(__popc(ballot)));
  first = __shfl_sync(0xFFFFFFFFU, first, 0);
  return first + __popc(ballot & ((1U << lane) - 1U));
}

template <int kThreads>
__device__ __forceinline__ void zero_histogram(Shared<kThreads>& s) {
#pragma unroll
  for (int j = 0; j < Config<kThreads>::kBucketsPerThread; ++j) {
    s.u.select.histogram[threadIdx.x + j * kThreads] = 0;
  }
}

// Finds the bucket holding the rank-th largest counted key (1-indexed),
// walking buckets from the top; thread t owns the kBucketsPerThread buckets
// below 2047 - t * kBucketsPerThread. Writes s.bucket, s.rank (rank within the
// bucket) and s.bucket_count.
template <int kThreads>
__device__ __forceinline__ void select_bucket(Shared<kThreads>& s, uint32_t rank) {
  constexpr int kPer = Config<kThreads>::kBucketsPerThread;
  const int high = kBuckets - 1 - kPer * static_cast<int>(threadIdx.x);
  uint32_t counts[kPer];
  uint32_t sum = 0;
#pragma unroll
  for (int j = 0; j < kPer; ++j) sum += counts[j] = s.u.select.histogram[high - j];
  uint32_t before;
  typename Config<kThreads>::BucketScan(s.scan).ExclusiveSum(sum, before);
  if (before < rank && rank <= before + sum) {
    uint32_t r = rank - before;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      if (r <= counts[j]) {
        s.bucket = high - j, s.rank = r, s.bucket_count = counts[j];
        break;
      }
      r -= counts[j];
    }
  }
  __syncthreads();
}

template <int kThreads>
__global__ void __launch_bounds__(kThreads, Config<kThreads>::kMinBlocks) stable_topk_per_row_kernel(
    const float* __restrict__ logits, const int* __restrict__ row_starts,
    const int* __restrict__ row_ends, int* __restrict__ out, int64_t stride0, int64_t stride1,
    int64_t out_stride, int width, int top_k) {
  using C = Config<kThreads>;
  __shared__ Shared<kThreads> s;

  const int row = blockIdx.x;
  const int start = max(0, row_starts[row]);
  const int end = min(width, row_ends[row]);
  const int length = max(0, end - start);
  const int taken = min(top_k, length);
  int* row_out = out + static_cast<int64_t>(row) * out_stride;

  if (taken == 0) {
    for (int c = threadIdx.x; c < top_k; c += kThreads) row_out[c] = -1;
    return;
  }
  const float* base = logits + static_cast<int64_t>(row) * stride0 +
                      static_cast<int64_t>(start) * stride1;

  if (threadIdx.x == 0) s.selected_count = 0, s.candidate_count = 0;

  if (length <= top_k) {
    // 0. Everything is taken.
    __syncthreads();
    for_each_key<kThreads>(base, stride1, length, [&](uint32_t key, int i, bool valid) {
      const uint32_t slot = claim_slot(&s.selected_count, valid);
      if (valid) s.u.select.selected[slot] = compose(key, i);
    });
  } else {
    // 1. Top 11 score bits.
    zero_histogram(s);
    __syncthreads();
    for_each_key<kThreads>(base, stride1, length, [&](uint32_t key, int, bool valid) {
      histogram_add(s.u.select.histogram, valid, key >> (32 - kDigitBits));
    });
    __syncthreads();
    select_bucket(s, static_cast<uint32_t>(taken));
    const uint32_t top_bucket = s.bucket;
    const uint32_t need = s.rank;            // keys of the bucket to keep
    const uint32_t in_bucket = s.bucket_count;
    const bool fits = in_bucket <= kCandidates;

    // 2. Above the bucket: keep. In the bucket: candidates (when they fit).
    for_each_key<kThreads>(base, stride1, length, [&](uint32_t key, int i, bool valid) {
      const uint32_t digit = key >> (32 - kDigitBits);
      const bool above = valid && digit > top_bucket;
      const uint32_t slot = claim_slot(&s.selected_count, above);
      if (above) s.u.select.selected[slot] = compose(key, i);
      if (fits) {
        const bool candidate = valid && digit == top_bucket;
        const uint32_t c = claim_slot(&s.candidate_count, candidate);
        if (candidate) s.u.select.candidates[c] = compose(key, i);
      }
    });
    __syncthreads();

    if (fits) {
      // 3. r-th largest 64-bit key among the candidates. Bits 63..53 are
      //    shared; resolve bits 52..0 in digits of 11 (the last pass reads 11
      //    bits but only bits 8..0 are still free, which keeps the order).
      uint64_t prefix = static_cast<uint64_t>(top_bucket) << (64 - kDigitBits);
      uint32_t rank = need;
      int done_shift = 64 - kDigitBits;  // bits >= done_shift are fixed in prefix
      const uint32_t count = s.candidate_count;
      for (int shift = 64 - 2 * kDigitBits; done_shift > 0; shift -= kDigitBits) {
        shift = max(shift, 0);
        const uint64_t fixed = ~uint64_t{0} << done_shift;
        zero_histogram(s);
        __syncthreads();
        for (uint32_t base_c = 0; base_c < count; base_c += kThreads) {
          const uint32_t c = base_c + threadIdx.x;
          const uint64_t e = c < count ? s.u.select.candidates[c] : 0;
          histogram_add(s.u.select.histogram, c < count && (e & fixed) == prefix,
                        static_cast<uint32_t>(e >> shift) & (kBuckets - 1));
        }
        __syncthreads();
        select_bucket(s, rank);
        prefix = (prefix & fixed) | (static_cast<uint64_t>(s.bucket) << shift);
        rank = s.rank;
        done_shift = shift;
        // Every remaining key in the bucket is kept: the bucket's floor is
        // the threshold.
        if (rank == s.bucket_count) break;
        __syncthreads();  // the histogram is reset next pass
      }
      __syncthreads();
      for (uint32_t base_c = 0; base_c < count; base_c += kThreads) {
        const uint32_t c = base_c + threadIdx.x;
        const bool keep = c < count && s.u.select.candidates[c] >= prefix;
        const uint64_t e = keep ? s.u.select.candidates[c] : 0;
        const uint32_t slot = claim_slot(&s.selected_count, keep);
        if (keep) s.u.select.selected[slot] = e;
      }
    } else {
      // Fallback for a huge bucket: finish on the remaining 21 score bits with
      // global passes (11 then 10 bits).
      uint32_t prefix = top_bucket << (32 - kDigitBits);
      uint32_t rank = need;
      uint32_t ties = in_bucket;
      int done_shift = 32 - kDigitBits;
      for (int shift = 32 - 2 * kDigitBits; done_shift > 0; shift -= kDigitBits) {
        shift = max(shift, 0);
        const uint32_t fixed = ~0U << done_shift;
        zero_histogram(s);
        __syncthreads();
        for_each_key<kThreads>(base, stride1, length, [&](uint32_t key, int, bool valid) {
          histogram_add(s.u.select.histogram, valid && (key & fixed) == prefix,
                        (key >> shift) & (kBuckets - 1));
        });
        __syncthreads();
        select_bucket(s, rank);
        prefix = (prefix & fixed) | (s.bucket << shift);
        rank = s.rank;
        ties = s.bucket_count;
        done_shift = shift;
        __syncthreads();
      }
      // prefix is the threshold score bits; keep `rank` of its `ties` keys,
      // those with the largest local indices.
      const uint32_t threshold = prefix;
      int cutoff = 0;
      if (rank < ties) {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        uint32_t seen = 0;
        for (int chunk_end = length; chunk_end > 0; chunk_end -= kThreads) {
          const int i = chunk_end - 1 - static_cast<int>(threadIdx.x);
          const bool tie =
              i >= 0 && ordered_bits(__ldg(base + static_cast<int64_t>(i) * stride1)) == threshold;
          const uint32_t ballot = __ballot_sync(0xFFFFFFFFU, tie);
          if (lane == 0) s.warp_ties[warp] = __popc(ballot);
          __syncthreads();
          uint32_t before = 0, total = 0;
#pragma unroll 8
          for (int w = 0; w < C::kWarps; ++w) {
            const uint32_t c = s.warp_ties[w];
            before += w < warp ? c : 0U;
            total += c;
          }
          // 1-indexed rank of this tie counted from the highest local index.
          if (tie && seen + before + __popc(ballot & ((1U << lane) - 1U)) + 1U == rank) {
            s.cutoff = i;
          }
          __syncthreads();
          seen += total;
          if (seen >= rank) break;
        }
        cutoff = s.cutoff;
      }
      // Gather the whole selection again (the pass-2 claims are discarded).
      if (threadIdx.x == 0) s.selected_count = 0;
      __syncthreads();
      for_each_key<kThreads>(base, stride1, length, [&](uint32_t key, int i, bool valid) {
        const bool keep = valid && (key > threshold || (key == threshold && i >= cutoff));
        const uint32_t slot = claim_slot(&s.selected_count, keep);
        if (keep) s.u.select.selected[slot] = compose(key, i);
      });
    }
  }
  __syncthreads();

  // 4. Sort the `taken` keys descending (slot order is scheduling-dependent),
  //    repacked as (score << idx_bits) | index so the radix sort covers only
  //    32 + idx_bits bits instead of 64.
  const int idx_bits = length > 1 ? 32 - __clz(length - 1) : 1;
  uint64_t keys[C::kItems];
#pragma unroll
  for (int item = 0; item < C::kItems; ++item) {
    const int slot = threadIdx.x * C::kItems + item;
    const uint64_t e = slot < taken ? s.u.select.selected[slot] : uint64_t{0};
    keys[item] = ((e >> 32) << idx_bits) | (e & 0xFFFFFFFFULL);
  }
  __syncthreads();
  typename C::KeySort(s.u.sort).SortDescendingBlockedToStriped(keys, 0, 32 + idx_bits);
  const uint64_t index_mask = (uint64_t{1} << idx_bits) - 1;
#pragma unroll
  for (int item = 0; item < C::kItems; ++item) {
    const int column = item * kThreads + threadIdx.x;
    if (column < top_k) {
      row_out[column] = column < taken ? static_cast<int>(keys[item] & index_mask) : -1;
    }
  }
}

template <int kThreads>
void launch(const torch::Tensor& logits, const torch::Tensor& row_starts,
            const torch::Tensor& row_ends, torch::Tensor& indices, int64_t top_k,
            cudaStream_t stream) {
  stable_topk_per_row_kernel<kThreads><<<static_cast<unsigned>(logits.size(0)), kThreads, 0,
                                         stream>>>(
      logits.data_ptr<float>(), row_starts.data_ptr<int>(), row_ends.data_ptr<int>(),
      indices.data_ptr<int>(), logits.stride(0), logits.stride(1), indices.stride(0),
      static_cast<int>(logits.size(1)), static_cast<int>(top_k));
}

}  // namespace

void stable_topk_per_row_out(torch::Tensor logits, torch::Tensor row_starts,
                             torch::Tensor row_ends, torch::Tensor indices, int64_t top_k,
                             int64_t threads) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 && logits.dim() == 2,
              "stable_topk: logits must be a 2-D FP32 CUDA tensor");
  const int64_t rows = logits.size(0);
  TORCH_CHECK(top_k > 0 && top_k <= kMaxTopK, "stable_topk: top_k must be in (0, ", kMaxTopK,
              "], got ", top_k);
  // Headroom below INT32_MAX keeps the unrolled index arithmetic in range.
  TORCH_CHECK(logits.size(1) <= (int64_t{1} << 30), "stable_topk: row width must be <= 2^30");
  for (const auto* t : {&row_starts, &row_ends}) {
    TORCH_CHECK(t->is_cuda() && t->device() == logits.device() &&
                    t->scalar_type() == torch::kInt32 && t->dim() == 1 && t->size(0) == rows &&
                    t->is_contiguous(),
                "stable_topk: row_starts/row_ends must be contiguous int32 [rows] on the "
                "logits device");
  }
  TORCH_CHECK(indices.is_cuda() && indices.device() == logits.device() &&
                  indices.scalar_type() == torch::kInt32 && indices.dim() == 2 &&
                  indices.size(0) == rows && indices.size(1) == top_k &&
                  (indices.stride(1) == 1 || top_k == 1),
              "stable_topk: indices must be int32 [rows, top_k] with unit column stride");
  if (rows == 0) return;

  const c10::cuda::CUDAGuard guard(logits.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (threads == 0) {
    // Narrow blocks let several rows share an SM and hide each other's
    // barriers (fastest on H200 everywhere but one case); only a grid with
    // fewer rows than SMs and very long rows needs wide blocks for bandwidth.
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    threads = rows < sms && logits.size(1) >= 32768 ? 1024 : 256;
  }
  switch (threads) {
    case 256: launch<256>(logits, row_starts, row_ends, indices, top_k, stream); break;
    case 512: launch<512>(logits, row_starts, row_ends, indices, top_k, stream); break;
    case 1024: launch<1024>(logits, row_starts, row_ends, indices, top_k, stream); break;
    default: TORCH_CHECK(false, "stable_topk: threads must be 0, 256, 512 or 1024");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
