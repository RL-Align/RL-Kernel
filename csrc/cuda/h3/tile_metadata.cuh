// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Segment-tile metadata shared by the H3 backward kernels that sum the rows of a gradient into the
// rows of a small table (adaln_row_gather, rmsnorm modulation, gate residual).
//
// The caller sorts the S positions by table row (`sorted_pos`), cuts every row's run into tiles
// `[tile_begin, tile_end)` over `sorted_pos`, and gives the first tile of each row in
// `seg_first_tile` (R + 1 entries). The partial kernels read gradients through `sorted_pos`, and the
// fold kernels read the partial rows of a segment's tiles, so a stale or short list would read out
// of bounds.
//
// check_tile_metadata() verifies the list on the device before any of that happens. It is a separate
// pass on purpose: an assert in or before the loops of the partial and fold kernels changes how the
// compiler schedules their loads and slowed them by 6-12% (and by several times inside the loop),
// while this pass costs one small launch (about 4 us).
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>

namespace h3_tiles {

static __global__ void check_tile_metadata_kernel(const int64_t* __restrict__ sorted_pos,
                                                  int64_t positions,
                                                  const int64_t* __restrict__ tile_begin,
                                                  const int64_t* __restrict__ tile_end,
                                                  int64_t tiles,
                                                  const int64_t* __restrict__ seg_first_tile,
                                                  int64_t segments) {
  const int64_t first_thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = first_thread; i < positions; i += stride) {
    CUDA_KERNEL_ASSERT(sorted_pos[i] >= 0 && sorted_pos[i] < positions);
  }
  for (int64_t i = first_thread; i < tiles; i += stride) {
    CUDA_KERNEL_ASSERT(tile_begin[i] >= 0 && tile_begin[i] <= tile_end[i] &&
                       tile_end[i] <= positions);
  }
  for (int64_t i = first_thread; i <= segments; i += stride) {
    CUDA_KERNEL_ASSERT(seg_first_tile[i] >= 0 && seg_first_tile[i] <= tiles);
    CUDA_KERNEL_ASSERT(i == segments || seg_first_tile[i] <= seg_first_tile[i + 1]);
  }
}

// `positions` entries in sorted_pos (each in [0, positions)), `tiles` tiles inside sorted_pos, and
// `segments` row segments (seg_first_tile has segments + 1 entries).
inline void check_tile_metadata(cudaStream_t stream, const int64_t* sorted_pos, int64_t positions,
                                const int64_t* tile_begin, const int64_t* tile_end, int64_t tiles,
                                const int64_t* seg_first_tile, int64_t segments) {
  const int64_t items = std::max({positions, tiles, segments + 1});
  const unsigned blocks = static_cast<unsigned>(std::min<int64_t>((items + 255) / 256, 1024));
  check_tile_metadata_kernel<<<blocks, 256, 0, stream>>>(sorted_pos, positions, tile_begin,
                                                         tile_end, tiles, seg_first_tile, segments);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Sequence-parallel entry points take explicit lists. `rows` (n_rows entries, each a row of an
// M-row activation, so in [0, row_limit)) and the tiles [tile_begin, tile_end) over it, and/or the
// segment boundaries seg_first_tile (segments + 1 entries, in [0, seg_limit], non-decreasing). A
// null pointer skips its part. Same device-side asserts, same single small launch.
static __global__ void check_lists_kernel(const int64_t* __restrict__ rows, int64_t n_rows,
                                          int64_t row_limit,
                                          const int64_t* __restrict__ tile_begin,
                                          const int64_t* __restrict__ tile_end, int64_t tiles,
                                          const int64_t* __restrict__ seg_first_tile,
                                          int64_t segments, int64_t seg_limit) {
  const int64_t first_thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  if (rows != nullptr) {
    for (int64_t i = first_thread; i < n_rows; i += stride) {
      CUDA_KERNEL_ASSERT(rows[i] >= 0 && rows[i] < row_limit);
    }
  }
  if (tile_begin != nullptr) {
    for (int64_t i = first_thread; i < tiles; i += stride) {
      CUDA_KERNEL_ASSERT(tile_begin[i] >= 0 && tile_begin[i] <= tile_end[i] &&
                         tile_end[i] <= n_rows);
    }
  }
  if (seg_first_tile != nullptr) {
    for (int64_t i = first_thread; i <= segments; i += stride) {
      CUDA_KERNEL_ASSERT(seg_first_tile[i] >= 0 && seg_first_tile[i] <= seg_limit);
      CUDA_KERNEL_ASSERT(i == segments || seg_first_tile[i] <= seg_first_tile[i + 1]);
    }
  }
}

inline void check_lists(cudaStream_t stream, const int64_t* rows, int64_t n_rows, int64_t row_limit,
                        const int64_t* tile_begin, const int64_t* tile_end, int64_t tiles,
                        const int64_t* seg_first_tile, int64_t segments, int64_t seg_limit) {
  const int64_t items = std::max({n_rows, tiles, segments + 1});
  const unsigned blocks = static_cast<unsigned>(std::min<int64_t>((items + 255) / 256, 1024));
  check_lists_kernel<<<blocks, 256, 0, stream>>>(rows, n_rows, row_limit, tile_begin, tile_end,
                                                 tiles, seg_first_tile, segments, seg_limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace h3_tiles
