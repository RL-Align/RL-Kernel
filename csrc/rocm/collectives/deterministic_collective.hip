// ROCm fixed-tree reduction kernels for the RCCL transport collective.
//
// RCCL is intentionally used only to transport rank-ordered tensors. The
// kernels below perform the arithmetic locally in the exact balanced tree used
// by the Python reference implementation. ReduceScatter receives a view that
// contains only the destination rank's shard, so it does not reduce unrelated
// rows.

#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPException.h>
#include <torch/extension.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <tuple>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;
constexpr int kMaxWorldSize = 8;
constexpr int64_t kIPCControlBytes = 256;
constexpr int64_t kIPCReadyOffset = 0;
constexpr int64_t kIPCDoneOffset = 64;
constexpr int64_t kIPCCloseOffset = 128;
constexpr int64_t kIPCSequenceOffset = 192;

struct PeerPointers {
  const void* values[kMaxWorldSize];
};

struct PeerSignals {
  uint64_t* ready[kMaxWorldSize];
  uint64_t* done[kMaxWorldSize];
  uint64_t* closed[kMaxWorldSize];
  uint64_t* sequence[kMaxWorldSize];
};

template <typename scalar_t>
__device__ __forceinline__ scalar_t ordered_add(scalar_t lower, scalar_t upper) {
  // Keep every parent as a separate expression. ROCm builds do not enable
  // fast-math, so this is the same dtype operation as torch.add_ for the
  // supported floating-point dtypes.
  return lower + upper;
}

template <typename scalar_t, int WorldSize>
__device__ __forceinline__ scalar_t fixed_tree_reduce(
    const scalar_t* values,
    int64_t rank_stride,
    int64_t index) {
  static_assert(
      WorldSize == 1 || WorldSize == 2 || WorldSize == 4 || WorldSize == 8,
      "unsupported deterministic collective world size");
  if constexpr (WorldSize == 1) {
    return values[index];
  } else {
    const scalar_t sum01 = ordered_add(
        values[index],
        values[rank_stride + index]);
    if constexpr (WorldSize == 2) {
      return sum01;
    } else {
      const scalar_t sum23 = ordered_add(
          values[2 * rank_stride + index],
          values[3 * rank_stride + index]);
      const scalar_t sum03 = ordered_add(sum01, sum23);
      if constexpr (WorldSize == 4) {
        return sum03;
      } else {
        const scalar_t sum45 = ordered_add(
            values[4 * rank_stride + index],
            values[5 * rank_stride + index]);
        const scalar_t sum67 = ordered_add(
            values[6 * rank_stride + index],
            values[7 * rank_stride + index]);
        const scalar_t sum47 = ordered_add(sum45, sum67);
        return ordered_add(sum03, sum47);
      }
    }
  }
}

template <typename scalar_t, int WorldSize>
__global__ void fixed_tree_reduce_kernel(
    const scalar_t* __restrict__ values,
    scalar_t* __restrict__ output,
    int64_t rank_stride,
    int64_t element_count) {
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_index; index < element_count; index += stride) {
    output[index] = fixed_tree_reduce<scalar_t, WorldSize>(values, rank_stride, index);
  }
}

template <typename scalar_t>
void launch_fixed_tree_reduce(
    const scalar_t* values,
    scalar_t* output,
    int64_t rank_stride,
    int64_t element_count,
    int64_t world_size,
    hipStream_t stream) {
  const int blocks = static_cast<int>(std::min<int64_t>(
      kMaxBlocks,
      (element_count + kThreads - 1) / kThreads));
  switch (world_size) {
    case 1:
      hipLaunchKernelGGL(
          (fixed_tree_reduce_kernel<scalar_t, 1>),
          dim3(blocks),
          dim3(kThreads),
          0,
          stream,
          values,
          output,
          rank_stride,
          element_count);
      break;
    case 2:
      hipLaunchKernelGGL(
          (fixed_tree_reduce_kernel<scalar_t, 2>),
          dim3(blocks),
          dim3(kThreads),
          0,
          stream,
          values,
          output,
          rank_stride,
          element_count);
      break;
    case 4:
      hipLaunchKernelGGL(
          (fixed_tree_reduce_kernel<scalar_t, 4>),
          dim3(blocks),
          dim3(kThreads),
          0,
          stream,
          values,
          output,
          rank_stride,
          element_count);
      break;
    case 8:
      hipLaunchKernelGGL(
          (fixed_tree_reduce_kernel<scalar_t, 8>),
          dim3(blocks),
          dim3(kThreads),
          0,
          stream,
          values,
          output,
          rank_stride,
          element_count);
      break;
    default:
      TORCH_CHECK(false, "unsupported deterministic collective world size ", world_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void validate_inputs(
    const torch::Tensor& rank_inputs,
    const torch::Tensor& output,
    const char* name) {
  TORCH_CHECK(rank_inputs.is_cuda(), name, ": rank_inputs must be a CUDA/ROCm tensor");
  TORCH_CHECK(output.is_cuda(), name, ": output must be a CUDA/ROCm tensor");
  TORCH_CHECK(rank_inputs.scalar_type() == output.scalar_type(), name, ": dtype mismatch");
  TORCH_CHECK(rank_inputs.dim() >= 1, name, ": rank_inputs must have a rank dimension");
  TORCH_CHECK(rank_inputs.size(0) == 1 || rank_inputs.size(0) == 2 ||
                  rank_inputs.size(0) == 4 || rank_inputs.size(0) == 8,
              name, ": unsupported rank dimension ", rank_inputs.size(0));
  TORCH_CHECK(rank_inputs.select(0, 0).is_contiguous(),
              name, ": each rank slice must be contiguous");
  TORCH_CHECK(output.is_contiguous(), name, ": output must be contiguous");
  TORCH_CHECK(rank_inputs.device() == output.device(), name, ": device mismatch");
  TORCH_CHECK(rank_inputs.numel() == output.numel() * rank_inputs.size(0),
              name, ": rank_inputs/output element count mismatch");
}

void launch_dispatch(
    const torch::Tensor& rank_inputs,
    const torch::Tensor& output,
    const char* name) {
  validate_inputs(rank_inputs, output, name);
  const int64_t world_size = rank_inputs.size(0);
  const int64_t element_count = output.numel();
  if (element_count == 0) {
    return;
  }
  const int64_t rank_stride = rank_inputs.stride(0);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      rank_inputs.scalar_type(),
      "deterministic_collective_rocm_fixed_tree",
      [&] {
        launch_fixed_tree_reduce<scalar_t>(
            rank_inputs.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rank_stride,
            element_count,
            world_size,
            stream);
      });
}

template <typename scalar_t, int WorldSize>
__device__ __forceinline__ scalar_t ipc_fixed_tree_reduce(
    const PeerPointers& peers,
    int64_t index) {
  const auto* rank0 = static_cast<const scalar_t*>(peers.values[0]);
  if constexpr (WorldSize == 1) {
    return rank0[index];
  } else {
    const auto* rank1 = static_cast<const scalar_t*>(peers.values[1]);
    const scalar_t sum01 = ordered_add(rank0[index], rank1[index]);
    if constexpr (WorldSize == 2) {
      return sum01;
    } else {
      const auto* rank2 = static_cast<const scalar_t*>(peers.values[2]);
      const auto* rank3 = static_cast<const scalar_t*>(peers.values[3]);
      const scalar_t sum23 = ordered_add(rank2[index], rank3[index]);
      const scalar_t sum03 = ordered_add(sum01, sum23);
      if constexpr (WorldSize == 4) {
        return sum03;
      } else {
        const auto* rank4 = static_cast<const scalar_t*>(peers.values[4]);
        const auto* rank5 = static_cast<const scalar_t*>(peers.values[5]);
        const auto* rank6 = static_cast<const scalar_t*>(peers.values[6]);
        const auto* rank7 = static_cast<const scalar_t*>(peers.values[7]);
        const scalar_t sum45 = ordered_add(rank4[index], rank5[index]);
        const scalar_t sum67 = ordered_add(rank6[index], rank7[index]);
        return ordered_add(sum03, ordered_add(sum45, sum67));
      }
    }
  }
}

template <typename packed_t>
__device__ __forceinline__ packed_t ordered_add_packed(
    packed_t lower,
    packed_t upper);

template <>
__device__ __forceinline__ __half2 ordered_add_packed(
    __half2 lower,
    __half2 upper) {
  return __hadd2(lower, upper);
}

template <>
__device__ __forceinline__ __hip_bfloat162 ordered_add_packed(
    __hip_bfloat162 lower,
    __hip_bfloat162 upper) {
  return __hadd2(lower, upper);
}

template <typename packed_t, int WorldSize>
__device__ __forceinline__ packed_t ipc_fixed_tree_reduce_packed(
    const PeerPointers& peers,
    int64_t index) {
  const auto* rank0 = static_cast<const packed_t*>(peers.values[0]);
  if constexpr (WorldSize == 1) {
    return rank0[index];
  } else {
    const auto* rank1 = static_cast<const packed_t*>(peers.values[1]);
    const packed_t sum01 = ordered_add_packed(rank0[index], rank1[index]);
    if constexpr (WorldSize == 2) {
      return sum01;
    } else {
      const auto* rank2 = static_cast<const packed_t*>(peers.values[2]);
      const auto* rank3 = static_cast<const packed_t*>(peers.values[3]);
      const packed_t sum23 = ordered_add_packed(rank2[index], rank3[index]);
      const packed_t sum03 = ordered_add_packed(sum01, sum23);
      if constexpr (WorldSize == 4) {
        return sum03;
      } else {
        const auto* rank4 = static_cast<const packed_t*>(peers.values[4]);
        const auto* rank5 = static_cast<const packed_t*>(peers.values[5]);
        const auto* rank6 = static_cast<const packed_t*>(peers.values[6]);
        const auto* rank7 = static_cast<const packed_t*>(peers.values[7]);
        const packed_t sum45 = ordered_add_packed(rank4[index], rank5[index]);
        const packed_t sum67 = ordered_add_packed(rank6[index], rank7[index]);
        return ordered_add_packed(sum03, ordered_add_packed(sum45, sum67));
      }
    }
  }
}

template <typename scalar_t, int WorldSize>
__global__ void ipc_fixed_tree_reduce_kernel(
    PeerPointers peers,
    scalar_t* __restrict__ output,
    int64_t input_offset,
    int64_t element_count) {
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_index; index < element_count; index += stride) {
    output[index] = ipc_fixed_tree_reduce<scalar_t, WorldSize>(
        peers,
        input_offset + index);
  }
}

template <typename packed_t, int WorldSize>
__global__ void ipc_fixed_tree_reduce_packed_kernel(
    PeerPointers peers,
    packed_t* __restrict__ output,
    int64_t input_offset,
    int64_t element_count) {
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_index; index < element_count; index += stride) {
    output[index] = ipc_fixed_tree_reduce_packed<packed_t, WorldSize>(
        peers,
        input_offset + index);
  }
}

template <typename scalar_t>
void launch_ipc_fixed_tree_reduce(
    const PeerPointers& peers,
    scalar_t* output,
    int64_t input_offset,
    int64_t element_count,
    int64_t world_size,
    hipStream_t stream) {
  const int blocks = static_cast<int>(std::min<int64_t>(
      kMaxBlocks,
      (element_count + kThreads - 1) / kThreads));
  switch (world_size) {
    case 1:
      hipLaunchKernelGGL(
          (ipc_fixed_tree_reduce_kernel<scalar_t, 1>),
          dim3(blocks), dim3(kThreads), 0, stream,
          peers, output, input_offset, element_count);
      break;
    case 2:
      hipLaunchKernelGGL(
          (ipc_fixed_tree_reduce_kernel<scalar_t, 2>),
          dim3(blocks), dim3(kThreads), 0, stream,
          peers, output, input_offset, element_count);
      break;
    case 4:
      hipLaunchKernelGGL(
          (ipc_fixed_tree_reduce_kernel<scalar_t, 4>),
          dim3(blocks), dim3(kThreads), 0, stream,
          peers, output, input_offset, element_count);
      break;
    case 8:
      hipLaunchKernelGGL(
          (ipc_fixed_tree_reduce_kernel<scalar_t, 8>),
          dim3(blocks), dim3(kThreads), 0, stream,
          peers, output, input_offset, element_count);
      break;
    default:
      TORCH_CHECK(false, "unsupported deterministic collective world size ", world_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename packed_t>
void launch_ipc_fixed_tree_reduce_packed(
    const PeerPointers& peers,
    packed_t* output,
    int64_t input_offset,
    int64_t element_count,
    int64_t world_size,
    hipStream_t stream) {
  const int blocks = static_cast<int>(std::min<int64_t>(
      kMaxBlocks,
      (element_count + kThreads - 1) / kThreads));
  switch (world_size) {
    case 1:
      hipLaunchKernelGGL(
          (ipc_fixed_tree_reduce_packed_kernel<packed_t, 1>),
          dim3(blocks), dim3(kThreads), 0, stream,
          peers, output, input_offset, element_count);
      break;
    case 2:
      hipLaunchKernelGGL(
          (ipc_fixed_tree_reduce_packed_kernel<packed_t, 2>),
          dim3(blocks), dim3(kThreads), 0, stream,
          peers, output, input_offset, element_count);
      break;
    case 4:
      hipLaunchKernelGGL(
          (ipc_fixed_tree_reduce_packed_kernel<packed_t, 4>),
          dim3(blocks), dim3(kThreads), 0, stream,
          peers, output, input_offset, element_count);
      break;
    case 8:
      hipLaunchKernelGGL(
          (ipc_fixed_tree_reduce_packed_kernel<packed_t, 8>),
          dim3(blocks), dim3(kThreads), 0, stream,
          peers, output, input_offset, element_count);
      break;
    default:
      TORCH_CHECK(false, "unsupported deterministic collective world size ", world_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void ipc_wait_signal_kernel(
    PeerSignals signals,
    uint64_t sequence,
    int64_t world_size,
    bool wait_for_done) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int peer = 0; peer < world_size; ++peer) {
    uint64_t* signal = wait_for_done ? signals.done[peer] : signals.ready[peer];
    while (__hip_atomic_load(
               signal,
               __ATOMIC_ACQUIRE,
               __HIP_MEMORY_SCOPE_SYSTEM) < sequence) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
}

__global__ void ipc_mark_signal_kernel(uint64_t* signal, uint64_t sequence) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __hip_atomic_store(
        signal,
        sequence,
        __ATOMIC_RELEASE,
        __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__global__ void ipc_reserve_sequence_kernel(
    PeerSignals signals,
    int64_t rank,
    int64_t world_size) {
  // One lane polls each peer. The next generation cannot reuse staging until
  // every rank has finished reading the previous generation.
  __shared__ uint64_t previous;
  if (threadIdx.x == 0) {
    previous = __hip_atomic_load(
        signals.sequence[rank], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
  if (threadIdx.x < world_size) {
    while (__hip_atomic_load(signals.done[threadIdx.x], __ATOMIC_ACQUIRE,
                            __HIP_MEMORY_SCOPE_SYSTEM) < previous) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_store(signals.sequence[rank], previous + 1,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__global__ void ipc_mark_ready_and_wait_dynamic_kernel(
    PeerSignals signals,
    int64_t rank,
    int64_t world_size) {
  // Publish the local payload, then wait for all peers in parallel.
  __shared__ uint64_t sequence;
  if (threadIdx.x == 0) {
    sequence = __hip_atomic_load(signals.sequence[rank], __ATOMIC_ACQUIRE,
                                __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(signals.ready[rank], sequence, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  __syncthreads();
  if (threadIdx.x < world_size) {
    while (__hip_atomic_load(signals.ready[threadIdx.x], __ATOMIC_ACQUIRE,
                            __HIP_MEMORY_SCOPE_SYSTEM) != sequence) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
}

__global__ void ipc_mark_done_dynamic_kernel(PeerSignals signals, int64_t rank) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const uint64_t sequence = __hip_atomic_load(
        signals.sequence[rank],
        __ATOMIC_ACQUIRE,
        __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(
        signals.done[rank],
        sequence,
        __ATOMIC_RELEASE,
        __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__global__ void ipc_synchronize_dynamic_kernel(
    PeerSignals signals,
    int64_t rank,
    int64_t world_size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const uint64_t sequence = __hip_atomic_load(
      signals.sequence[rank],
      __ATOMIC_ACQUIRE,
      __HIP_MEMORY_SCOPE_SYSTEM);
  for (int peer = 0; peer < world_size; ++peer) {
    while (__hip_atomic_load(
               signals.done[peer],
               __ATOMIC_ACQUIRE,
               __HIP_MEMORY_SCOPE_SYSTEM) < sequence) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
}

__global__ void ipc_mark_ready_and_wait_kernel(
    PeerSignals signals,
    int64_t rank,
    uint64_t sequence,
    int64_t world_size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  __hip_atomic_store(
      signals.ready[rank],
      sequence,
      __ATOMIC_RELEASE,
      __HIP_MEMORY_SCOPE_SYSTEM);
  for (int peer = 0; peer < world_size; ++peer) {
    while (__hip_atomic_load(
               signals.ready[peer],
               __ATOMIC_ACQUIRE,
               __HIP_MEMORY_SCOPE_SYSTEM) < sequence) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
}

__global__ void ipc_close_and_wait_kernel(
    PeerSignals signals,
    int64_t rank,
    int64_t world_size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int peer = 0; peer < world_size; ++peer) {
    __hip_atomic_fetch_add(
        signals.closed[peer],
        static_cast<uint64_t>(1),
        __ATOMIC_ACQ_REL,
        __HIP_MEMORY_SCOPE_SYSTEM);
  }
  while (__hip_atomic_load(
             signals.closed[rank],
             __ATOMIC_ACQUIRE,
             __HIP_MEMORY_SCOPE_SYSTEM) < static_cast<uint64_t>(world_size)) {
    __builtin_amdgcn_s_sleep(1);
  }
}

__global__ void ipc_all_gather_uint4_kernel(
    PeerPointers peers,
    uint4* __restrict__ output,
    int64_t vectors_per_rank,
    int64_t world_size) {
  const int64_t total_vectors = vectors_per_rank * world_size;
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_index; index < total_vectors; index += stride) {
    const int peer = static_cast<int>(index / vectors_per_rank);
    const int64_t peer_index = index - static_cast<int64_t>(peer) * vectors_per_rank;
    output[index] = static_cast<const uint4*>(peers.values[peer])[peer_index];
  }
}

__global__ void ipc_all_gather_bytes_kernel(
    PeerPointers peers,
    uint8_t* __restrict__ output,
    int64_t bytes_per_rank,
    int64_t world_size) {
  const int64_t total_bytes = bytes_per_rank * world_size;
  const int64_t thread_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = thread_index; index < total_bytes; index += stride) {
    const int peer = static_cast<int>(index / bytes_per_rank);
    const int64_t peer_index = index - static_cast<int64_t>(peer) * bytes_per_rank;
    output[index] = static_cast<const uint8_t*>(peers.values[peer])[peer_index];
  }
}

void launch_ipc_all_gather(
    const PeerPointers& peers,
    void* output,
    int64_t bytes_per_rank,
    int64_t world_size,
    hipStream_t stream) {
  if (bytes_per_rank == 0) {
    return;
  }
  if (bytes_per_rank % static_cast<int64_t>(sizeof(uint4)) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(uint4) == 0) {
    const int64_t vectors_per_rank = bytes_per_rank / sizeof(uint4);
    const int64_t total_vectors = vectors_per_rank * world_size;
    const int blocks = static_cast<int>(std::min<int64_t>(
        kMaxBlocks,
        (total_vectors + kThreads - 1) / kThreads));
    hipLaunchKernelGGL(
        ipc_all_gather_uint4_kernel,
        dim3(blocks),
        dim3(kThreads),
        0,
        stream,
        peers,
        static_cast<uint4*>(output),
        vectors_per_rank,
        world_size);
  } else {
    const int64_t total_bytes = bytes_per_rank * world_size;
    const int blocks = static_cast<int>(std::min<int64_t>(
        kMaxBlocks,
        (total_bytes + kThreads - 1) / kThreads));
    hipLaunchKernelGGL(
        ipc_all_gather_bytes_kernel,
        dim3(blocks),
        dim3(kThreads),
        0,
        stream,
        peers,
        static_cast<uint8_t*>(output),
        bytes_per_rank,
        world_size);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_wait_signal(
    const PeerSignals& signals,
    uint64_t sequence,
    int64_t world_size,
    bool wait_for_done,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      ipc_wait_signal_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      signals,
      sequence,
      world_size,
      wait_for_done);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mark_signal(uint64_t* signal, uint64_t sequence, hipStream_t stream) {
  hipLaunchKernelGGL(
      ipc_mark_signal_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      signal,
      sequence);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mark_ready_and_wait(
    const PeerSignals& signals,
    int64_t rank,
    uint64_t sequence,
    int64_t world_size,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      ipc_mark_ready_and_wait_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      signals,
      rank,
      sequence,
      world_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_reserve_sequence(
    const PeerSignals& signals,
    int64_t rank,
    int64_t world_size,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      ipc_reserve_sequence_kernel,
      dim3(1),
      dim3(64),
      0,
      stream,
      signals,
      rank,
      world_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mark_ready_and_wait_dynamic(
    const PeerSignals& signals,
    int64_t rank,
    int64_t world_size,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      ipc_mark_ready_and_wait_dynamic_kernel,
      dim3(1),
      dim3(64),
      0,
      stream,
      signals,
      rank,
      world_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mark_done_dynamic(
    const PeerSignals& signals,
    int64_t rank,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      ipc_mark_done_dynamic_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      signals,
      rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_synchronize_dynamic(
    const PeerSignals& signals,
    int64_t rank,
    int64_t world_size,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      ipc_synchronize_dynamic_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      signals,
      rank,
      world_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_close_and_wait(
    const PeerSignals& signals,
    int64_t rank,
    int64_t world_size,
    hipStream_t stream) {
  hipLaunchKernelGGL(
      ipc_close_and_wait_kernel,
      dim3(1),
      dim3(1),
      0,
      stream,
      signals,
      rank,
      world_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

class ROCmIPCCollectiveState {
 public:
  ROCmIPCCollectiveState(
      torch::Tensor staging,
      const std::vector<std::vector<int64_t>>& handles,
      const std::vector<int64_t>& offsets,
      int64_t rank)
      : rank_(rank),
        world_size_(static_cast<int64_t>(handles.size())),
        device_index_(staging.get_device()),
        capacity_bytes_(staging.numel() * staging.element_size() - kIPCControlBytes) {
    TORCH_CHECK(staging.is_cuda(), "ROCm IPC staging buffer must be on device");
    TORCH_CHECK(staging.is_contiguous(), "ROCm IPC staging buffer must be contiguous");
    TORCH_CHECK(staging.scalar_type() == torch::kUInt8,
                "ROCm IPC staging buffer must have dtype uint8");
    TORCH_CHECK(capacity_bytes_ > 0, "ROCm IPC staging capacity must be positive");
    TORCH_CHECK(
        world_size_ == 1 || world_size_ == 2 || world_size_ == 4 || world_size_ == 8,
        "ROCm IPC deterministic collectives require world size 1, 2, 4, or 8");
    TORCH_CHECK(offsets.size() == handles.size(), "one IPC offset is required per rank");
    TORCH_CHECK(rank_ >= 0 && rank_ < world_size_, "invalid ROCm IPC rank");

    set_peer_pointers(rank_, staging.data_ptr());
    try {
      for (int peer = 0; peer < world_size_; ++peer) {
        if (peer == rank_) {
          continue;
        }
        TORCH_CHECK(handles[peer].size() == sizeof(hipIpcMemHandle_t),
                    "invalid ROCm IPC handle size for rank ", peer);
        TORCH_CHECK(offsets[peer] >= 0, "invalid negative ROCm IPC offset");
        hipIpcMemHandle_t handle{};
        auto* raw_handle = reinterpret_cast<uint8_t*>(&handle);
        for (size_t byte = 0; byte < sizeof(handle); ++byte) {
          TORCH_CHECK(handles[peer][byte] >= 0 && handles[peer][byte] <= 255,
                      "invalid ROCm IPC handle byte for rank ", peer);
          raw_handle[byte] = static_cast<uint8_t>(handles[peer][byte]);
        }
        void* base = nullptr;
        C10_HIP_CHECK(hipIpcOpenMemHandle(
            &base,
            handle,
            hipIpcMemLazyEnablePeerAccess));
        imported_bases_[peer] = base;
        set_peer_pointers(
            peer,
            static_cast<uint8_t*>(base) + offsets[peer]);
      }
    } catch (...) {
      close_imports();
      throw;
    }
  }

  ~ROCmIPCCollectiveState() {
    int previous_device = -1;
    if (hipGetDevice(&previous_device) == hipSuccess && previous_device != device_index_) {
      if (hipSetDevice(device_index_) != hipSuccess) {
        return;
      }
    }
    close_imports();
    if (previous_device >= 0 && previous_device != device_index_) {
      C10_CUDA_IGNORE_ERROR(hipSetDevice(previous_device));
    }
  }

  int device_index() const {
    return device_index_;
  }

  void stage(torch::Tensor input, hipStream_t stream) {
    check_tensor(input, "input");
    const int64_t input_bytes = input.numel() * input.element_size();
    TORCH_CHECK(input_bytes <= capacity_bytes_,
                "input exceeds ROCm IPC staging capacity");
    launch_reserve_sequence(signals_, rank_, world_size_, stream);
    if (input_bytes > 0) {
      C10_HIP_CHECK(hipMemcpyAsync(
          const_cast<void*>(peers_.values[rank_]),
          input.data_ptr(),
          input_bytes,
          hipMemcpyDeviceToDevice,
          stream));
    }
    launch_mark_ready_and_wait_dynamic(signals_, rank_, world_size_, stream);
    staged_bytes_ = input_bytes;
    staged_type_ = input.scalar_type();
  }

  void prepare_staged(torch::Tensor input, hipStream_t stream) {
    check_tensor(input, "direct staging input");
    const int64_t input_bytes = input.numel() * input.element_size();
    TORCH_CHECK(input_bytes <= capacity_bytes_,
                "direct staging input exceeds ROCm IPC capacity");
    TORCH_CHECK(input.data_ptr() == peers_.values[rank_],
                "direct staging input must start at the local IPC payload");
    TORCH_CHECK(!staging_reserved_, "direct staging is already reserved");
    launch_reserve_sequence(signals_, rank_, world_size_, stream);
    staged_bytes_ = input_bytes;
    staged_type_ = input.scalar_type();
    staging_reserved_ = true;
  }

  void finish_staged(torch::Tensor input, hipStream_t stream) {
    check_tensor(input, "direct staging input");
    TORCH_CHECK(staging_reserved_, "direct staging must be reserved before reduction");
    TORCH_CHECK(input.data_ptr() == peers_.values[rank_],
                "direct staging input must start at the local IPC payload");
    TORCH_CHECK(input.numel() * input.element_size() == staged_bytes_,
                "direct staging input size changed after reserve");
    TORCH_CHECK(input.scalar_type() == staged_type_,
                "direct staging input dtype changed after reserve");
    launch_mark_ready_and_wait_dynamic(signals_, rank_, world_size_, stream);
    staging_reserved_ = false;
  }

  void all_reduce(torch::Tensor output, hipStream_t stream) const {
    check_reduction_output(output, staged_bytes_, "all_reduce");
    launch(output, 0, output.numel(), stream);
    launch_mark_done_dynamic(signals_, rank_, stream);
  }

  void reduce_scatter(torch::Tensor output, hipStream_t stream) const {
    check_reduction_output(
        output,
        staged_bytes_ / world_size_,
        "reduce_scatter");
    launch(
        output,
        rank_ * output.numel(),
        output.numel(),
        stream);
    launch_mark_done_dynamic(signals_, rank_, stream);
  }

  void reduce_scatter_many(
      const std::vector<torch::Tensor>& inputs,
      const std::vector<torch::Tensor>& outputs,
      hipStream_t stream) {
    TORCH_CHECK(!inputs.empty(), "reduce_scatter_many requires at least one input");
    TORCH_CHECK(inputs.size() == outputs.size(),
                "reduce_scatter_many input/output count mismatch");

    int64_t total_bytes = 0;
    const auto scalar_type = inputs.front().scalar_type();
    for (size_t index = 0; index < inputs.size(); ++index) {
      const auto& input = inputs[index];
      const auto& output = outputs[index];
      check_tensor(input, "reduce_scatter_many input");
      check_tensor(output, "reduce_scatter_many output");
      TORCH_CHECK(input.scalar_type() == scalar_type && output.scalar_type() == scalar_type,
                  "reduce_scatter_many dtype mismatch");
      const int64_t input_bytes = input.numel() * input.element_size();
      TORCH_CHECK(input_bytes == output.numel() * output.element_size() * world_size_,
                  "reduce_scatter_many output size mismatch");
      TORCH_CHECK(input_bytes <= capacity_bytes_ - total_bytes,
                  "reduce_scatter_many inputs exceed ROCm IPC staging capacity");
      total_bytes += input_bytes;
    }

    launch_reserve_sequence(signals_, rank_, world_size_, stream);
    int64_t byte_offset = 0;
    for (const auto& input : inputs) {
      const int64_t input_bytes = input.numel() * input.element_size();
      if (input_bytes > 0) {
        C10_HIP_CHECK(hipMemcpyAsync(
            static_cast<uint8_t*>(const_cast<void*>(peers_.values[rank_])) + byte_offset,
            input.data_ptr(),
            input_bytes,
            hipMemcpyDeviceToDevice,
            stream));
      }
      byte_offset += input_bytes;
    }
    launch_mark_ready_and_wait_dynamic(signals_, rank_, world_size_, stream);

    int64_t element_offset = 0;
    for (size_t index = 0; index < outputs.size(); ++index) {
      const auto& input = inputs[index];
      const auto& output = outputs[index];
      launch(
          output,
          element_offset + rank_ * output.numel(),
          output.numel(),
          stream);
      element_offset += input.numel();
    }
    launch_mark_done_dynamic(signals_, rank_, stream);
  }

  void all_gather(torch::Tensor output, hipStream_t stream) const {
    check_tensor(output, "all_gather");
    TORCH_CHECK(staged_type_ != at::ScalarType::Undefined, "stage must be called first");
    TORCH_CHECK(output.scalar_type() == staged_type_, "all_gather dtype mismatch");
    TORCH_CHECK(
        output.numel() * output.element_size() == staged_bytes_ * world_size_,
        "all_gather output size mismatch");
    launch_ipc_all_gather(
        peers_,
        output.data_ptr(),
        staged_bytes_,
        world_size_,
        stream);
    launch_mark_done_dynamic(signals_, rank_, stream);
  }

  void synchronize(hipStream_t stream) const {
    launch_synchronize_dynamic(signals_, rank_, world_size_, stream);
    launch_close_and_wait(signals_, rank_, world_size_, stream);
  }

 private:
  void set_peer_pointers(int peer, void* allocation_base) {
    auto* bytes = static_cast<uint8_t*>(allocation_base);
    signals_.ready[peer] = reinterpret_cast<uint64_t*>(bytes + kIPCReadyOffset);
    signals_.done[peer] = reinterpret_cast<uint64_t*>(bytes + kIPCDoneOffset);
    signals_.closed[peer] = reinterpret_cast<uint64_t*>(bytes + kIPCCloseOffset);
    signals_.sequence[peer] = reinterpret_cast<uint64_t*>(bytes + kIPCSequenceOffset);
    peers_.values[peer] = bytes + kIPCControlBytes;
  }

  void check_tensor(const torch::Tensor& tensor, const char* name) const {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a ROCm tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.get_device() == device_index_, name, " device mismatch");
  }

  void check_reduction_output(
      const torch::Tensor& output,
      int64_t expected_bytes,
      const char* name) const {
    check_tensor(output, name);
    TORCH_CHECK(staged_type_ != at::ScalarType::Undefined, "stage must be called first");
    TORCH_CHECK(output.scalar_type() == staged_type_, name, " dtype mismatch");
    TORCH_CHECK(output.numel() * output.element_size() == expected_bytes,
                name, " output size mismatch");
  }

  void launch(
      torch::Tensor output,
      int64_t input_offset,
      int64_t element_count,
      hipStream_t stream) const {
    if (element_count == 0) {
      return;
    }
    if (element_count % 2 == 0 && input_offset % 2 == 0) {
      if (output.scalar_type() == at::ScalarType::Half &&
          reinterpret_cast<uintptr_t>(output.data_ptr()) % alignof(__half2) == 0) {
        launch_ipc_fixed_tree_reduce_packed<__half2>(
            peers_,
            reinterpret_cast<__half2*>(output.data_ptr()),
            input_offset / 2,
            element_count / 2,
            world_size_,
            stream);
        return;
      }
      if (output.scalar_type() == at::ScalarType::BFloat16 &&
          reinterpret_cast<uintptr_t>(output.data_ptr()) %
                  alignof(__hip_bfloat162) ==
              0) {
        launch_ipc_fixed_tree_reduce_packed<__hip_bfloat162>(
            peers_,
            reinterpret_cast<__hip_bfloat162*>(output.data_ptr()),
            input_offset / 2,
            element_count / 2,
            world_size_,
            stream);
        return;
      }
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        output.scalar_type(),
        "deterministic_collective_rocm_ipc_fixed_tree",
        [&] {
          launch_ipc_fixed_tree_reduce<scalar_t>(
              peers_,
              output.data_ptr<scalar_t>(),
              input_offset,
              element_count,
              world_size_,
              stream);
        });
  }

  void close_imports() noexcept {
    for (int peer = 0; peer < world_size_; ++peer) {
      if (imported_bases_[peer] != nullptr) {
        C10_CUDA_IGNORE_ERROR(hipIpcCloseMemHandle(imported_bases_[peer]));
        imported_bases_[peer] = nullptr;
      }
    }
  }

  int64_t rank_;
  int64_t world_size_;
  int device_index_;
  int64_t capacity_bytes_;
  int64_t staged_bytes_{0};
  at::ScalarType staged_type_{at::ScalarType::Undefined};
  bool staging_reserved_{false};
  PeerPointers peers_{};
  PeerSignals signals_{};
  std::array<void*, kMaxWorldSize> imported_bases_{};
};

ROCmIPCCollectiveState* ipc_state(int64_t handle) {
  TORCH_CHECK(handle != 0, "ROCm IPC collective handle is closed");
  return reinterpret_cast<ROCmIPCCollectiveState*>(handle);
}

}  // namespace

void deterministic_collective_rocm_all_reduce(
    torch::Tensor rank_inputs,
    torch::Tensor output) {
  launch_dispatch(rank_inputs, output, "deterministic_collective_rocm_all_reduce");
}

void deterministic_collective_rocm_reduce_scatter(
    torch::Tensor rank_inputs,
    torch::Tensor output) {
  launch_dispatch(rank_inputs, output, "deterministic_collective_rocm_reduce_scatter");
}

torch::Tensor deterministic_collective_rocm_ipc_allocate(int64_t size_bytes) {
  TORCH_CHECK(size_bytes > 0, "ROCm IPC allocation size must be positive");
  int device_index = -1;
  C10_HIP_CHECK(hipGetDevice(&device_index));
  const int64_t allocation_bytes = size_bytes + kIPCControlBytes;
  void* pointer = nullptr;
  C10_HIP_CHECK(hipMalloc(&pointer, static_cast<size_t>(allocation_bytes)));
  C10_HIP_CHECK(hipMemset(pointer, 0, static_cast<size_t>(kIPCControlBytes)));
  const auto options = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .device(torch::Device(torch::kCUDA, device_index));
  return torch::from_blob(
      pointer,
      {allocation_bytes},
      [device_index](void* allocation) {
        int previous_device = -1;
        if (hipGetDevice(&previous_device) != hipSuccess) {
          return;
        }
        if (previous_device != device_index && hipSetDevice(device_index) != hipSuccess) {
          return;
        }
        C10_CUDA_IGNORE_ERROR(hipFree(allocation));
        if (previous_device != device_index) {
          C10_CUDA_IGNORE_ERROR(hipSetDevice(previous_device));
        }
      },
      options);
}

std::tuple<std::vector<int64_t>, int64_t>
deterministic_collective_rocm_ipc_meta(torch::Tensor tensor) {
  const c10::cuda::CUDAGuard device_guard(tensor.device());
  TORCH_CHECK(tensor.is_cuda(), "ROCm IPC tensor must be on device");
  TORCH_CHECK(tensor.is_contiguous(), "ROCm IPC tensor must be contiguous");
  TORCH_CHECK(tensor.numel() > 0, "ROCm IPC tensor must be non-empty");

  hipIpcMemHandle_t handle{};
  const hipError_t export_error = hipIpcGetMemHandle(&handle, tensor.data_ptr());
  TORCH_CHECK(
      export_error == hipSuccess,
      "hipIpcGetMemHandle failed: ",
      hipGetErrorString(export_error));
  const auto* raw_handle = reinterpret_cast<const uint8_t*>(&handle);
  std::vector<int64_t> bytes(sizeof(handle));
  for (size_t byte = 0; byte < sizeof(handle); ++byte) {
    bytes[byte] = raw_handle[byte];
  }
  return std::make_tuple(bytes, 0);
}

int64_t deterministic_collective_rocm_ipc_create(
    torch::Tensor staging,
    const std::vector<std::vector<int64_t>>& handles,
    const std::vector<int64_t>& offsets,
    int64_t rank) {
  const c10::cuda::CUDAGuard device_guard(staging.device());
  auto state = std::make_unique<ROCmIPCCollectiveState>(
      staging,
      handles,
      offsets,
      rank);
  return reinterpret_cast<int64_t>(state.release());
}

void deterministic_collective_rocm_ipc_destroy(int64_t handle) {
  delete ipc_state(handle);
}

void deterministic_collective_rocm_ipc_synchronize(int64_t handle) {
  auto* state = ipc_state(handle);
  const c10::cuda::CUDAGuard device_guard(
      torch::Device(torch::kCUDA, state->device_index()));
  const auto stream = at::cuda::getCurrentCUDAStream();
  state->synchronize(stream);
}

void deterministic_collective_rocm_ipc_stage(int64_t handle, torch::Tensor input) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  ipc_state(handle)->stage(input, stream);
}

void deterministic_collective_rocm_ipc_prepare_staged(
    int64_t handle,
    torch::Tensor input) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  ipc_state(handle)->prepare_staged(input, stream);
}

void deterministic_collective_rocm_ipc_all_reduce_staged(
    int64_t handle,
    torch::Tensor input,
    torch::Tensor output) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto* state = ipc_state(handle);
  state->finish_staged(input, stream);
  state->all_reduce(output, stream);
}

void deterministic_collective_rocm_ipc_all_reduce(
    int64_t handle,
    torch::Tensor output) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  ipc_state(handle)->all_reduce(output, stream);
}

void deterministic_collective_rocm_ipc_all_reduce_input(
    int64_t handle,
    torch::Tensor input,
    torch::Tensor output) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto* state = ipc_state(handle);
  state->stage(input, stream);
  state->all_reduce(output, stream);
}

void deterministic_collective_rocm_ipc_reduce_scatter(
    int64_t handle,
    torch::Tensor output) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  ipc_state(handle)->reduce_scatter(output, stream);
}

void deterministic_collective_rocm_ipc_reduce_scatter_input(
    int64_t handle,
    torch::Tensor input,
    torch::Tensor output) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto* state = ipc_state(handle);
  state->stage(input, stream);
  state->reduce_scatter(output, stream);
}

void deterministic_collective_rocm_ipc_reduce_scatter_many(
    int64_t handle,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs) {
  TORCH_CHECK(!inputs.empty(), "reduce_scatter_many requires at least one input");
  const c10::cuda::CUDAGuard device_guard(inputs.front().device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  ipc_state(handle)->reduce_scatter_many(inputs, outputs, stream);
}

void deterministic_collective_rocm_ipc_all_gather(
    int64_t handle,
    torch::Tensor output) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  ipc_state(handle)->all_gather(output, stream);
}

void deterministic_collective_rocm_ipc_all_gather_input(
    int64_t handle,
    torch::Tensor input,
    torch::Tensor output) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto* state = ipc_state(handle);
  state->stage(input, stream);
  state->all_gather(output, stream);
}
