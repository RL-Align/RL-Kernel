// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t, bool backward>
__global__ void copy_rows_kernel(
    const scalar_t* text,
    const scalar_t* video,
    const scalar_t* audio,
    const int64_t* text_indices,
    const int64_t* video_indices,
    const int64_t* audio_indices,
    scalar_t* packed,
    scalar_t* d_text,
    scalar_t* d_video,
    scalar_t* d_audio,
    int64_t count,
    int64_t text_rows,
    int64_t video_rows,
    int64_t source_rows,
    int64_t sequence_length,
    int64_t hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const int64_t column = i % hidden;
  const int64_t row = (i / hidden) % source_rows;
  const int64_t batch = i / (source_rows * hidden);
  const scalar_t* source;
  scalar_t* gradient;
  int64_t source_row;
  int64_t destination_row;
  int64_t rows;
  if (row < text_rows) {
    source = text;
    gradient = d_text;
    source_row = row;
    destination_row = text_indices[source_row];
    rows = text_rows;
  } else if (row < text_rows + video_rows) {
    source = video;
    gradient = d_video;
    source_row = row - text_rows;
    destination_row = video_indices[source_row];
    rows = video_rows;
  } else {
    source = audio;
    gradient = d_audio;
    source_row = row - text_rows - video_rows;
    destination_row = audio_indices[source_row];
    rows = source_rows - text_rows - video_rows;
  }
  const int64_t source_offset = (batch * rows + source_row) * hidden + column;
  const int64_t packed_offset = (batch * sequence_length + destination_row) * hidden + column;
  if constexpr (backward) {
    gradient[source_offset] = packed[packed_offset];
  } else {
    packed[packed_offset] = source[source_offset];
  }
}

void check_source(const torch::Tensor& source, const char* name) {
  TORCH_CHECK(source.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(source.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(!source.is_neg(), name, " must resolve the negative view bit before native access");
  TORCH_CHECK(source.dim() == 3 && source.size(0) > 0 && source.size(2) > 0,
              name, " must have shape [B, S, D] with non-empty batch and hidden dimensions");
  TORCH_CHECK(source.scalar_type() == at::kFloat || source.scalar_type() == at::kHalf ||
                  source.scalar_type() == at::kBFloat16,
              name, " must be fp16, bf16, or fp32");
}

void check_indices(
    const torch::Tensor& reference,
    const torch::Tensor& text_indices,
    const torch::Tensor& video_indices,
    const torch::Tensor& audio_indices,
    int64_t sequence_length) {
  TORCH_CHECK(sequence_length > 0, "sequence_length must be positive");
  for (const auto& indices : {text_indices, video_indices, audio_indices}) {
    TORCH_CHECK(indices.device() == reference.device(), "sources and indices must share a device");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(!indices.is_neg(), "indices must resolve the negative view bit before native access");
    TORCH_CHECK(indices.scalar_type() == at::kLong && indices.dim() == 1,
                "indices must be one-dimensional int64 tensors");
  }
  const auto ordered = std::get<0>(torch::cat({text_indices, video_indices, audio_indices}).sort());
  const int64_t rows = ordered.numel();
  TORCH_CHECK(rows > 0, "at least one modality must contain a source row");
  const auto in_range = (ordered[0] >= 0).logical_and(ordered[rows - 1] < sequence_length);
  const auto unique = (ordered.slice(0, 1) > ordered.slice(0, 0, rows - 1)).all();
  // The single host check is intentional: invalid layouts must fail before
  // any scatter/gather memory access. Sorting only validates integer metadata.
  TORCH_CHECK(in_range.logical_and(unique).item<bool>(),
              "indices must be in range, unique and disjoint");
}

int64_t blocks_for(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  TORCH_CHECK(blocks <= 2147483647, "tensor is too large for the launch grid");
  return blocks;
}

}  // namespace

torch::Tensor packed_index_copy_forward(
    torch::Tensor text,
    torch::Tensor video,
    torch::Tensor audio,
    torch::Tensor text_indices,
    torch::Tensor video_indices,
    torch::Tensor audio_indices,
    int64_t sequence_length) {
  check_source(text, "text");
  check_source(video, "video");
  check_source(audio, "audio");
  for (const auto& source : {video, audio}) {
    TORCH_CHECK(source.device() == text.device(), "sources must share a device");
    TORCH_CHECK(source.scalar_type() == text.scalar_type(), "native sources must share dtype");
    TORCH_CHECK(source.size(0) == text.size(0) && source.size(2) == text.size(2),
                "sources must share batch and hidden dimensions");
  }
  TORCH_CHECK(text_indices.numel() == text.size(1) && video_indices.numel() == video.size(1) &&
                  audio_indices.numel() == audio.size(1),
              "each modality must have one index per source row");
  const c10::cuda::CUDAGuard guard(text.device());
  check_indices(text, text_indices, video_indices, audio_indices, sequence_length);
  auto out = torch::zeros({text.size(0), sequence_length, text.size(2)}, text.options());
  const int64_t rows = text.size(1) + video.size(1) + audio.size(1);
  const int64_t count = text.numel() + video.numel() + audio.numel();
  const auto blocks = blocks_for(count);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, text.scalar_type(), "packed_index_copy_forward", [&] {
        copy_rows_kernel<scalar_t, false><<<blocks, kThreads, 0, stream>>>(
            text.data_ptr<scalar_t>(), video.data_ptr<scalar_t>(), audio.data_ptr<scalar_t>(),
            text_indices.data_ptr<int64_t>(), video_indices.data_ptr<int64_t>(),
            audio_indices.data_ptr<int64_t>(), out.data_ptr<scalar_t>(), nullptr, nullptr, nullptr,
            count, text.size(1), video.size(1), rows, sequence_length, text.size(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> packed_index_copy_backward(
    torch::Tensor grad,
    torch::Tensor text_indices,
    torch::Tensor video_indices,
    torch::Tensor audio_indices) {
  check_source(grad, "grad");
  const c10::cuda::CUDAGuard guard(grad.device());
  check_indices(grad, text_indices, video_indices, audio_indices, grad.size(1));
  auto d_text = torch::empty({grad.size(0), text_indices.numel(), grad.size(2)}, grad.options());
  auto d_video = torch::empty({grad.size(0), video_indices.numel(), grad.size(2)}, grad.options());
  auto d_audio = torch::empty({grad.size(0), audio_indices.numel(), grad.size(2)}, grad.options());
  const int64_t rows = text_indices.numel() + video_indices.numel() + audio_indices.numel();
  const int64_t count = d_text.numel() + d_video.numel() + d_audio.numel();
  const auto blocks = blocks_for(count);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, grad.scalar_type(), "packed_index_copy_backward", [&] {
        copy_rows_kernel<scalar_t, true><<<blocks, kThreads, 0, stream>>>(
            nullptr, nullptr, nullptr, text_indices.data_ptr<int64_t>(),
            video_indices.data_ptr<int64_t>(), audio_indices.data_ptr<int64_t>(),
            grad.data_ptr<scalar_t>(), d_text.data_ptr<scalar_t>(), d_video.data_ptr<scalar_t>(),
            d_audio.data_ptr<scalar_t>(), count, text_indices.numel(), video_indices.numel(),
            rows, grad.size(1), grad.size(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d_text, d_video, d_audio};
}
