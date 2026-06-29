// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>
#include <cuda_bf16.h>

// Fused LogP Declarations
torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids);

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_SM90)
torch::Tensor fused_logp_sm90_forward(torch::Tensor logits, torch::Tensor labels);
std::vector<torch::Tensor> fused_linear_logp_sm90_forward(torch::Tensor hidden,
                                                          torch::Tensor weight,
                                                          torch::Tensor target,
                                                          torch::optional<torch::Tensor> bias);
#endif

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_CUDA)
torch::Tensor fused_logp_forward_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor output);
torch::Tensor fused_logp_forward_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_indexed_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices, torch::Tensor output);
torch::Tensor fused_logp_forward_indexed_fp32(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices);
torch::Tensor fused_logp_forward_online_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor output);
torch::Tensor fused_logp_forward_online_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_online_indexed_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices, torch::Tensor output);
torch::Tensor fused_logp_forward_online_indexed_fp32(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices);

// RMSNorm Declarations & Wrappers

void rmsnorm_forward_cuda(
  torch::Tensor x,
  torch::Tensor weight,
  torch::Tensor y,
  torch::Tensor rstd,
  double eps);

void rmsnorm_backward_dx_cuda(
  torch::Tensor dy,
  torch::Tensor x,
  torch::Tensor weight,
  torch::Tensor rstd,
  torch::Tensor dx);

void rmsnorm_backward_partial_dw_cuda(
  torch::Tensor dy,
  torch::Tensor x,
  torch::Tensor rstd,
  torch::Tensor mask,
  torch::Tensor partial_dw);

void rmsnorm_backward_reduce_dw_cuda(
  torch::Tensor partial_dw,
  torch::Tensor dw);

static void rmsnorm_check_input(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

std::vector<torch::Tensor> rmsnorm_forward(
  torch::Tensor x,
  torch::Tensor weight,
  double eps)
{
  rmsnorm_check_input(x, "x");
  rmsnorm_check_input(weight, "weight");

  TORCH_CHECK(x.dim() == 2, "x must be 2D [T, H]");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D [H]");
  TORCH_CHECK(x.size(1) == weight.size(0), "x.size(1) must equal weight.size(0)");

  auto T = x.size(0);
  auto y = torch::empty_like(x);
  auto rstd = torch::empty({T}, x.options().dtype(torch::kFloat32));

  rmsnorm_forward_cuda(x, weight, y, rstd, eps);

  return {y, rstd};
}

torch::Tensor rmsnorm_backward_dx(
  torch::Tensor dy,
  torch::Tensor x,
  torch::Tensor weight,
  torch::Tensor rstd)
{
  rmsnorm_check_input(dy, "dy");
  rmsnorm_check_input(x, "x");
  rmsnorm_check_input(weight, "weight");
  rmsnorm_check_input(rstd, "rstd");

  TORCH_CHECK(dy.sizes() == x.sizes(), "dy and x must have same shape");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [T, H]");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D [H]");
  TORCH_CHECK(rstd.dim() == 1, "rstd must be 1D [T]");
  TORCH_CHECK(rstd.size(0) == x.size(0), "rstd.size(0) must equal x.size(0)");

  auto dx = torch::empty_like(x);

  rmsnorm_backward_dx_cuda(dy, x, weight, rstd, dx);

  return dx;
}

torch::Tensor rmsnorm_backward_dw(
  torch::Tensor dy,
  torch::Tensor x,
  torch::Tensor rstd,
  torch::Tensor mask)
{
  rmsnorm_check_input(dy, "dy");
  rmsnorm_check_input(x, "x");
  rmsnorm_check_input(rstd, "rstd");
  rmsnorm_check_input(mask, "mask");

  TORCH_CHECK(dy.sizes() == x.sizes(), "dy and x must have same shape");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [T, H]");
  TORCH_CHECK(rstd.dim() == 1, "rstd must be 1D [T]");
  TORCH_CHECK(mask.dim() == 1, "mask must be 1D [T]");
  TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");
  TORCH_CHECK(rstd.size(0) == x.size(0), "rstd.size(0) must equal x.size(0)");
  TORCH_CHECK(mask.size(0) == x.size(0), "mask.size(0) must equal x.size(0)");

  auto T = x.size(0);
  auto H = x.size(1);

  auto partial_dw = torch::empty({T, H}, x.options().dtype(torch::kFloat32));
  auto dw = torch::empty({H}, x.options().dtype(torch::kFloat32));

  rmsnorm_backward_partial_dw_cuda(dy, x, rstd, mask, partial_dw);
  rmsnorm_backward_reduce_dw_cuda(partial_dw, dw);

  return dw;
}

// Prefix-Shared Attention Declarations & Wrappers

void prefix_shared_attention_forward(
  const __nv_bfloat16 *Q,  // [bs, G, len_q, DIM]
  const __nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const __nv_bfloat16 *V,  // [bs, len_kv, DIM]
  __nv_bfloat16 *O,        // [bs, G, len_q, DIM]
  int bs,
  int G,
  int len_q,
  int len_kv,
  int dim);

at::Tensor prefix_shared_attention(
  const at::Tensor& Q,
  const at::Tensor& K,
  const at::Tensor& V)
{
  TORCH_CHECK(Q.dim() == 4, "Q must be [bs, G, len_q, DIM]");
  TORCH_CHECK(K.dim() == 3, "K must be [bs, len_kv, DIM]");
  TORCH_CHECK(V.dim() == 3, "V must be [bs, len_kv, DIM]");

  TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Only BFloat16 is supported");
  TORCH_CHECK(Q.is_cuda() && Q.is_contiguous(), "Tensors must be CUDA and contiguous");
  TORCH_CHECK(K.is_cuda() && K.is_contiguous(), "Tensors must be CUDA and contiguous");
  TORCH_CHECK(V.is_cuda() && V.is_contiguous(), "Tensors must be CUDA and contiguous");

  const int bs = Q.size(0);
  const int G = Q.size(1);
  const int len_q = Q.size(2);
  const int dim = Q.size(3);
  const int len_kv = K.size(1);

  at::Tensor O = at::empty_like(Q);

  auto Q_ptr = reinterpret_cast<const __nv_bfloat16 *>(Q.data_ptr());
  auto K_ptr = reinterpret_cast<const __nv_bfloat16 *>(K.data_ptr());
  auto V_ptr = reinterpret_cast<const __nv_bfloat16 *>(V.data_ptr());
  auto O_ptr = reinterpret_cast<__nv_bfloat16 *>(O.data_ptr());

  prefix_shared_attention_forward(Q_ptr, K_ptr, V_ptr, O_ptr, bs, G, len_q, len_kv, dim);

  return O;
}
#endif

// PyBind11 Module Registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RL-Kernel High-Performance Operator Extension Library";

    m.def("fused_logp", &fused_logp_forward, "Fused logp forward fallback");

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_SM90)
    m.def("fused_logp_sm90", &fused_logp_sm90_forward, "TMA-accelerated Online Softmax Fused LogP");
    m.def("fused_linear_logp_sm90", &fused_linear_logp_sm90_forward,
          "TMA+WGMMA fused linear log-prob (hidden @ W^T -> selected-token logp), SM90");
#endif

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_CUDA)
    m.def("fused_logp_forward_out", &fused_logp_forward_out, "Fused logp out");
    m.def("fused_logp_forward_fp32", &fused_logp_forward_fp32, "Fused logp fp32");
    m.def("fused_logp_forward_indexed_out", &fused_logp_forward_indexed_out, "Fused logp indexed out");
    m.def("fused_logp_forward_indexed_fp32", &fused_logp_forward_indexed_fp32, "Fused logp indexed fp32");
    m.def("fused_logp_forward_online_out", &fused_logp_forward_online_out, "Fused logp online out");
    m.def("fused_logp_forward_online_fp32", &fused_logp_forward_online_fp32, "Fused logp online fp32");
    m.def("fused_logp_forward_online_indexed_out", &fused_logp_forward_online_indexed_out, "Fused logp online indexed out");
    m.def("fused_logp_forward_online_indexed_fp32", &fused_logp_forward_online_indexed_fp32, "Fused logp online indexed fp32");

    // registry Prefix-Shared Attention
    m.def("prefix_shared_attention", &prefix_shared_attention, "Prefix-Shared Fused Attention for GRPO");

    // registry RMSNorm
    m.def("rmsnorm_forward", &rmsnorm_forward, "Batch-invariant RMSNorm forward CUDA");
    m.def("rmsnorm_backward_dx", &rmsnorm_backward_dx, "Batch-invariant RMSNorm backward dx CUDA");
    m.def("rmsnorm_backward_dw", &rmsnorm_backward_dw, "Deterministic RMSNorm backward dweight CUDA");
#endif
}
