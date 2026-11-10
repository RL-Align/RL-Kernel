// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>
#include <cuda_bf16.h>

// Fused LogP Declarations
torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids);
#if !defined(USE_ROCM) && !defined(KERNEL_ALIGN_WITH_ROCM)
std::vector<torch::Tensor> ordered_logp_merge(torch::Tensor lse, torch::Tensor target);
#endif

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_SM90)
torch::Tensor fused_logp_sm90_forward(torch::Tensor logits, torch::Tensor labels);
std::vector<torch::Tensor> fused_linear_logp_sm90_forward(torch::Tensor hidden,
                                                          torch::Tensor weight,
                                                          torch::Tensor target,
                                                          torch::optional<torch::Tensor> bias,
                                                          torch::optional<torch::Tensor> temperature,
                                                          bool return_logits,
                                                          int64_t real_vocab_size);
std::vector<torch::Tensor> batch_invariant_logp_sm90_forward(torch::Tensor logits,
                                                             torch::Tensor target,
                                                             int64_t ignore_index);
std::vector<torch::Tensor> fused_linear_logp_sm90_global_target_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor target,
    torch::optional<torch::Tensor> bias,
    int64_t vocab_start_index,
    torch::optional<torch::Tensor> temperature,
    int64_t real_vocab_size);
std::vector<torch::Tensor> fused_linear_logp_sm90_backward(torch::Tensor grad_logp,
                                                           torch::Tensor hidden,
                                                           torch::Tensor weight,
                                                           torch::Tensor target,
                                                           torch::Tensor lse,
                                                           torch::optional<torch::Tensor> bias,
                                                           int64_t vocab_start_index,
                                                           bool compute_grad_hidden,
                                                           bool compute_grad_weight,
                                                           bool compute_grad_bias,
                                                           bool use_global_lse);
std::vector<torch::Tensor> linear_logp_probs_bf16_forward(torch::Tensor logits,
                                                          torch::Tensor target,
                                                          int64_t vocab_start_index);
std::vector<torch::Tensor> linear_logp_bf16_forward(torch::Tensor logits,
                                                    torch::Tensor target,
                                                    int64_t vocab_start_index);
std::vector<torch::Tensor> linear_logp_local_probs_bf16_forward(torch::Tensor logits,
                                                                torch::Tensor target,
                                                                int64_t vocab_start_index);
std::vector<torch::Tensor> linear_logp_local_bf16_forward(torch::Tensor logits,
                                                          torch::Tensor target,
                                                          int64_t vocab_start_index);
std::vector<torch::Tensor> linear_logp_top_p_local_bf16_forward(
    torch::Tensor logits,
    torch::Tensor target,
    torch::Tensor replay_ids,
    torch::Tensor replay_logprobs,
    torch::Tensor temperature,
    int64_t vocab_start_index);
torch::Tensor linear_logp_probs_bf16_to_dlogits_(torch::Tensor probs,
                                                 torch::Tensor target,
                                                 torch::Tensor grad_logp,
                                                 int64_t vocab_start_index);
torch::Tensor linear_logp_local_probs_bf16_to_dlogits_(torch::Tensor probs,
                                                       torch::Tensor target,
                                                       torch::Tensor grad_logp,
                                                       torch::Tensor local_lse,
                                                       torch::Tensor global_lse,
                                                       int64_t vocab_start_index);
torch::Tensor linear_logp_logits_bf16_to_dlogits(torch::Tensor logits,
                                                 torch::Tensor dlogits,
                                                 torch::Tensor target,
                                                 torch::Tensor grad_logp,
                                                 torch::Tensor lse,
                                                 int64_t vocab_start_index);
// RoPE (rotate-half) apply for SM90; cos/sin precomputed fp32, sin_sign = +1 fwd / -1 bwd.
torch::Tensor rope_apply_sm90(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, double sin_sign);
torch::Tensor embedding_sm90_forward(torch::Tensor token_ids, torch::Tensor weight);
torch::Tensor embedding_sm90_forward_fp32(torch::Tensor token_ids, torch::Tensor weight);
torch::Tensor lm_head_sm90_forward(torch::Tensor hidden,
                                   torch::Tensor weight,
                                   torch::optional<torch::Tensor> bias);
torch::Tensor lm_head_sm90_forward_fp32(torch::Tensor hidden,
                                        torch::Tensor weight,
                                        torch::optional<torch::Tensor> bias);
#endif

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_CUDA) || defined(KERNEL_ALIGN_WITH_ROCM)
torch::Tensor fused_logp_forward_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor output);
torch::Tensor fused_logp_forward_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_indexed_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices, torch::Tensor output);
torch::Tensor fused_logp_forward_indexed_fp32(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices);
torch::Tensor fused_logp_forward_online_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor output);
torch::Tensor fused_logp_forward_online_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor fused_logp_forward_online_indexed_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices, torch::Tensor output);
torch::Tensor fused_logp_forward_online_indexed_fp32(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices);
torch::Tensor deterministic_logp_forward(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor deterministic_logp_forward_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor output);
torch::Tensor deterministic_logp_forward_fp32(torch::Tensor logits, torch::Tensor token_ids);
torch::Tensor deterministic_logp_forward_indexed_out(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices, torch::Tensor output);
torch::Tensor deterministic_logp_forward_indexed_fp32(torch::Tensor logits, torch::Tensor token_ids, torch::Tensor row_indices);
std::vector<torch::Tensor> deterministic_logp_tile_stats(
    torch::Tensor logits,
    int64_t vocab_start,
    int64_t real_vocab,
    int64_t num_tiles);
torch::Tensor deterministic_logp_backward(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor coef_logp,
    torch::Tensor coef_lse,
    torch::Tensor target_local,
    int64_t vocab_start,
    int64_t real_vocab,
    bool has_lse_grad);

// ROCm-tuned WS2 vocab-parallel logprob kernels (csrc/rocm/logprob/hip_deterministic_logp_kernel.hip).
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
std::vector<torch::Tensor> hip_deterministic_logp_tile_stats(
    torch::Tensor logits,
    int64_t vocab_start,
    int64_t real_vocab,
    int64_t num_tiles);
torch::Tensor hip_deterministic_logp_backward(
    torch::Tensor logits,
    torch::Tensor lse,
    torch::Tensor coef_logp,
    torch::Tensor coef_lse,
    torch::Tensor target_local,
    int64_t vocab_start,
    int64_t real_vocab,
    bool has_lse_grad);
std::vector<torch::Tensor> hip_sparse_nucleus_logp_forward(
    torch::Tensor logits,
    torch::Tensor ids,
    torch::Tensor valid,
    torch::Tensor targets,
    torch::Tensor inverse_temperature);
std::vector<torch::Tensor> hip_replicated_sparse_nucleus_logp(
    torch::Tensor logits, torch::Tensor ids, torch::Tensor targets,
    double inverse_temperature, bool transport_sum);
torch::Tensor hip_sparse_nucleus_logp_backward(
    torch::Tensor logits,
    torch::Tensor ids,
    torch::Tensor valid,
    torch::Tensor targets,
    torch::Tensor inverse_temperature,
    torch::Tensor lse,
    torch::Tensor grad_result,
    torch::Tensor grad_lse);
#endif

#if !defined(USE_ROCM) && !defined(KERNEL_ALIGN_WITH_ROCM)
// Single-node TP=8 deterministic CUDA IPC collectives. ROCm uses the
// rank-ordered RCCL transport in rl_engine.distributed.algorithms.collectives.
std::tuple<std::vector<int64_t>, int64_t> deterministic_collective_ipc_meta(
    torch::Tensor& tensor);
int64_t deterministic_collective_create(
    torch::Tensor& staging,
    const std::vector<std::vector<int64_t>>& handles,
    const std::vector<int64_t>& offsets,
    int64_t rank);
void deterministic_collective_destroy(int64_t handle);
void deterministic_collective_stage(int64_t handle, torch::Tensor& input);
void deterministic_collective_all_reduce(int64_t handle, torch::Tensor& output);
void deterministic_collective_prepare_staged(int64_t handle, torch::Tensor& input);
void deterministic_collective_all_reduce_staged(
    int64_t handle, torch::Tensor& input, torch::Tensor& output);
void deterministic_collective_all_reduce_fused(
    int64_t handle, torch::Tensor& input, torch::Tensor& output);
void deterministic_collective_reduce_scatter(int64_t handle, torch::Tensor& output);
void deterministic_collective_all_gather(int64_t handle, torch::Tensor& output);
void deterministic_collective_all_gather_fused(
    int64_t handle, torch::Tensor& input, torch::Tensor& output);
void deterministic_collective_all_gather_many(
    int64_t handle,
    std::vector<torch::Tensor> inputs,
    std::vector<torch::Tensor> outputs);
#endif

#if defined(KERNEL_ALIGN_WITH_ROCM)
// ROCm keeps arithmetic in a fixed balanced tree while using either RCCL or
// HIP IPC for rank-ordered transport. These kernels expose the local and IPC
// reduction paths without changing the CUDA implementation.
void deterministic_collective_rocm_all_reduce(
    torch::Tensor rank_inputs,
    torch::Tensor output);
void deterministic_collective_rocm_reduce_scatter(
    torch::Tensor rank_inputs,
    torch::Tensor output);
torch::Tensor deterministic_collective_rocm_ipc_allocate(int64_t size_bytes);
std::tuple<std::vector<int64_t>, int64_t>
deterministic_collective_rocm_ipc_meta(torch::Tensor tensor);
int64_t deterministic_collective_rocm_ipc_create(
    torch::Tensor staging,
    const std::vector<std::vector<int64_t>>& handles,
    const std::vector<int64_t>& offsets,
    int64_t rank);
void deterministic_collective_rocm_ipc_synchronize(int64_t handle);
void deterministic_collective_rocm_ipc_destroy(int64_t handle);
void deterministic_collective_rocm_ipc_stage(int64_t handle, torch::Tensor input);
void deterministic_collective_rocm_ipc_prepare_staged(
    int64_t handle,
    torch::Tensor input);
void deterministic_collective_rocm_ipc_all_reduce_staged(
    int64_t handle,
    torch::Tensor input,
    torch::Tensor output);
void deterministic_collective_rocm_ipc_all_reduce(
    int64_t handle,
    torch::Tensor output);
void deterministic_collective_rocm_ipc_all_reduce_input(
    int64_t handle,
    torch::Tensor input,
    torch::Tensor output);
void deterministic_collective_rocm_ipc_reduce_scatter(
    int64_t handle,
    torch::Tensor output);
void deterministic_collective_rocm_ipc_reduce_scatter_input(
    int64_t handle,
    torch::Tensor input,
    torch::Tensor output);
void deterministic_collective_rocm_ipc_reduce_scatter_many(
    int64_t handle,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs);
void deterministic_collective_rocm_ipc_all_gather(
    int64_t handle,
    torch::Tensor output);
void deterministic_collective_rocm_ipc_all_gather_input(
    int64_t handle,
    torch::Tensor input,
    torch::Tensor output);
#endif

// Batch-Invariant Deterministic GEMM Declarations
bool det_gemm_sm90_compiled();
torch::Tensor det_gemm_fwd(torch::Tensor a, torch::Tensor b);
torch::Tensor det_gemm_fwd_rhs_transposed(torch::Tensor a, torch::Tensor bt);
torch::Tensor det_gemm_da(torch::Tensor dc, torch::Tensor b);
torch::Tensor det_gemm_db(torch::Tensor a, torch::Tensor dc);
torch::Tensor det_gemm_db_transposed(torch::Tensor a, torch::Tensor dc);
// SiLU / SwiGLU Declarations (elementwise activation, general CUDA)
torch::Tensor silu_forward_cuda(torch::Tensor x);
torch::Tensor silu_backward_cuda(torch::Tensor dy, torch::Tensor x);
torch::Tensor swiglu_forward_cuda(torch::Tensor gate, torch::Tensor up);
std::vector<torch::Tensor> swiglu_backward_cuda(
    torch::Tensor dy,
    torch::Tensor gate,
    torch::Tensor up);
torch::Tensor swiglu_packed_forward_cuda(torch::Tensor gate_up);
std::vector<torch::Tensor> swiglu_packed_backward_cuda(
    torch::Tensor dy,
    torch::Tensor gate_up);

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

int64_t rmsnorm_backward_dw_chunks_cuda(int64_t rows);

#if !defined(USE_ROCM)
void reduce_rows_fp32_left_fold_cuda(
  torch::Tensor rows,
  torch::Tensor output);
#endif

static void rmsnorm_check_input(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
}

// Muse-Glimmer [F1]: post-norm, residual add, pre-norm. csrc/cuda/norm/fused_add_norm.cu
void fused_add_norm_forward_cuda(
    torch::Tensor hidden, torch::Tensor residual,
    torch::Tensor post_w, torch::Tensor pre_w,
    torch::Tensor res_out, torch::Tensor pre_out,
    torch::Tensor rstd_post, torch::Tensor rstd_pre,
    double post_eps, double pre_eps);
void fused_add_norm_backward_cuda(
    torch::Tensor g_res, torch::Tensor g_pre,
    torch::Tensor hidden, torch::Tensor res_out,
    torch::Tensor post_w, torch::Tensor pre_w,
    torch::Tensor rstd_post, torch::Tensor rstd_pre,
    torch::Tensor d_hidden, torch::Tensor d_residual,
    torch::Tensor partial_post_w, torch::Tensor partial_pre_w);

// FP64 has to be rejected here. AT_DISPATCH_FLOATING_TYPES_AND2 instantiates
// double, and the kernel's generic load/store cast through float, so a FP64 call
// returned FP64 tensors carrying FP32 precision -- about 7e-07 of error -- with no
// error raised. The Python wrappers reject FP64 first, so this is only reachable
// by calling the binding directly, but failing open on precision is the wrong
// default for an operator whose whole contract is about where the casts are.
static void fused_add_norm_check_dtype(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(
      t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kFloat16 ||
          t.scalar_type() == torch::kBFloat16,
      name, " must have dtype fp16, bf16, or fp32, got ", t.scalar_type());
}

static void fused_add_norm_check_pair(
    const torch::Tensor& a, const char* an, const torch::Tensor& b, const char* bn) {
  rmsnorm_check_input(a, an);
  rmsnorm_check_input(b, bn);
  fused_add_norm_check_dtype(a, an);
  fused_add_norm_check_dtype(b, bn);
  TORCH_CHECK(a.sizes() == b.sizes(), an, " and ", bn, " must share shape");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), an, " and ", bn, " must share dtype");
  TORCH_CHECK(a.device() == b.device(), an, " and ", bn, " must be on the same device, got ",
              a.device(), " and ", b.device());
}

// Every tensor has to be on hidden's device, because the launchers take their
// CUDAGuard from hidden alone. A residual or a weight on another device is then
// dereferenced as if it lived on hidden's, which is an illegal access at best and
// silent garbage at worst, and neither says what went wrong.
static void fused_add_norm_check_device(
    const torch::Tensor& ref, const char* rn, const torch::Tensor& t, const char* tn) {
  TORCH_CHECK(t.device() == ref.device(), tn, " must be on the same device as ", rn,
              ", got ", t.device(), " and ", ref.device());
}

// rstd_post / rstd_pre come back from the forward and go into the backward, so a
// direct caller can hand over anything shaped [T]. The kernel reads them through
// data_ptr<float>() with an implicit stride of one: data_ptr throws on a wrong
// dtype, but a strided 1-D view such as rstd[::2] is read as if it were packed and
// silently produces different gradients. Measured on a 4x64 row: the two calls
// disagree.
static void fused_add_norm_check_rstd(
    const torch::Tensor& t, const char* name, int64_t rows, const torch::Tensor& ref) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32, got ",
              t.scalar_type());
  TORCH_CHECK(t.dim() == 1 && t.size(0) == rows, name, " must be [T] with T=", rows,
              ", got shape ", t.sizes());
  fused_add_norm_check_device(ref, "hidden", t, name);
}

std::vector<torch::Tensor> fused_add_norm_forward(
  torch::Tensor hidden, torch::Tensor residual,
  torch::Tensor post_w, torch::Tensor pre_w,
  double post_eps, double pre_eps)
{
  fused_add_norm_check_pair(hidden, "hidden", residual, "residual");
  fused_add_norm_check_pair(post_w, "post_weight", pre_w, "pre_weight");
  fused_add_norm_check_device(hidden, "hidden", post_w, "post_weight");
  fused_add_norm_check_device(hidden, "hidden", pre_w, "pre_weight");
  TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D [T, H]");
  // A zero feature dimension passes every other check -- dim 2, weight 1-D, 0 == 0 --
  // and reaches the kernel, where the row mean is div_rn(sum, 0.0f) and every rstd
  // comes back NaN. Measured through this entry point: rstd_post was [nan, nan, nan,
  // nan]. The Python wrapper rejects it first; this binding is public.
  TORCH_CHECK(hidden.size(1) > 0, "hidden.size(1) must be non-zero");
  TORCH_CHECK(post_w.dim() == 1, "post_weight must be 1D [H]");
  TORCH_CHECK(hidden.size(1) == post_w.size(0), "hidden.size(1) must equal weight size");

  auto T = hidden.size(0);
  auto res_out = torch::empty_like(hidden);
  auto pre_out = torch::empty_like(hidden);
  auto f32 = hidden.options().dtype(torch::kFloat32);
  auto rstd_post = torch::empty({T}, f32);
  auto rstd_pre = torch::empty({T}, f32);

  fused_add_norm_forward_cuda(hidden, residual, post_w, pre_w,
                              res_out, pre_out, rstd_post, rstd_pre, post_eps, pre_eps);
  return {res_out, pre_out, rstd_post, rstd_pre};
}

std::vector<torch::Tensor> fused_add_norm_backward(
  torch::Tensor g_res, torch::Tensor g_pre,
  torch::Tensor hidden, torch::Tensor res_out,
  torch::Tensor post_w, torch::Tensor pre_w,
  torch::Tensor rstd_post, torch::Tensor rstd_pre)
{
  fused_add_norm_check_pair(g_res, "grad_residual_out", g_pre, "grad_pre_normed");
  fused_add_norm_check_pair(hidden, "hidden", res_out, "residual_out");
  fused_add_norm_check_pair(post_w, "post_weight", pre_w, "pre_weight");
  TORCH_CHECK(g_res.sizes() == hidden.sizes(), "cotangents and hidden must share shape");
  // The launcher dispatches on hidden.scalar_type() and then takes
  // g_res.data_ptr<x_t>(), so FP32 cotangents against BF16 hidden raised from the
  // pointer cast with a message about the cast and not about the input.
  TORCH_CHECK(g_res.scalar_type() == hidden.scalar_type(),
              "grad_residual_out must have hidden's dtype, got ", g_res.scalar_type(),
              " and ", hidden.scalar_type());
  TORCH_CHECK(g_pre.scalar_type() == hidden.scalar_type(),
              "grad_pre_normed must have hidden's dtype, got ", g_pre.scalar_type(),
              " and ", hidden.scalar_type());
  TORCH_CHECK(hidden.dim() == 2, "hidden must be 2D [T, H]");
  TORCH_CHECK(hidden.size(1) > 0, "hidden.size(1) must be non-zero");  // see the forward
  // The forward checked the weight rank and length and the backward did not, so a
  // weight shorter than the width was read past its allocation for every column:
  // the kernel indexes post_w + col and pre_w + col for all col < H. Measured with
  // a half-width weight at 4x64, the call returned without raising.
  TORCH_CHECK(post_w.dim() == 1, "post_weight must be 1D [H]");
  TORCH_CHECK(hidden.size(1) == post_w.size(0), "hidden.size(1) must equal weight size");
  fused_add_norm_check_device(hidden, "hidden", g_res, "grad_residual_out");
  fused_add_norm_check_device(hidden, "hidden", g_pre, "grad_pre_normed");
  fused_add_norm_check_device(hidden, "hidden", post_w, "post_weight");
  fused_add_norm_check_device(hidden, "hidden", pre_w, "pre_weight");
  fused_add_norm_check_rstd(rstd_post, "rstd_post", hidden.size(0), hidden);
  fused_add_norm_check_rstd(rstd_pre, "rstd_pre", hidden.size(0), hidden);

  auto f32 = hidden.options().dtype(torch::kFloat32);
  auto d_hidden = torch::empty(hidden.sizes(), f32);
  auto d_residual = torch::empty(hidden.sizes(), f32);
  auto partial_post = torch::empty(hidden.sizes(), f32);
  auto partial_pre = torch::empty(hidden.sizes(), f32);

  fused_add_norm_backward_cuda(g_res, g_pre, hidden, res_out, post_w, pre_w,
                               rstd_post, rstd_pre,
                               d_hidden, d_residual, partial_post, partial_pre);
  return {d_hidden, d_residual, partial_post, partial_pre};
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
  auto chunks = rmsnorm_backward_dw_chunks_cuda(T);

  auto partial_dw = torch::empty({chunks, H}, x.options().dtype(torch::kFloat32));
  auto dw = torch::empty({H}, x.options().dtype(torch::kFloat32));

  rmsnorm_backward_partial_dw_cuda(dy, x, rstd, mask, partial_dw);
  rmsnorm_backward_reduce_dw_cuda(partial_dw, dw);

  return dw;
}

#if !defined(USE_ROCM)
torch::Tensor reduce_rows_fp32_left_fold(torch::Tensor rows)
{
  rmsnorm_check_input(rows, "rows");
  TORCH_CHECK(rows.dim() == 2, "rows must be 2D [R, C]");
  TORCH_CHECK(rows.scalar_type() == torch::kFloat32, "rows must be float32");

  auto output = torch::empty({rows.size(1)}, rows.options());
  if (rows.size(1) != 0) {
    reduce_rows_fp32_left_fold_cuda(rows, output);
  }
  return output;
}
#endif

// SiLU / SwiGLU wrappers (WS1 elementwise activations)
torch::Tensor silu_forward(torch::Tensor x) {
  return silu_forward_cuda(x);
}

torch::Tensor silu_backward(torch::Tensor dy, torch::Tensor x) {
  return silu_backward_cuda(dy, x);
}

torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up) {
  return swiglu_forward_cuda(gate, up);
}

std::vector<torch::Tensor> swiglu_backward(
    torch::Tensor dy,
    torch::Tensor gate,
    torch::Tensor up) {
  return swiglu_backward_cuda(dy, gate, up);
}

torch::Tensor swiglu_packed_forward(torch::Tensor gate_up) {
  return swiglu_packed_forward_cuda(gate_up);
}

std::vector<torch::Tensor> swiglu_packed_backward(
    torch::Tensor dy,
    torch::Tensor gate_up) {
  return swiglu_packed_backward_cuda(dy, gate_up);
}

// Deterministic standard-softmax attention (issue #147)
std::vector<torch::Tensor> deterministic_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal,
    double scale,
    torch::optional<torch::Tensor> key_padding_mask);

std::vector<torch::Tensor> deterministic_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor P,
    bool causal,
    double scale,
    torch::optional<torch::Tensor> key_padding_mask);

#if defined(KERNEL_ALIGN_WITH_ROCM)
torch::Tensor deterministic_rope_apply_rocm(
    torch::Tensor x,
    torch::Tensor cos,
    torch::Tensor sin,
    double sin_sign);
torch::Tensor deterministic_rope_apply_token_major_rocm(
    torch::Tensor x,
    torch::Tensor positions,
    torch::Tensor cos,
    torch::Tensor sin,
    int64_t head_dim,
    double sin_sign);
#endif

#if !defined(USE_ROCM)
// Prefix-Shared Attention Declarations & Wrappers (NVIDIA PTX only).

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
#endif

// PyBind11 Module Registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RL-Kernel High-Performance Operator Extension Library";

    m.def("fused_logp", &fused_logp_forward, "Fused logp forward fallback");
#if !defined(USE_ROCM) && !defined(KERNEL_ALIGN_WITH_ROCM)
    m.def("ordered_logp_merge", &ordered_logp_merge, "Rank-ordered FP32 logp merge");
#endif

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_SM90)
    m.def("fused_logp_sm90", &fused_logp_sm90_forward, "TMA-accelerated Online Softmax Fused LogP");
    m.def("fused_linear_logp_sm90", &fused_linear_logp_sm90_forward,
          "TMA+WGMMA fused linear log-prob (hidden @ W^T -> selected-token logp), SM90; "
          "frozen reduction contract with optional temperature, logits, and real-vocab mask",
          py::arg("hidden"), py::arg("weight"), py::arg("target"),
          py::arg("bias") = py::none(), py::arg("temperature") = py::none(),
          py::arg("return_logits") = false, py::arg("real_vocab_size") = -1);
    m.def("batch_invariant_logp_sm90", &batch_invariant_logp_sm90_forward,
          "TMA online-softmax batch-invariant selected-token log-prob from logits, SM90");
    m.def("fused_linear_logp_sm90_global_target", &fused_linear_logp_sm90_global_target_forward,
          "TMA+WGMMA local-shard target-logit/lse for vocab-parallel linear log-prob, SM90",
          py::arg("hidden"), py::arg("weight"), py::arg("target"),
          py::arg("bias") = py::none(), py::arg("vocab_start_index"),
          py::arg("temperature") = py::none(), py::arg("real_vocab_size") = -1);
    m.def("fused_linear_logp_sm90_backward", &fused_linear_logp_sm90_backward,
          "CUDA fused backward for linear log-prob, SM90 backend");
    m.def("linear_logp_probs_bf16_forward", &linear_logp_probs_bf16_forward,
          "Build bf16 softmax probabilities and selected log-prob from bf16 logits");
    m.def("linear_logp_bf16_forward", &linear_logp_bf16_forward,
          "Build selected log-prob and lse from bf16 logits without saving probabilities");
    m.def("linear_logp_local_probs_bf16_forward", &linear_logp_local_probs_bf16_forward,
          "Build local bf16 softmax probabilities, target logits, and lse from bf16 logits");
    m.def("linear_logp_local_bf16_forward", &linear_logp_local_bf16_forward,
          "Build local target logits and lse from bf16 logits without saving probabilities");
    m.def("linear_logp_top_p_local_bf16_forward",
          &linear_logp_top_p_local_bf16_forward,
          "Build local target logits and lse from a compact top-p replay set");
    m.def("linear_logp_probs_bf16_to_dlogits_", &linear_logp_probs_bf16_to_dlogits_,
          "In-place bf16 probs -> dlogits for selected log-prob backward");
    m.def("linear_logp_local_probs_bf16_to_dlogits_",
          &linear_logp_local_probs_bf16_to_dlogits_,
          "In-place local bf16 probs -> TP dlogits for selected log-prob backward");
    m.def("linear_logp_logits_bf16_to_dlogits", &linear_logp_logits_bf16_to_dlogits,
          "Build bf16 dlogits from bf16 logits and fp32 lse");
    // RoPE rotate-half apply, SM90 (forward and backward share the kernel via sin_sign)
    m.def("rope_apply_sm90", &rope_apply_sm90, "RoPE rotate-half apply (GPT-NeoX), SM90");
    m.def("embedding_sm90_forward", &embedding_sm90_forward,
          "Single-card SM90 batch-invariant embedding forward");
    m.def("embedding_sm90_forward_fp32", &embedding_sm90_forward_fp32,
          "Single-card SM90 batch-invariant embedding forward with fp32 output");
    m.def("lm_head_sm90_forward", &lm_head_sm90_forward,
          "Single-card SM90 batch-invariant LM-head forward");
    m.def("lm_head_sm90_forward_fp32", &lm_head_sm90_forward_fp32,
          "Single-card SM90 batch-invariant LM-head forward with fp32 output");
#endif

#if defined(__CUDACC__) || defined(KERNEL_ALIGN_WITH_CUDA) || defined(KERNEL_ALIGN_WITH_ROCM)
    m.def("fused_logp_forward_out", &fused_logp_forward_out, "Fused logp out");
    m.def("fused_logp_forward_fp32", &fused_logp_forward_fp32, "Fused logp fp32");
    m.def("fused_logp_forward_indexed_out", &fused_logp_forward_indexed_out, "Fused logp indexed out");
    m.def("fused_logp_forward_indexed_fp32", &fused_logp_forward_indexed_fp32, "Fused logp indexed fp32");
    m.def("fused_logp_forward_online_out", &fused_logp_forward_online_out, "Fused logp online out");
    m.def("fused_logp_forward_online_fp32", &fused_logp_forward_online_fp32, "Fused logp online fp32");
    m.def("fused_logp_forward_online_indexed_out", &fused_logp_forward_online_indexed_out, "Fused logp online indexed out");
    m.def("fused_logp_forward_online_indexed_fp32", &fused_logp_forward_online_indexed_fp32, "Fused logp online indexed fp32");
    m.def("deterministic_logp", &deterministic_logp_forward, "Batch-invariant deterministic logp");
    m.def("deterministic_logp_forward_out", &deterministic_logp_forward_out, "Batch-invariant deterministic logp out");
    m.def("deterministic_logp_forward_fp32", &deterministic_logp_forward_fp32, "Batch-invariant deterministic logp fp32");
    m.def("deterministic_logp_forward_indexed_out", &deterministic_logp_forward_indexed_out, "Batch-invariant deterministic logp indexed out");
    m.def("deterministic_logp_forward_indexed_fp32", &deterministic_logp_forward_indexed_fp32, "Batch-invariant deterministic logp indexed fp32");
    m.def("deterministic_logp_tile_stats", &deterministic_logp_tile_stats,
          "Deterministic local vocab-tile FP32 max and sumexp partials");
    m.def("deterministic_logp_backward", &deterministic_logp_backward,
          "Fused vocab-parallel selected-logprob/LSE backward on the local shard");
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
    m.def("hip_deterministic_logp_tile_stats", &hip_deterministic_logp_tile_stats,
          "ROCm-tuned vocab-tile FP32 max and sumexp partials read from the stored shard");
    m.def("hip_deterministic_logp_backward", &hip_deterministic_logp_backward,
          "ROCm fused vocab-parallel selected-logprob/LSE backward on the local shard");
    m.def("hip_sparse_nucleus_logp_forward", &hip_sparse_nucleus_logp_forward,
          "ROCm fixed-binary 128-candidate sparse nucleus logprob forward");
    m.def("hip_replicated_sparse_nucleus_logp", &hip_replicated_sparse_nucleus_logp,
          "ROCm sparse rollout scoring from already replicated raw logits");
    m.def("hip_sparse_nucleus_logp_backward", &hip_sparse_nucleus_logp_backward,
          "ROCm fixed-binary 128-candidate sparse nucleus logprob backward");
#endif

#if !defined(USE_ROCM) && !defined(KERNEL_ALIGN_WITH_ROCM)
    // Single-node TP=8 fixed-tree CUDA IPC collectives. ROCm dispatches to
    // the Python RCCL transport implementation instead.
    m.def(
        "deterministic_collective_ipc_meta",
        &deterministic_collective_ipc_meta,
        "Export a CUDA allocation for deterministic collectives");
    m.def(
        "deterministic_collective_create",
        &deterministic_collective_create,
        "Create a single-node TP=8 deterministic collective state");
    m.def(
        "deterministic_collective_destroy",
        &deterministic_collective_destroy,
        "Destroy a deterministic collective state");
    m.def(
        "deterministic_collective_stage",
        &deterministic_collective_stage,
        "Stage one rank's input in symmetric CUDA IPC memory");
    m.def(
        "deterministic_collective_all_reduce",
        &deterministic_collective_all_reduce,
        "Run the TP=8 deterministic fixed-tree all-reduce kernel");
    m.def(
        "deterministic_collective_prepare_staged",
        &deterministic_collective_prepare_staged,
        "Reserve the local IPC payload for direct GEMM output");
    m.def(
        "deterministic_collective_all_reduce_staged",
        &deterministic_collective_all_reduce_staged,
        "Reduce a GEMM result already resident in the local IPC payload");
    m.def(
        "deterministic_collective_all_reduce_fused",
        &deterministic_collective_all_reduce_fused,
        "Run a fused small-message deterministic fixed-tree all-reduce");
    m.def(
        "deterministic_collective_reduce_scatter",
        &deterministic_collective_reduce_scatter,
        "Run the TP=8 deterministic fixed-tree reduce-scatter kernel");
    m.def(
        "deterministic_collective_all_gather",
        &deterministic_collective_all_gather,
        "Run the TP=8 deterministic rank-ordered all-gather kernel");
    m.def(
        "deterministic_collective_all_gather_fused",
        &deterministic_collective_all_gather_fused,
        "Run a fused small-message deterministic rank-ordered all-gather");
    m.def(
        "deterministic_collective_all_gather_many",
        &deterministic_collective_all_gather_many,
        "Gather multiple tensors with one deterministic staging handshake");
#endif

#if defined(KERNEL_ALIGN_WITH_ROCM)
    m.def(
        "deterministic_collective_rocm_all_reduce",
        &deterministic_collective_rocm_all_reduce,
        "Run the ROCm fixed-tree all-reduce kernel");
    m.def(
        "deterministic_collective_rocm_reduce_scatter",
        &deterministic_collective_rocm_reduce_scatter,
        "Run the ROCm fixed-tree reduce-scatter kernel");
    m.def("deterministic_collective_rocm_ipc_meta",
          &deterministic_collective_rocm_ipc_meta,
          "Export a ROCm allocation for IPC deterministic collectives");
    m.def("deterministic_collective_rocm_ipc_allocate",
          &deterministic_collective_rocm_ipc_allocate,
          "Allocate ROCm memory that supports IPC export");
    m.def("deterministic_collective_rocm_ipc_create",
          &deterministic_collective_rocm_ipc_create,
          "Create a ROCm IPC deterministic collective state");
    m.def("deterministic_collective_rocm_ipc_destroy",
          &deterministic_collective_rocm_ipc_destroy,
          "Destroy a ROCm IPC deterministic collective state");
    m.def("deterministic_collective_rocm_ipc_synchronize",
          &deterministic_collective_rocm_ipc_synchronize,
          "Wait until every rank finishes reading ROCm IPC staging");
    m.def("deterministic_collective_rocm_ipc_stage",
          &deterministic_collective_rocm_ipc_stage,
          "Stage an input for ROCm IPC deterministic collectives");
    m.def("deterministic_collective_rocm_ipc_prepare_staged",
          &deterministic_collective_rocm_ipc_prepare_staged,
          "Reserve the direct-output ROCm IPC payload");
    m.def("deterministic_collective_rocm_ipc_all_reduce_staged",
          &deterministic_collective_rocm_ipc_all_reduce_staged,
          "Reduce a result already resident in the ROCm IPC payload");
    m.def("deterministic_collective_rocm_ipc_all_reduce",
          &deterministic_collective_rocm_ipc_all_reduce,
          "Run a direct ROCm IPC fixed-tree all-reduce");
    m.def("deterministic_collective_rocm_ipc_all_reduce_input",
          &deterministic_collective_rocm_ipc_all_reduce_input,
          "Stage and run a direct ROCm IPC fixed-tree all-reduce");
    m.def("deterministic_collective_rocm_ipc_reduce_scatter",
          &deterministic_collective_rocm_ipc_reduce_scatter,
          "Run a direct ROCm IPC fixed-tree reduce-scatter");
    m.def("deterministic_collective_rocm_ipc_reduce_scatter_input",
          &deterministic_collective_rocm_ipc_reduce_scatter_input,
          "Stage and run a direct ROCm IPC fixed-tree reduce-scatter");
    m.def("deterministic_collective_rocm_ipc_reduce_scatter_many",
          &deterministic_collective_rocm_ipc_reduce_scatter_many,
          "Run multiple ROCm IPC fixed-tree reduce-scatters with one synchronization");
    m.def("deterministic_collective_rocm_ipc_all_gather",
          &deterministic_collective_rocm_ipc_all_gather,
          "Run a direct ROCm IPC rank-ordered all-gather");
    m.def("deterministic_collective_rocm_ipc_all_gather_input",
          &deterministic_collective_rocm_ipc_all_gather_input,
          "Stage and run a direct ROCm IPC rank-ordered all-gather");
#endif

#if !defined(USE_ROCM)
    // Prefix-shared attention uses NVIDIA PTX; the declaration above carries the
    // same guard, so the registration must repeat it or a ROCm build fails on an
    // undeclared identifier.
    m.def("prefix_shared_attention", &prefix_shared_attention, "Prefix-Shared Fused Attention for GRPO");
#endif

    // registry Batch-Invariant Deterministic GEMM
    m.def("det_gemm_sm90_compiled", &det_gemm_sm90_compiled,
          "Whether the extension contains the SM90 deterministic GEMM implementation");
    m.def("det_gemm_fwd", &det_gemm_fwd, "Batch-invariant deterministic GEMM forward (C=A@B)");
    m.def(
        "det_gemm_fwd_rhs_transposed",
        &det_gemm_fwd_rhs_transposed,
        "Batch-invariant deterministic GEMM with physical Bt[N,K] (C=A@Bt^T)");
    m.def("det_gemm_da", &det_gemm_da, "Batch-invariant deterministic GEMM backward dA (dC@B^T)");
    m.def("det_gemm_db", &det_gemm_db, "Batch-invariant deterministic GEMM backward dB (A^T@dC)");
    m.def(
        "det_gemm_db_transposed",
        &det_gemm_db_transposed,
        "Batch-invariant deterministic GEMM backward in canonical [N,K] layout");
    // registry RMSNorm
    m.def("rmsnorm_forward", &rmsnorm_forward, "Batch-invariant RMSNorm forward CUDA");
    m.def("rmsnorm_backward_dx", &rmsnorm_backward_dx, "Batch-invariant RMSNorm backward dx CUDA");
    m.def("rmsnorm_backward_dw", &rmsnorm_backward_dw, "Deterministic RMSNorm backward dweight CUDA");
    m.def("fused_add_norm_forward", &fused_add_norm_forward,
          "Muse-Glimmer post-norm + residual add + pre-norm forward CUDA");
    m.def("fused_add_norm_backward", &fused_add_norm_backward,
          "Muse-Glimmer post-norm + residual add + pre-norm backward CUDA");
#if !defined(USE_ROCM)
    m.def(
        "reduce_rows_fp32_left_fold",
        &reduce_rows_fp32_left_fold,
        "Ascending-row FP32 left-fold reduction CUDA");
#endif

    // registry SiLU / SwiGLU (elementwise activation)
    m.def("silu_forward", &silu_forward, "Batch-invariant SiLU forward CUDA");
    m.def("silu_backward", &silu_backward, "Batch-invariant SiLU backward CUDA");
    m.def("swiglu_forward", &swiglu_forward, "Batch-invariant SwiGLU forward CUDA");
    m.def("swiglu_backward", &swiglu_backward, "Batch-invariant SwiGLU backward CUDA");
    m.def("swiglu_packed_forward", &swiglu_packed_forward,
          "Batch-invariant SwiGLU forward for [rows, 2 * intermediate]");
    m.def("swiglu_packed_backward", &swiglu_packed_backward,
          "Batch-invariant SwiGLU backward for [rows, 2 * intermediate]");

    // Deterministic standard-softmax attention (issue #147)
    m.def(
        "deterministic_attention_forward",
        &deterministic_attention_forward,
        "Deterministic standard softmax attention forward (out, lse)");
    m.def(
        "deterministic_attention_backward",
        &deterministic_attention_backward,
        "Deterministic standard softmax attention backward (dQ, dK, dV)");
#if defined(KERNEL_ALIGN_WITH_ROCM)
    m.def(
        "deterministic_rope_apply_rocm",
        &deterministic_rope_apply_rocm,
        "Deterministic GPT-NeoX RoPE apply for ROCm");
    m.def(
        "deterministic_rope_apply_token_major_rocm",
        &deterministic_rope_apply_token_major_rocm,
        "Deterministic GPT-NeoX token-major RoPE apply for ROCm");
#endif
#endif
}
