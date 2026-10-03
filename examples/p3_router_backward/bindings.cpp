// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors

#include <torch/extension.h>

void route_backward_core_out(
    const torch::Tensor& dweights,
    const torch::Tensor& ids,
    const torch::Tensor& p,
    const torch::Tensor& z,
    const torch::Tensor& row_active,
    torch::Tensor out,
    int64_t threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("route_backward_core_out", &route_backward_core_out,
        "Internal T06 arithmetic experiment; not the sealed P3 operator ABI");
}
