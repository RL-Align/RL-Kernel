// SPDX-License-Identifier: Apache-2.0
// Python module entry point for the optional RoPE-only wheel. Operators are
// registered by multi_axis_rope.cu through the PyTorch dispatcher.
#include <pybind11/pybind11.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Qwen-Image RoPE dispatcher library";
}
