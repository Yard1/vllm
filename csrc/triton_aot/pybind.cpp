#include "ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Triton AOT ops
  pybind11::module triton_ops = m.def_submodule("triton_ops", "Triton AOT ops");

  triton_ops.def("sample", &sample, "Perform sampling.");
  triton_ops.def("apply_penalties", &apply_penalties, "Apply penalties.");
  triton_ops.def("seeded_uniform", &seeded_uniform, "Generate random numbers from uniform distribution following seeds.");
}
