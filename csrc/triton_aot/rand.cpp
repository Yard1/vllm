#include "gencode/seeded_uniform_fp32.h"
#include "gencode/seeded_uniform_bf16.h"
#include "gencode/seeded_uniform_fp16.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void seeded_uniform(torch::Tensor &out,   // [n] or [n, m] or [n, m, k]
                    torch::Tensor &seeds  // [n]
                    ) {
  cudaSetDevice(out.get_device());
  CUstream stream = at::cuda::getCurrentCUDAStream();

  int64_t n_rows, n_3d, stride_row, stride_3d;
  int32_t n_cols;
  auto n_dims = out.dim();

  if (n_dims == 3) {
      n_rows = out.size(0);
      n_3d = out.size(1);
      n_cols = out.size(2);
      stride_row = out.stride(0);
      stride_3d = out.stride(1);
  }
  else if (n_dims == 2) {
      n_rows = out.size(0);
      n_cols = out.size(1);
      n_3d = 1;
      stride_row = out.stride(0);
      stride_3d = 1;
  }
  else {
      n_rows = 1;
      n_cols = out.size(0);
      n_3d = 1;
      stride_row = 1;
      stride_3d = 1;
  }

  switch (out.scalar_type()) {
  case at::ScalarType::Float:
    seeded_uniform_fp32(
        stream, reinterpret_cast<CUdeviceptr>(out.data_ptr()),
        reinterpret_cast<CUdeviceptr>(seeds.data_ptr()),
        stride_row, stride_3d, seeds.stride(0), n_rows, n_3d, n_cols);
    break;
  case at::ScalarType::Half:
    seeded_uniform_fp16(
        stream, reinterpret_cast<CUdeviceptr>(out.data_ptr()),
        reinterpret_cast<CUdeviceptr>(seeds.data_ptr()),
        stride_row, stride_3d, seeds.stride(0), n_rows, n_3d, n_cols);
    break;
  case at::ScalarType::BFloat16:
    seeded_uniform_bf16(
        stream, reinterpret_cast<CUdeviceptr>(out.data_ptr()),
        reinterpret_cast<CUdeviceptr>(seeds.data_ptr()),
        stride_row, stride_3d, seeds.stride(0), n_rows, n_3d, n_cols);
    break;
  default:
    TORCH_CHECK(false, "Unsupported scalar type for seeded_uniform: ",
                out.scalar_type());
    break;
  }
}