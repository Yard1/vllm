#include "gencode/sample_triton.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void sample(torch::Tensor &sample_indices,         // [num_samples]
            torch::Tensor &output_samples,         // [num_samples, n_best]
            torch::Tensor &output_logprobs,        // [num_samples, n_best]
            torch::Tensor &output_modified_probs,  // [batch_size, vocab_size] 
            torch::Tensor &probs,                  // [batch_size, vocab_size]
            torch::Tensor &logprobs,               // [batch_size, vocab_size]
            torch::Tensor &seeds,                  // [num_samples]
            torch::Tensor &uniform_noise,          // [num_samples, n_best, vocab_size]
            bool modify_greedy_probs, bool save_logprobs, bool save_modified_probs) {
  int64_t n_samples = sample_indices.size(0);
  int64_t n_cols = probs.size(1);
  int64_t n_best = output_samples.dim() > 1 ? output_samples.size(1) : 1;

  cudaSetDevice(probs.get_device());
  CUstream stream = at::cuda::getCurrentCUDAStream();

  sample_triton(stream,
                reinterpret_cast<CUdeviceptr>(sample_indices.data_ptr()),
                reinterpret_cast<CUdeviceptr>(output_samples.data_ptr()),
                reinterpret_cast<CUdeviceptr>(output_logprobs.data_ptr()),
                reinterpret_cast<CUdeviceptr>(output_modified_probs.data_ptr()),
                reinterpret_cast<CUdeviceptr>(probs.data_ptr()),
                reinterpret_cast<CUdeviceptr>(logprobs.data_ptr()),
                reinterpret_cast<CUdeviceptr>(seeds.data_ptr()),
                reinterpret_cast<CUdeviceptr>(uniform_noise.data_ptr()),
                output_samples.stride(0), probs.stride(0), uniform_noise.stride(0), n_best > 1 ? uniform_noise.stride(1) : 1,
                n_samples, n_cols, n_best, modify_greedy_probs, save_logprobs, save_modified_probs);
}