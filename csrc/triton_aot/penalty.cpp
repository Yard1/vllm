
#include "gencode/apply_penalty_bf16.h"
#include "gencode/apply_penalty_fp16.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void apply_penalties(torch::Tensor &logits, // [batch_size, vocab_size]
                     torch::Tensor &presence_penalty,   // [batch_size]
                     torch::Tensor &freqency_penalty,   // [batch_size]
                     torch::Tensor &repetition_penalty, // [batch_size]
                     torch::Tensor &p_token_ids,        // [num_unique_tokens]
                     torch::Tensor &p_token_counts,     // [num_unique_tokens]
                     torch::Tensor &p_cumsum_seq_len,   // [batch_size]
                     int64_t p_max_len_in_batch) {
  cudaSetDevice(logits.get_device());
  CUstream stream = at::cuda::getCurrentCUDAStream();

  switch (logits.scalar_type()) {
  case at::ScalarType::Half:
    apply_penalty_fp16(
        stream, reinterpret_cast<CUdeviceptr>(logits.data_ptr()),
        reinterpret_cast<CUdeviceptr>(presence_penalty.data_ptr()),
        reinterpret_cast<CUdeviceptr>(freqency_penalty.data_ptr()),
        reinterpret_cast<CUdeviceptr>(repetition_penalty.data_ptr()),
        reinterpret_cast<CUdeviceptr>(p_token_ids.data_ptr()),
        reinterpret_cast<CUdeviceptr>(p_token_counts.data_ptr()),
        reinterpret_cast<CUdeviceptr>(p_cumsum_seq_len.data_ptr()),
        logits.stride(0), logits.size(0), p_max_len_in_batch);
    break;
  case at::ScalarType::BFloat16:
    apply_penalty_bf16(
        stream, reinterpret_cast<CUdeviceptr>(logits.data_ptr()),
        reinterpret_cast<CUdeviceptr>(presence_penalty.data_ptr()),
        reinterpret_cast<CUdeviceptr>(freqency_penalty.data_ptr()),
        reinterpret_cast<CUdeviceptr>(repetition_penalty.data_ptr()),
        reinterpret_cast<CUdeviceptr>(p_token_ids.data_ptr()),
        reinterpret_cast<CUdeviceptr>(p_token_counts.data_ptr()),
        reinterpret_cast<CUdeviceptr>(p_cumsum_seq_len.data_ptr()),
        logits.stride(0), logits.size(0), p_max_len_in_batch);
    break;
  default:
    TORCH_CHECK(false, "Unsupported scalar type for apply_penalties: ",
                logits.scalar_type());
    break;
  }
}