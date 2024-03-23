#include <torch/extension.h>

void sample(torch::Tensor &sample_indices,         // [num_samples]
            torch::Tensor &output_samples,         // [num_samples, n_best]
            torch::Tensor &output_logprobs,        // [num_samples, n_best]
            torch::Tensor &output_modified_probs,  // [num_samples, n_best]
            torch::Tensor &probs,                  // [batch_size, vocab_size]
            torch::Tensor &logprobs,               // [batch_size, vocab_size]
            torch::Tensor &seeds,                  // [num_samples]
            torch::Tensor &uniform_noise,          // [num_samples, n_best, vocab_size]
            bool modify_greedy_probs, bool save_logprobs, bool save_modified_probs);

void apply_penalties(torch::Tensor &logits,             // [batch_size, vocab_size]
                     torch::Tensor &presence_penalty,   // [batch_size]
                     torch::Tensor &freqency_penalty,   // [batch_size]
                     torch::Tensor &repetition_penalty, // [batch_size]
                     torch::Tensor &p_token_ids,        // [num_unique_tokens]
                     torch::Tensor &p_token_counts,     // [num_unique_tokens]
                     torch::Tensor &p_cumsum_seq_len,   // [batch_size]
                     int64_t p_max_len_in_batch);

void seeded_uniform(torch::Tensor &out,   // [n] or [n, m] or [n, m, k]
                    torch::Tensor &seeds  // [n]
                   );