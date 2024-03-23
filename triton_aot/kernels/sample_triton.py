import torch
import triton
import triton.language as tl

_EPS = 1e-6


def sample(probs: torch.Tensor,
           logprobs: torch.Tensor,
           sample_indices: torch.Tensor,
           output_samples: torch.Tensor,
           output_logprobs: torch.Tensor,
           output_modified_probs: torch.Tensor,
           seeds: torch.Tensor,
           uniform_noise: torch.Tensor,
           *,
           modify_greedy_probs: bool = False,
           save_logprobs: bool = True,
           save_modified_probs: bool = False) -> torch.Tensor:
    """Sample tokens from probs.

    Args:
        probs [batch_size, vocab_size]: probs to sample from.
        logprobs [batch_size, vocab_size]: logprobs (used when
            save_logprobsis True).
        sample_indices [n]: Indices of the samples to use for each row of probs.
        output_samples [n, n_best]: Output tensor to store samples in.
        output_logprobs [n, n_best]: Output tensor to store logprobs in.
        output_modified_probs [n, n_best]: Output tensor to store
            probs of chosen tokens in (modified with noise).
        seeds [n]: Seeds to use for sampling. If the seed is 0, we use
            greedy sampling. Note this is ONLY used for determining
            whether to use random sampling or not. The actual random
            noise should be passed as uniform_noise.
        uniform_noise [batch_size, n_best, vocab_size]: Uniform
            noise to use for random sampling (will be converted
            to exponential gumbel noise by the kernel).
        modify_greedy_probs: If True, we modify the probs tensor in-place
            to encode the sampling method used for each row. This is used
            in speculative decoding. Only applies in greedy decoding.
        save_logprobs: If True, we save the logprobs of the sampled tokens
            in the output_logprobs tensor.
        save_modified_probs: If True, we save the modified probs (with noise)
            of the sampled tokens in the output_modified_probs tensor.
            DOES NOT include the modification done by modify_greedy_probs
            (because we want to use the unmodified probs to pick the best
            split in case of multi-split sampling).
    """
    n_samples = sample_indices.shape[0]
    n_cols = probs.shape[1]
    n_best = output_samples.shape[1] if len(output_samples.shape) > 1 else 1

    # The block size is the smallest power of two greater than the number of
    # columns in probs
    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4
    # Manual tuning. This seems to give best performance on A100 for
    # simple kernels like this.
    if block_size >= 8192:
        num_warps = 32
    elif block_size >= 4096:
        num_warps = 16
    elif block_size >= 2048:
        num_warps = 8

    # Enqueue kernel. The 1D launch grid is simple: we have one kernel
    # instance per row of the probs matrix
    _sample_triton[(n_samples, n_best)](
        sample_indices,
        output_samples,
        output_logprobs,
        output_modified_probs,
        probs,
        logprobs,
        seeds,
        uniform_noise,
        output_samples.stride(0),
        probs.stride(0),
        uniform_noise.stride(0),
        uniform_noise.stride(1) if n_best > 1 else 1,
        n_samples,
        n_cols,
        n_best,
        num_warps=num_warps,
        block_size=block_size,
        modify_greedy_probs=modify_greedy_probs,
        save_logprobs=save_logprobs,
        save_modified_probs=save_modified_probs,
    )
    return output_samples, output_logprobs, output_modified_probs


@triton.jit
def _sample_triton(
        sample_indices_ptr: torch.Tensor, output_ptr: torch.Tensor,
        output_logprobs_ptr: torch.Tensor,
        output_modified_probs_ptr: torch.Tensor, probs_ptr: torch.Tensor,
        logprobs_ptr: torch.Tensor, seeds_ptr: torch.Tensor,
        uniform_noise_ptr: torch.Tensor, output_row_stride: int,
        probs_row_stride: int, uniform_noise_row_stride: int,
        uniform_noise_best_stride: int, n_samples: int, n_cols: int,
        n_best: int, block_size: tl.constexpr,
        modify_greedy_probs: tl.constexpr, save_logprobs: tl.constexpr,
        save_modified_probs: tl.constexpr):
    # The rows are independent, so we parallelize across those
    sample_idx = tl.program_id(0)
    best_idx = tl.program_id(1)

    # Load the row index from DRAM
    row_idx = tl.load(sample_indices_ptr + sample_idx)
    seed = tl.load(seeds_ptr + sample_idx)
    uses_random_sampling = seed != 0

    # The stride represents how much we need to increase the
    # pointer to advance 1 row
    row_start_ptr = probs_ptr + row_idx * probs_row_stride

    # The block size is the next power of two greater than n_cols,
    # so we can fit each row in a single block
    col_offsets = tl.arange(0, block_size)

    # Load the row into SRAM, using a mask since block_size may be > than n_cols
    row = tl.load(row_start_ptr + col_offsets,
                  mask=col_offsets < n_cols,
                  other=float("-inf"))

    if uses_random_sampling:
        uniform_noise_start_ptr = uniform_noise_ptr + sample_idx * uniform_noise_row_stride + best_idx * uniform_noise_best_stride
        uniform_noise = tl.load(uniform_noise_start_ptr + col_offsets,
                                mask=col_offsets < n_cols,
                                other=0.5)

        # NEEDS TO BE MANUALLY KEPT IN SYNC WITH vllm/tests/ops/test_sampler.py
        # tl.rand returns values in [0, 1), so we clamp lower bound
        # to _EPS to avoid log(0) and thus division by nan later
        lb = tl.full(uniform_noise.shape, _EPS, uniform_noise.dtype)
        uniform_noise = tl.maximum(uniform_noise, lb)
        # Use the inversion method to turn uniform samples
        # into exponential samples
        exponential_noise = -tl.log(uniform_noise)

        row /= exponential_noise

    sampled_value, sampled_token = tl.max(row, axis=0, return_indices=True)
    # clamp sampled token to n_cols - 1
    # this should not be necessary, but we do it
    # just in case
    if sampled_token >= n_cols:
        sampled_token = n_cols - 1
    # Write back output to DRAM
    output_row_start_ptr = (output_ptr + sample_idx * output_row_stride +
                            best_idx)
    tl.store(output_row_start_ptr, sampled_token)

    if modify_greedy_probs:
        if not uses_random_sampling:
            # Set the probability of the sampled token to 1, all other
            # tokens to zero. This is used in speculative decoding where
            # the sampling method must be encoded within the sampled
            # probability distributions.
            row = tl.where(col_offsets == sampled_token, 1.0, 0.0)
            tl.store(row_start_ptr + col_offsets,
                     row,
                     mask=col_offsets < n_cols)

    if save_modified_probs:
        output_row_start_ptr = (output_modified_probs_ptr +
                                sample_idx * output_row_stride + best_idx)
        tl.store(output_row_start_ptr, sampled_value)

    if save_logprobs:
        # Load the row into SRAM, using a mask since block_size
        # may be > than n_cols
        sampled_logprob = tl.load(logprobs_ptr + row_idx * probs_row_stride +
                                  sampled_token)
        # Write back output to DRAM
        output_row_start_ptr = (output_logprobs_ptr +
                                sample_idx * output_row_stride + best_idx)
        tl.store(output_row_start_ptr, sampled_logprob)
