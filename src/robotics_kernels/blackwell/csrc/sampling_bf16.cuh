// Shared BF16 logits processing and argmax; each extension keeps private
// symbols.
#pragma once

#include <climits>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ bool argmax_better(float candidate_score,
                                              int candidate_index,
                                              float current_score,
                                              int current_index) {
  return candidate_score > current_score ||
         (candidate_score == current_score && candidate_index < current_index);
}

__global__ void
process_logits_bf16_kernel(const __nv_bfloat16 *__restrict__ logits,
                           const uint8_t *__restrict__ seen_tokens,
                           const uint8_t *__restrict__ eos_tokens,
                           const int32_t *__restrict__ generated_count,
                           float *__restrict__ processed_logits, int vocab_size,
                           float repetition_penalty, int min_new_tokens) {
  constexpr float kNegativeInfinity = -__FLT_MAX__;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vocab_size) {
    return;
  }
  float score = __bfloat162float(logits[index]);
  if (seen_tokens[index] != 0 && repetition_penalty != 1.0f) {
    score =
        score < 0.0f ? score * repetition_penalty : score / repetition_penalty;
  }
  if (generated_count[0] < min_new_tokens && eos_tokens[index] != 0) {
    score = kNegativeInfinity;
  }
  processed_logits[index] = score;
}

__global__ void processed_argmax_stage1_bf16_kernel(
    const __nv_bfloat16 *__restrict__ logits,
    const uint8_t *__restrict__ seen_tokens,
    const uint8_t *__restrict__ eos_tokens,
    const int32_t *__restrict__ generated_count,
    float *__restrict__ partial_scores, int32_t *__restrict__ partial_indices,
    int vocab_size, float repetition_penalty, int min_new_tokens) {
  constexpr float kNegativeInfinity = -__FLT_MAX__;
  float best_score = kNegativeInfinity;
  int best_index = INT_MAX;
  int generated = generated_count[0];
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < vocab_size;
       index += gridDim.x * blockDim.x) {
    float score = __bfloat162float(logits[index]);
    if (seen_tokens[index] != 0 && repetition_penalty != 1.0f) {
      score = score < 0.0f ? score * repetition_penalty
                           : score / repetition_penalty;
    }
    if (generated < min_new_tokens && eos_tokens[index] != 0) {
      score = kNegativeInfinity;
    }
    if (argmax_better(score, index, best_score, best_index)) {
      best_score = score;
      best_index = index;
    }
  }

  constexpr unsigned kMask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    float other_score = __shfl_down_sync(kMask, best_score, offset);
    int other_index = __shfl_down_sync(kMask, best_index, offset);
    if (argmax_better(other_score, other_index, best_score, best_index)) {
      best_score = other_score;
      best_index = other_index;
    }
  }
  __shared__ float warp_scores[32];
  __shared__ int warp_indices[32];
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;
  if (lane == 0) {
    warp_scores[warp] = best_score;
    warp_indices[warp] = best_index;
  }
  __syncthreads();
  if (warp == 0) {
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    best_score = lane < warp_count ? warp_scores[lane] : kNegativeInfinity;
    best_index = lane < warp_count ? warp_indices[lane] : INT_MAX;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      float other_score = __shfl_down_sync(kMask, best_score, offset);
      int other_index = __shfl_down_sync(kMask, best_index, offset);
      if (argmax_better(other_score, other_index, best_score, best_index)) {
        best_score = other_score;
        best_index = other_index;
      }
    }
    if (lane == 0) {
      partial_scores[blockIdx.x] = best_score;
      partial_indices[blockIdx.x] = best_index;
    }
  }
}

__global__ void processed_argmax_finalize_kernel(
    const float *__restrict__ partial_scores,
    const int32_t *__restrict__ partial_indices,
    uint8_t *__restrict__ seen_tokens, int32_t *__restrict__ generated_count,
    int64_t *__restrict__ output_token, int partial_count) {
  constexpr float kNegativeInfinity = -__FLT_MAX__;
  float best_score = kNegativeInfinity;
  int best_index = INT_MAX;
  for (int index = threadIdx.x; index < partial_count; index += blockDim.x) {
    float score = partial_scores[index];
    int token = partial_indices[index];
    if (argmax_better(score, token, best_score, best_index)) {
      best_score = score;
      best_index = token;
    }
  }
  constexpr unsigned kMask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    float other_score = __shfl_down_sync(kMask, best_score, offset);
    int other_index = __shfl_down_sync(kMask, best_index, offset);
    if (argmax_better(other_score, other_index, best_score, best_index)) {
      best_score = other_score;
      best_index = other_index;
    }
  }
  __shared__ float warp_scores[32];
  __shared__ int warp_indices[32];
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;
  if (lane == 0) {
    warp_scores[warp] = best_score;
    warp_indices[warp] = best_index;
  }
  __syncthreads();
  if (warp == 0) {
    int warp_count = (blockDim.x + warpSize - 1) / warpSize;
    best_score = lane < warp_count ? warp_scores[lane] : kNegativeInfinity;
    best_index = lane < warp_count ? warp_indices[lane] : INT_MAX;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      float other_score = __shfl_down_sync(kMask, best_score, offset);
      int other_index = __shfl_down_sync(kMask, best_index, offset);
      if (argmax_better(other_score, other_index, best_score, best_index)) {
        best_score = other_score;
        best_index = other_index;
      }
    }
    if (lane == 0) {
      output_token[0] = static_cast<int64_t>(best_index);
      seen_tokens[best_index] = 1;
      generated_count[0] += 1;
    }
  }
}

} // namespace
