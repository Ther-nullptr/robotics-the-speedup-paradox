#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <torch/library.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/threadblock/epilogue_with_scaling_factor.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/device/gemv_blockscaled.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/gemv_blockscaled.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/packed_stride.hpp"

#include "sampling_bf16.cuh"

namespace {

constexpr int kSfVecSize = 16;
constexpr int kElementsPerAccess = 32;
constexpr int64_t kDecodePolicyBalanced = 0;
constexpr int64_t kDecodePolicyLowClock = 1;
constexpr int64_t kDecodePolicyMidClock = 2;
constexpr int64_t kDecodePolicyLegacy = 3;
std::atomic<int64_t> g_decode_policy{kDecodePolicyBalanced};

int64_t ceil_div_int64(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

int64_t round_up_int64(int64_t value, int64_t divisor) {
  return ceil_div_int64(value, divisor) * divisor;
}

int64_t sf_blocks_k(int64_t k) {
  return ceil_div_int64(ceil_div_int64(k, kSfVecSize), 4);
}

int64_t sf_blocks_m(int64_t m) { return ceil_div_int64(m, 128); }

int64_t sfd_elems(int64_t rows) {
  return ceil_div_int64(ceil_div_int64(rows, kSfVecSize), 4) * 512;
}

__device__ __forceinline__ __nv_bfloat16 round_bf16(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float as_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ int64_t offset_4d(int64_t batch, int64_t head,
                                             int64_t sequence, int64_t column,
                                             int64_t stride_batch,
                                             int64_t stride_head,
                                             int64_t stride_sequence,
                                             int64_t stride_column) {
  return batch * stride_batch + head * stride_head +
         sequence * stride_sequence + column * stride_column;
}

__device__ __forceinline__ void decode_bhsd(int64_t index, int64_t heads,
                                            int64_t sequence_length,
                                            int64_t head_dim, int64_t &batch,
                                            int64_t &head, int64_t &sequence,
                                            int64_t &column) {
  column = index % head_dim;
  index /= head_dim;
  sequence = index % sequence_length;
  index /= sequence_length;
  head = index % heads;
  batch = index / heads;
}

__device__ __forceinline__ __nv_bfloat16
rope_element(const __nv_bfloat16 *input, const __nv_bfloat16 *cos,
             const __nv_bfloat16 *sin, int64_t batch, int64_t head,
             int64_t sequence, int64_t column, int64_t head_dim,
             int64_t input_stride_batch, int64_t input_stride_head,
             int64_t input_stride_sequence, int64_t input_stride_column,
             int64_t cos_batch, int64_t cos_sequence, int64_t cos_stride_batch,
             int64_t cos_stride_sequence, int64_t cos_stride_column) {
  int64_t input_offset =
      offset_4d(batch, head, sequence, column, input_stride_batch,
                input_stride_head, input_stride_sequence, input_stride_column);
  int64_t half_dim = head_dim / 2;
  int64_t rotated_column =
      column < half_dim ? column + half_dim : column - half_dim;
  int64_t rotated_offset =
      offset_4d(batch, head, sequence, rotated_column, input_stride_batch,
                input_stride_head, input_stride_sequence, input_stride_column);
  int64_t embedding_batch = cos_batch == 1 ? 0 : batch;
  int64_t embedding_offset = embedding_batch * cos_stride_batch +
                             cos_sequence * cos_stride_sequence +
                             column * cos_stride_column;

  float value = as_float(input[input_offset]);
  float rotated = as_float(input[rotated_offset]);
  if (column < half_dim) {
    rotated = -rotated;
  }
  __nv_bfloat16 first = round_bf16(value * as_float(cos[embedding_offset]));
  __nv_bfloat16 second = round_bf16(rotated * as_float(sin[embedding_offset]));
  return round_bf16(as_float(first) + as_float(second));
}

__global__ void fused_rope_qk_bf16_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *key,
    const __nv_bfloat16 *cos, const __nv_bfloat16 *sin,
    __nv_bfloat16 *query_output, __nv_bfloat16 *key_output,
    int64_t query_elements, int64_t key_elements, int64_t query_heads,
    int64_t key_heads, int64_t query_sequence, int64_t key_sequence,
    int64_t head_dim, int64_t query_stride_batch, int64_t query_stride_head,
    int64_t query_stride_sequence, int64_t query_stride_column,
    int64_t key_stride_batch, int64_t key_stride_head,
    int64_t key_stride_sequence, int64_t key_stride_column, int64_t cos_batch,
    int64_t cos_sequence, int64_t cos_stride_batch, int64_t cos_stride_sequence,
    int64_t cos_stride_column) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = query_elements + key_elements;
  if (index >= total) {
    return;
  }

  int64_t batch;
  int64_t head;
  int64_t sequence;
  int64_t column;
  if (index < query_elements) {
    decode_bhsd(index, query_heads, query_sequence, head_dim, batch, head,
                sequence, column);
    query_output[index] =
        rope_element(query, cos, sin, batch, head, sequence, column, head_dim,
                     query_stride_batch, query_stride_head,
                     query_stride_sequence, query_stride_column, cos_batch,
                     sequence + cos_sequence - query_sequence, cos_stride_batch,
                     cos_stride_sequence, cos_stride_column);
    return;
  }

  int64_t key_index = index - query_elements;
  decode_bhsd(key_index, key_heads, key_sequence, head_dim, batch, head,
              sequence, column);
  key_output[key_index] = rope_element(
      key, cos, sin, batch, head, sequence, column, head_dim, key_stride_batch,
      key_stride_head, key_stride_sequence, key_stride_column, cos_batch,
      sequence + cos_sequence - key_sequence, cos_stride_batch,
      cos_stride_sequence, cos_stride_column);
}

__global__ void fused_rope_qk_pair_bf16_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *key,
    const __nv_bfloat16 *cos, const __nv_bfloat16 *sin,
    __nv_bfloat16 *query_output, __nv_bfloat16 *key_output, int64_t query_pairs,
    int64_t key_pairs, int64_t query_heads, int64_t key_heads,
    int64_t query_sequence, int64_t key_sequence, int64_t head_dim,
    int64_t query_stride_batch, int64_t query_stride_head,
    int64_t query_stride_sequence, int64_t query_stride_column,
    int64_t key_stride_batch, int64_t key_stride_head,
    int64_t key_stride_sequence, int64_t key_stride_column, int64_t cos_batch,
    int64_t cos_sequence, int64_t cos_stride_batch, int64_t cos_stride_sequence,
    int64_t cos_stride_column) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= query_pairs + key_pairs) {
    return;
  }

  bool is_key = index >= query_pairs;
  int64_t pair_index = is_key ? index - query_pairs : index;
  int64_t heads = is_key ? key_heads : query_heads;
  int64_t sequence_length = is_key ? key_sequence : query_sequence;
  int64_t half_dim = head_dim / 2;
  int64_t column = pair_index % half_dim;
  int64_t remaining = pair_index / half_dim;
  int64_t sequence = remaining % sequence_length;
  remaining /= sequence_length;
  int64_t head = remaining % heads;
  int64_t batch = remaining / heads;
  int64_t paired_column = column + half_dim;

  const __nv_bfloat16 *input = is_key ? key : query;
  __nv_bfloat16 *output = is_key ? key_output : query_output;
  int64_t stride_batch = is_key ? key_stride_batch : query_stride_batch;
  int64_t stride_head = is_key ? key_stride_head : query_stride_head;
  int64_t stride_sequence =
      is_key ? key_stride_sequence : query_stride_sequence;
  int64_t stride_column = is_key ? key_stride_column : query_stride_column;
  int64_t first_offset = offset_4d(batch, head, sequence, column, stride_batch,
                                   stride_head, stride_sequence, stride_column);
  int64_t second_offset =
      offset_4d(batch, head, sequence, paired_column, stride_batch, stride_head,
                stride_sequence, stride_column);
  int64_t embedding_batch = cos_batch == 1 ? 0 : batch;
  int64_t embedding_sequence = sequence + cos_sequence - sequence_length;
  int64_t embedding_base = embedding_batch * cos_stride_batch +
                           embedding_sequence * cos_stride_sequence;
  int64_t first_embedding = embedding_base + column * cos_stride_column;
  int64_t second_embedding = embedding_base + paired_column * cos_stride_column;

  float first_value = as_float(input[first_offset]);
  float second_value = as_float(input[second_offset]);
  __nv_bfloat16 first_product =
      round_bf16(first_value * as_float(cos[first_embedding]));
  __nv_bfloat16 first_rotated_product =
      round_bf16(-second_value * as_float(sin[first_embedding]));
  __nv_bfloat16 second_product =
      round_bf16(second_value * as_float(cos[second_embedding]));
  __nv_bfloat16 second_rotated_product =
      round_bf16(first_value * as_float(sin[second_embedding]));

  int64_t output_base =
      ((batch * heads + head) * sequence_length + sequence) * head_dim;
  output[output_base + column] =
      round_bf16(as_float(first_product) + as_float(first_rotated_product));
  output[output_base + paired_column] =
      round_bf16(as_float(second_product) + as_float(second_rotated_product));
}

__global__ void fused_vision_rope_qk_bf16_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *key, const float *cos,
    const float *sin, __nv_bfloat16 *query_output, __nv_bfloat16 *key_output,
    int64_t elements, int64_t heads, int64_t sequence_length, int64_t head_dim,
    int64_t query_stride_sequence, int64_t query_stride_head,
    int64_t query_stride_column, int64_t key_stride_sequence,
    int64_t key_stride_head, int64_t key_stride_column,
    int64_t cos_stride_sequence, int64_t cos_stride_column) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t pair_count = elements / 2;
  if (index >= pair_count) {
    return;
  }

  int64_t half_dim = head_dim / 2;
  int64_t column = index % half_dim;
  int64_t remaining = index / half_dim;
  int64_t head = remaining % heads;
  int64_t sequence = remaining / heads;
  int64_t paired_column = column + half_dim;
  int64_t embedding_first =
      sequence * cos_stride_sequence + column * cos_stride_column;
  int64_t embedding_second =
      sequence * cos_stride_sequence + paired_column * cos_stride_column;
  int64_t output_base = (sequence * heads + head) * head_dim;

  auto rotate_pair = [&](const __nv_bfloat16 *input, __nv_bfloat16 *output,
                         int64_t stride_sequence, int64_t stride_head,
                         int64_t stride_column) {
    int64_t first_offset = sequence * stride_sequence + head * stride_head +
                           column * stride_column;
    int64_t second_offset = sequence * stride_sequence + head * stride_head +
                            paired_column * stride_column;
    float first_value = as_float(input[first_offset]);
    float second_value = as_float(input[second_offset]);
    float first = __fadd_rn(__fmul_rn(first_value, cos[embedding_first]),
                            __fmul_rn(-second_value, sin[embedding_first]));
    float second = __fadd_rn(__fmul_rn(second_value, cos[embedding_second]),
                             __fmul_rn(first_value, sin[embedding_second]));
    output[output_base + column] = round_bf16(first);
    output[output_base + paired_column] = round_bf16(second);
  };

  rotate_pair(query, query_output, query_stride_sequence, query_stride_head,
              query_stride_column);
  rotate_pair(key, key_output, key_stride_sequence, key_stride_head,
              key_stride_column);
}

__global__ void fused_vision_rope_qk_scalar_bf16_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *key, const float *cos,
    const float *sin, __nv_bfloat16 *query_output, __nv_bfloat16 *key_output,
    int64_t elements, int64_t heads, int64_t head_dim,
    int64_t query_stride_sequence, int64_t query_stride_head,
    int64_t query_stride_column, int64_t key_stride_sequence,
    int64_t key_stride_head, int64_t key_stride_column,
    int64_t cos_stride_sequence, int64_t cos_stride_column) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= 2 * elements) {
    return;
  }
  bool is_key = index >= elements;
  int64_t output_index = is_key ? index - elements : index;
  int64_t column = output_index % head_dim;
  int64_t remaining = output_index / head_dim;
  int64_t head = remaining % heads;
  int64_t sequence = remaining / heads;
  int64_t half_dim = head_dim / 2;
  int64_t rotated_column =
      column < half_dim ? column + half_dim : column - half_dim;
  const __nv_bfloat16 *input = is_key ? key : query;
  __nv_bfloat16 *output = is_key ? key_output : query_output;
  int64_t stride_sequence =
      is_key ? key_stride_sequence : query_stride_sequence;
  int64_t stride_head = is_key ? key_stride_head : query_stride_head;
  int64_t stride_column = is_key ? key_stride_column : query_stride_column;
  int64_t input_offset =
      sequence * stride_sequence + head * stride_head + column * stride_column;
  int64_t rotated_offset = sequence * stride_sequence + head * stride_head +
                           rotated_column * stride_column;
  int64_t embedding_offset =
      sequence * cos_stride_sequence + column * cos_stride_column;
  float value = as_float(input[input_offset]);
  float rotated = as_float(input[rotated_offset]);
  if (column < half_dim) {
    rotated = -rotated;
  }
  float first = __fmul_rn(value, cos[embedding_offset]);
  float second = __fmul_rn(rotated, sin[embedding_offset]);
  output[output_index] = round_bf16(__fadd_rn(first, second));
}

__global__ void fused_paged_cache_update_rope_bf16_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *new_key,
    const __nv_bfloat16 *new_value, __nv_bfloat16 *raw_key_storage,
    __nv_bfloat16 *rotated_key_storage, __nv_bfloat16 *value_storage,
    const __nv_bfloat16 *cos, const __nv_bfloat16 *sin,
    __nv_bfloat16 *query_output, int64_t query_elements,
    int64_t rotated_key_elements, int64_t value_vectors, int64_t query_heads,
    int64_t key_heads, int64_t query_sequence, int64_t old_sequence,
    int64_t capacity, int64_t head_dim, int64_t query_stride_batch,
    int64_t query_stride_head, int64_t query_stride_sequence,
    int64_t query_stride_column, int64_t key_stride_batch,
    int64_t key_stride_head, int64_t key_stride_sequence,
    int64_t key_stride_column, int64_t value_stride_batch,
    int64_t value_stride_head, int64_t value_stride_sequence,
    int64_t value_stride_column, int64_t cos_batch, int64_t cos_sequence,
    int64_t cos_stride_batch, int64_t cos_stride_sequence,
    int64_t cos_stride_column, bool rerotate_full) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = query_elements + rotated_key_elements + value_vectors;
  if (index >= total) {
    return;
  }

  int64_t batch;
  int64_t head;
  int64_t sequence;
  int64_t column;
  if (index < query_elements) {
    decode_bhsd(index, query_heads, query_sequence, head_dim, batch, head,
                sequence, column);
    query_output[index] =
        rope_element(query, cos, sin, batch, head, sequence, column, head_dim,
                     query_stride_batch, query_stride_head,
                     query_stride_sequence, query_stride_column, cos_batch,
                     sequence + cos_sequence - query_sequence, cos_stride_batch,
                     cos_stride_sequence, cos_stride_column);
    return;
  }

  int64_t key_region_index = index - query_elements;
  int64_t full_sequence = old_sequence + query_sequence;
  int64_t rotated_sequence = rerotate_full ? full_sequence : query_sequence;
  if (key_region_index < rotated_key_elements) {
    decode_bhsd(key_region_index, key_heads, rotated_sequence, head_dim, batch,
                head, sequence, column);
    int64_t storage_sequence =
        rerotate_full ? sequence : old_sequence + sequence;
    bool from_prefix = rerotate_full && sequence < old_sequence;
    int64_t source_sequence =
        from_prefix ? storage_sequence : storage_sequence - old_sequence;
    const __nv_bfloat16 *source = from_prefix ? raw_key_storage : new_key;
    int64_t source_stride_batch =
        from_prefix ? capacity * key_heads * head_dim : key_stride_batch;
    int64_t source_stride_head = from_prefix ? head_dim : key_stride_head;
    int64_t source_stride_sequence =
        from_prefix ? key_heads * head_dim : key_stride_sequence;
    int64_t source_stride_column = from_prefix ? 1 : key_stride_column;
    int64_t storage_offset =
        (storage_sequence * key_heads + head) * head_dim + column;
    if (!from_prefix) {
      int64_t source_offset =
          offset_4d(batch, head, source_sequence, column, key_stride_batch,
                    key_stride_head, key_stride_sequence, key_stride_column);
      raw_key_storage[storage_offset] = new_key[source_offset];
    }
    int64_t cos_position = rerotate_full ? storage_sequence : sequence;
    rotated_key_storage[storage_offset] = rope_element(
        source, cos, sin, batch, head, source_sequence, column, head_dim,
        source_stride_batch, source_stride_head, source_stride_sequence,
        source_stride_column, cos_batch, cos_position, cos_stride_batch,
        cos_stride_sequence, cos_stride_column);
    return;
  }

  constexpr int kVectorElements = sizeof(uint4) / sizeof(__nv_bfloat16);
  int64_t value_vector_index = key_region_index - rotated_key_elements;
  int64_t column_vector;
  decode_bhsd(value_vector_index, key_heads, query_sequence,
              head_dim / kVectorElements, batch, head, sequence, column_vector);
  column = column_vector * kVectorElements;
  int64_t source_offset =
      offset_4d(batch, head, sequence, column, value_stride_batch,
                value_stride_head, value_stride_sequence, value_stride_column);
  int64_t storage_offset =
      ((old_sequence + sequence) * key_heads + head) * head_dim + column;
  *reinterpret_cast<uint4 *>(value_storage + storage_offset) =
      *reinterpret_cast<const uint4 *>(new_value + source_offset);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <bool AddResidual>
__global__ void
rms_norm_bf16_kernel(const __nv_bfloat16 *input, const __nv_bfloat16 *residual,
                     const __nv_bfloat16 *weight, __nv_bfloat16 *summed_output,
                     __nv_bfloat16 *normalized_output, int64_t columns,
                     float epsilon) {
  int64_t row = blockIdx.x;
  int64_t row_offset = row * columns;
  float sum_squares = 0.0f;

  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    __nv_bfloat16 value = input[row_offset + column];
    if constexpr (AddResidual) {
      value =
          round_bf16(as_float(value) + as_float(residual[row_offset + column]));
      summed_output[row_offset + column] = value;
    }
    float value_float = as_float(value);
    sum_squares = fmaf(value_float, value_float, sum_squares);
  }

  sum_squares = warp_sum(sum_squares);
  __shared__ float warp_sums[32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum_squares;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    block_sum = warp_sum(block_sum);
    if (lane == 0) {
      warp_sums[0] = rsqrtf(block_sum / static_cast<float>(columns) + epsilon);
    }
  }
  __syncthreads();
  float inverse_rms = warp_sums[0];

  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    __nv_bfloat16 value = AddResidual ? summed_output[row_offset + column]
                                      : input[row_offset + column];
    __nv_bfloat16 normalized = round_bf16(as_float(value) * inverse_rms);
    normalized_output[row_offset + column] =
        round_bf16(as_float(normalized) * as_float(weight[column]));
  }
}

__global__ void fused_cache_update_rope_bf16_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *old_key,
    const __nv_bfloat16 *old_value, const __nv_bfloat16 *old_rotated_key,
    const __nv_bfloat16 *new_key, const __nv_bfloat16 *new_value,
    const __nv_bfloat16 *cos, const __nv_bfloat16 *sin,
    __nv_bfloat16 *query_output, __nv_bfloat16 *key_output,
    __nv_bfloat16 *value_output, __nv_bfloat16 *rotated_key_output,
    int64_t query_elements, int64_t full_cache_vectors, int64_t query_heads,
    int64_t key_heads, int64_t query_sequence, int64_t old_sequence,
    int64_t new_sequence, int64_t head_dim, int64_t query_stride_batch,
    int64_t query_stride_head, int64_t query_stride_sequence,
    int64_t query_stride_column, int64_t old_key_stride_batch,
    int64_t old_key_stride_head, int64_t old_key_stride_sequence,
    int64_t old_key_stride_column, int64_t old_value_stride_batch,
    int64_t old_value_stride_head, int64_t old_value_stride_sequence,
    int64_t old_value_stride_column, int64_t old_rotated_stride_batch,
    int64_t old_rotated_stride_head, int64_t old_rotated_stride_sequence,
    int64_t old_rotated_stride_column, int64_t new_key_stride_batch,
    int64_t new_key_stride_head, int64_t new_key_stride_sequence,
    int64_t new_key_stride_column, int64_t new_value_stride_batch,
    int64_t new_value_stride_head, int64_t new_value_stride_sequence,
    int64_t new_value_stride_column, int64_t cos_batch,
    int64_t cos_stride_batch, int64_t cos_stride_sequence,
    int64_t cos_stride_column) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = query_elements + 3 * full_cache_vectors;
  if (index >= total) {
    return;
  }

  int64_t batch;
  int64_t head;
  int64_t sequence;
  int64_t column;
  if (index < query_elements) {
    decode_bhsd(index, query_heads, query_sequence, head_dim, batch, head,
                sequence, column);
    query_output[index] = rope_element(
        query, cos, sin, batch, head, sequence, column, head_dim,
        query_stride_batch, query_stride_head, query_stride_sequence,
        query_stride_column, cos_batch, sequence, cos_stride_batch,
        cos_stride_sequence, cos_stride_column);
    return;
  }

  constexpr int kVectorElements = sizeof(uint4) / sizeof(__nv_bfloat16);
  int64_t region_index = index - query_elements;
  int region = static_cast<int>(region_index / full_cache_vectors);
  int64_t cache_vector_index = region_index % full_cache_vectors;
  int64_t full_sequence = old_sequence + new_sequence;
  int64_t column_vector;
  decode_bhsd(cache_vector_index, key_heads, full_sequence,
              head_dim / kVectorElements, batch, head, sequence, column_vector);
  column = column_vector * kVectorElements;

  bool from_prefix = sequence < old_sequence;
  if (region == 0) {
    int64_t source;
    if (from_prefix) {
      source = offset_4d(batch, head, sequence, column, old_key_stride_batch,
                         old_key_stride_head, old_key_stride_sequence,
                         old_key_stride_column);
    } else {
      source = offset_4d(batch, head, sequence - old_sequence, column,
                         new_key_stride_batch, new_key_stride_head,
                         new_key_stride_sequence, new_key_stride_column);
    }
    const __nv_bfloat16 *source_base = from_prefix ? old_key : new_key;
    reinterpret_cast<uint4 *>(key_output)[cache_vector_index] =
        *reinterpret_cast<const uint4 *>(source_base + source);
    return;
  }

  if (region == 1) {
    int64_t source;
    if (from_prefix) {
      source = offset_4d(batch, head, sequence, column, old_value_stride_batch,
                         old_value_stride_head, old_value_stride_sequence,
                         old_value_stride_column);
    } else {
      source = offset_4d(batch, head, sequence - old_sequence, column,
                         new_value_stride_batch, new_value_stride_head,
                         new_value_stride_sequence, new_value_stride_column);
    }
    const __nv_bfloat16 *source_base = from_prefix ? old_value : new_value;
    reinterpret_cast<uint4 *>(value_output)[cache_vector_index] =
        *reinterpret_cast<const uint4 *>(source_base + source);
    return;
  }

  if (from_prefix) {
    int64_t source =
        offset_4d(batch, head, sequence, column, old_rotated_stride_batch,
                  old_rotated_stride_head, old_rotated_stride_sequence,
                  old_rotated_stride_column);
    reinterpret_cast<uint4 *>(rotated_key_output)[cache_vector_index] =
        *reinterpret_cast<const uint4 *>(old_rotated_key + source);
  } else {
    int64_t tail_sequence = sequence - old_sequence;
#pragma unroll
    for (int element = 0; element < kVectorElements; ++element) {
      int64_t cache_index = cache_vector_index * kVectorElements + element;
      rotated_key_output[cache_index] = rope_element(
          new_key, cos, sin, batch, head, tail_sequence, column + element,
          head_dim, new_key_stride_batch, new_key_stride_head,
          new_key_stride_sequence, new_key_stride_column, cos_batch,
          tail_sequence, cos_stride_batch, cos_stride_sequence,
          cos_stride_column);
    }
  }
}

__global__ void fused_prefill_cache_update_rope_bf16_kernel(
    const __nv_bfloat16 *query, const __nv_bfloat16 *old_key,
    const __nv_bfloat16 *old_value, const __nv_bfloat16 *new_key,
    const __nv_bfloat16 *new_value, const __nv_bfloat16 *cos,
    const __nv_bfloat16 *sin, __nv_bfloat16 *query_output,
    __nv_bfloat16 *key_output, __nv_bfloat16 *value_output,
    __nv_bfloat16 *rotated_key_output, int64_t query_elements,
    int64_t full_cache_elements, int64_t full_cache_vectors,
    int64_t query_heads, int64_t key_heads, int64_t query_sequence,
    int64_t old_sequence, int64_t new_sequence, int64_t head_dim,
    int64_t query_stride_batch, int64_t query_stride_head,
    int64_t query_stride_sequence, int64_t query_stride_column,
    int64_t old_key_stride_batch, int64_t old_key_stride_head,
    int64_t old_key_stride_sequence, int64_t old_key_stride_column,
    int64_t old_value_stride_batch, int64_t old_value_stride_head,
    int64_t old_value_stride_sequence, int64_t old_value_stride_column,
    int64_t new_key_stride_batch, int64_t new_key_stride_head,
    int64_t new_key_stride_sequence, int64_t new_key_stride_column,
    int64_t new_value_stride_batch, int64_t new_value_stride_head,
    int64_t new_value_stride_sequence, int64_t new_value_stride_column,
    int64_t cos_batch, int64_t cos_sequence, int64_t cos_stride_batch,
    int64_t cos_stride_sequence, int64_t cos_stride_column) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = query_elements + full_cache_elements + full_cache_vectors;
  if (index >= total) {
    return;
  }

  int64_t batch;
  int64_t head;
  int64_t sequence;
  int64_t column;
  if (index < query_elements) {
    decode_bhsd(index, query_heads, query_sequence, head_dim, batch, head,
                sequence, column);
    query_output[index] =
        rope_element(query, cos, sin, batch, head, sequence, column, head_dim,
                     query_stride_batch, query_stride_head,
                     query_stride_sequence, query_stride_column, cos_batch,
                     sequence + cos_sequence - query_sequence, cos_stride_batch,
                     cos_stride_sequence, cos_stride_column);
    return;
  }

  int64_t full_sequence = old_sequence + new_sequence;
  int64_t cache_region_index = index - query_elements;
  if (cache_region_index < full_cache_elements) {
    int64_t cache_index = cache_region_index;
    decode_bhsd(cache_index, key_heads, full_sequence, head_dim, batch, head,
                sequence, column);
    bool from_prefix = sequence < old_sequence;
    int64_t source_sequence = from_prefix ? sequence : sequence - old_sequence;
    const __nv_bfloat16 *source = from_prefix ? old_key : new_key;
    int64_t stride_batch =
        from_prefix ? old_key_stride_batch : new_key_stride_batch;
    int64_t stride_head =
        from_prefix ? old_key_stride_head : new_key_stride_head;
    int64_t stride_sequence =
        from_prefix ? old_key_stride_sequence : new_key_stride_sequence;
    int64_t stride_column =
        from_prefix ? old_key_stride_column : new_key_stride_column;
    int64_t source_offset =
        offset_4d(batch, head, source_sequence, column, stride_batch,
                  stride_head, stride_sequence, stride_column);
    key_output[cache_index] = source[source_offset];
    rotated_key_output[cache_index] = rope_element(
        source, cos, sin, batch, head, source_sequence, column, head_dim,
        stride_batch, stride_head, stride_sequence, stride_column, cos_batch,
        sequence, cos_stride_batch, cos_stride_sequence, cos_stride_column);
    return;
  }

  constexpr int kVectorElements = sizeof(uint4) / sizeof(__nv_bfloat16);
  int64_t value_vector_index = cache_region_index - full_cache_elements;
  int64_t column_vector;
  decode_bhsd(value_vector_index, key_heads, full_sequence,
              head_dim / kVectorElements, batch, head, sequence, column_vector);
  column = column_vector * kVectorElements;
  bool from_prefix = sequence < old_sequence;
  int64_t source;
  const __nv_bfloat16 *source_base;
  if (from_prefix) {
    source = offset_4d(batch, head, sequence, column, old_value_stride_batch,
                       old_value_stride_head, old_value_stride_sequence,
                       old_value_stride_column);
    source_base = old_value;
  } else {
    source = offset_4d(batch, head, sequence - old_sequence, column,
                       new_value_stride_batch, new_value_stride_head,
                       new_value_stride_sequence, new_value_stride_column);
    source_base = new_value;
  }
  reinterpret_cast<uint4 *>(value_output)[value_vector_index] =
      *reinterpret_cast<const uint4 *>(source_base + source);
}

__device__ __forceinline__ int64_t sfa_offset(int64_t row, int64_t sf_idx,
                                              int64_t blocks_k) {
  return ((row >> 7) * blocks_k << 9) + ((row & 0x1f) << 4) +
         (((row & 0x7f) >> 5) << 2) + ((sf_idx >> 2) << 9) + (sf_idx & 0x3);
}

__device__ __forceinline__ int64_t sfb_offset(int64_t sf_idx) {
  return ((sf_idx >> 2) << 9) + (sf_idx & 0x3);
}

template <typename scalar_t>
__device__ __forceinline__ float load_as_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__global__ void pack_matrix_fp4_kernel(const scalar_t *__restrict__ input,
                                       uint8_t *__restrict__ packed,
                                       uint8_t *__restrict__ scales,
                                       int64_t rows, int64_t cols,
                                       int64_t packed_cols, int64_t blocks_k,
                                       int64_t sf_per_row) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * sf_per_row;
  if (idx >= total) {
    return;
  }

  int64_t row = idx / sf_per_row;
  int64_t sf_idx = idx - row * sf_per_row;
  int64_t col0 = sf_idx * kSfVecSize;

  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < kSfVecSize; ++i) {
    int64_t col = col0 + i;
    float value = col < cols ? load_as_float(input[row * cols + col]) : 0.0f;
    max_abs = fmaxf(max_abs, fabsf(value));
  }

  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  cutlass::float_e4m3_t scale_fp8(scale);
  float scale_deq = static_cast<float>(scale_fp8);
  if (!(scale_deq > 0.0f)) {
    scale_deq = scale;
  }
  scales[sfa_offset(row, sf_idx, blocks_k)] = scale_fp8.raw();

#pragma unroll
  for (int i = 0; i < kSfVecSize; i += 2) {
    int64_t col = col0 + i;
    float v0 =
        col < cols ? load_as_float(input[row * cols + col]) / scale_deq : 0.0f;
    float v1 = (col + 1) < cols
                   ? load_as_float(input[row * cols + col + 1]) / scale_deq
                   : 0.0f;
    cutlass::float_e2m1_t q0(v0);
    cutlass::float_e2m1_t q1(v1);
    packed[row * packed_cols + (col >> 1)] =
        static_cast<uint8_t>((q0.raw() & 0x0f) | ((q1.raw() & 0x0f) << 4));
  }
}

template <typename scalar_t>
__global__ void pack_vector_fp4_kernel(const scalar_t *__restrict__ input,
                                       uint8_t *__restrict__ packed,
                                       uint8_t *__restrict__ scales,
                                       int64_t cols, int64_t sf_per_row) {
  int64_t sf_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (sf_idx >= sf_per_row) {
    return;
  }

  int64_t col0 = sf_idx * kSfVecSize;
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < kSfVecSize; ++i) {
    int64_t col = col0 + i;
    float value = col < cols ? load_as_float(input[col]) : 0.0f;
    max_abs = fmaxf(max_abs, fabsf(value));
  }

  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  cutlass::float_e4m3_t scale_fp8(scale);
  float scale_deq = static_cast<float>(scale_fp8);
  if (!(scale_deq > 0.0f)) {
    scale_deq = scale;
  }
  scales[sfb_offset(sf_idx)] = scale_fp8.raw();

#pragma unroll
  for (int i = 0; i < kSfVecSize; i += 2) {
    int64_t col = col0 + i;
    float v0 = col < cols ? load_as_float(input[col]) / scale_deq : 0.0f;
    float v1 =
        (col + 1) < cols ? load_as_float(input[col + 1]) / scale_deq : 0.0f;
    cutlass::float_e2m1_t q0(v0);
    cutlass::float_e2m1_t q1(v1);
    packed[col >> 1] =
        static_cast<uint8_t>((q0.raw() & 0x0f) | ((q1.raw() & 0x0f) << 4));
  }
}

template <typename scalar_t>
__global__ void pack_matrix_fp4_ue_rowmajor_kernel(
    const scalar_t *__restrict__ input, uint8_t *__restrict__ packed,
    uint8_t *__restrict__ scales, int64_t rows, int64_t cols,
    int64_t packed_cols, int64_t blocks_k, int64_t sf_per_row) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * sf_per_row;
  if (idx >= total) {
    return;
  }

  int64_t row = idx / sf_per_row;
  int64_t sf_idx = idx - row * sf_per_row;
  int64_t col0 = sf_idx * kSfVecSize;

  const scalar_t *input_group = input + row * cols + col0;
  const uint4 *input_vec = reinterpret_cast<const uint4 *>(input_group);
  uint4 lower = input_vec[0];
  uint4 upper = input_vec[1];
  const scalar_t *lower_values = reinterpret_cast<const scalar_t *>(&lower);
  const scalar_t *upper_values = reinterpret_cast<const scalar_t *>(&upper);
  float values[kSfVecSize];

  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < kSfVecSize / 2; ++i) {
    values[i] = load_as_float(lower_values[i]);
    values[kSfVecSize / 2 + i] = load_as_float(upper_values[i]);
    max_abs = fmaxf(max_abs, fabsf(values[i]));
    max_abs = fmaxf(max_abs, fabsf(values[kSfVecSize / 2 + i]));
  }

  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  cutlass::float_ue4m3_t scale_fp8(scale);
  float scale_deq = static_cast<float>(scale_fp8);
  if (!(scale_deq > 0.0f)) {
    scale_deq = scale;
  }

  float reciprocal_scale = 1.0f / scale_deq;
  uint64_t packed_bits = 0;
#pragma unroll
  for (int i = 0; i < kSfVecSize; i += 2) {
    __nv_fp4x2_storage_t pair =
        __nv_cvt_float2_to_fp4x2(make_float2(values[i] * reciprocal_scale,
                                             values[i + 1] * reciprocal_scale),
                                 __NV_E2M1, cudaRoundNearest);
    packed_bits |= static_cast<uint64_t>(static_cast<uint8_t>(pair)) << (4 * i);
  }
  reinterpret_cast<uint64_t *>(packed + row * packed_cols)[sf_idx] =
      packed_bits;
  scales[sfa_offset(row, sf_idx, blocks_k)] = scale_fp8.raw();
}

template <bool AddResidual, bool WriteNormalized>
__global__ void rms_norm_fp4_quant_bf16_kernel(
    const __nv_bfloat16 *input, const __nv_bfloat16 *residual,
    const __nv_bfloat16 *weight, __nv_bfloat16 *summed_output,
    __nv_bfloat16 *normalized_output, uint8_t *packed_output,
    uint8_t *scale_output, int64_t columns, int64_t packed_cols,
    int64_t blocks_k, int64_t sf_per_row, float epsilon) {
  extern __shared__ __nv_bfloat16 normalized_shared[];
  __shared__ float warp_sums[32];
  int64_t row = blockIdx.x;
  int64_t row_offset = row * columns;
  float sum_squares = 0.0f;

  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    __nv_bfloat16 value = input[row_offset + column];
    if constexpr (AddResidual) {
      value =
          round_bf16(as_float(value) + as_float(residual[row_offset + column]));
      summed_output[row_offset + column] = value;
    }
    float value_float = as_float(value);
    sum_squares = fmaf(value_float, value_float, sum_squares);
  }

  sum_squares = warp_sum(sum_squares);
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum_squares;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
    block_sum = warp_sum(block_sum);
    if (lane == 0) {
      warp_sums[0] = rsqrtf(block_sum / static_cast<float>(columns) + epsilon);
    }
  }
  __syncthreads();
  float inverse_rms = warp_sums[0];

  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    __nv_bfloat16 value = AddResidual ? summed_output[row_offset + column]
                                      : input[row_offset + column];
    __nv_bfloat16 normalized = round_bf16(as_float(value) * inverse_rms);
    normalized = round_bf16(as_float(normalized) * as_float(weight[column]));
    if constexpr (WriteNormalized) {
      normalized_output[row_offset + column] = normalized;
    }
    normalized_shared[column] = normalized;
  }
  __syncthreads();

  for (int64_t sf_idx = threadIdx.x; sf_idx < sf_per_row;
       sf_idx += blockDim.x) {
    int64_t col0 = sf_idx * kSfVecSize;
    float values[kSfVecSize];
    float max_abs = 0.0f;
#pragma unroll
    for (int i = 0; i < kSfVecSize; ++i) {
      values[i] = as_float(normalized_shared[col0 + i]);
      max_abs = fmaxf(max_abs, fabsf(values[i]));
    }

    float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
    cutlass::float_ue4m3_t scale_fp8(scale);
    float scale_deq = static_cast<float>(scale_fp8);
    if (!(scale_deq > 0.0f)) {
      scale_deq = scale;
    }
    float reciprocal_scale = 1.0f / scale_deq;
    uint64_t packed_bits = 0;
#pragma unroll
    for (int i = 0; i < kSfVecSize; i += 2) {
      __nv_fp4x2_storage_t pair = __nv_cvt_float2_to_fp4x2(
          make_float2(values[i] * reciprocal_scale,
                      values[i + 1] * reciprocal_scale),
          __NV_E2M1, cudaRoundNearest);
      packed_bits |= static_cast<uint64_t>(static_cast<uint8_t>(pair))
                     << (4 * i);
    }
    reinterpret_cast<uint64_t *>(packed_output + row * packed_cols)[sf_idx] =
        packed_bits;
    scale_output[sfa_offset(row, sf_idx, blocks_k)] = scale_fp8.raw();
  }
}

__global__ void add_bias_bf16_kernel(const at::BFloat16 *__restrict__ bias,
                                     at::BFloat16 *__restrict__ output,
                                     int64_t leading, int64_t rows) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = leading * rows;
  if (idx >= total) {
    return;
  }
  int64_t col = idx % rows;
  output[idx] = at::BFloat16(static_cast<float>(output[idx]) +
                             static_cast<float>(bias[col]));
}

template <bool kInterleaved>
__global__ void silu_mul_bf16_kernel(const __nv_bfloat162 *__restrict__ gate_up,
                                     __nv_bfloat162 *__restrict__ output,
                                     int64_t leading, int64_t pairs_per_row) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = leading * pairs_per_row;
  if (idx >= total) {
    return;
  }
  int64_t row = idx / pairs_per_row;
  int64_t col_pair = idx - row * pairs_per_row;
  int64_t row_offset = row * pairs_per_row * 2;
  int64_t gate_idx = row_offset + col_pair;
  int64_t up_idx = row_offset + pairs_per_row + col_pair;
  if constexpr (kInterleaved) {
    constexpr int64_t kPairsPerBlock = 32;
    int64_t block = col_pair / kPairsPerBlock;
    int64_t pair_in_block = col_pair % kPairsPerBlock;
    up_idx = row_offset + block * (2 * kPairsPerBlock) + pair_in_block;
    gate_idx = up_idx + kPairsPerBlock;
  }
  __nv_bfloat162 gate = gate_up[gate_idx];
  __nv_bfloat162 up = gate_up[up_idx];
  __nv_bfloat162 denominator =
      __hadd2(__float2bfloat162_rn(1.0f), h2exp(__hneg2(gate)));
  output[idx] = __hmul2(__hmul2(gate, h2rcp(denominator)), up);
}

template <bool kInterleaved>
__global__ void
silu_mul_bf16_scalar_kernel(const at::BFloat16 *__restrict__ gate_up,
                            at::BFloat16 *__restrict__ output,
                            int64_t intermediate_size) {
  int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= intermediate_size) {
    return;
  }
  int64_t gate_idx = col;
  int64_t up_idx = intermediate_size + col;
  if constexpr (kInterleaved) {
    constexpr int64_t kBlockCols = 64;
    int64_t block = col / kBlockCols;
    int64_t col_in_block = col % kBlockCols;
    up_idx = block * (2 * kBlockCols) + col_in_block;
    gate_idx = up_idx + kBlockCols;
  }
  float gate = static_cast<float>(gate_up[gate_idx]);
  float up = static_cast<float>(gate_up[up_idx]);
  float silu = gate / (1.0f + expf(-gate));
  output[col] = at::BFloat16(silu * up);
}

template <bool kInterleaved>
__global__ void silu_mul_pack_fp4_ue_rowmajor_kernel(
    const __nv_bfloat162 *__restrict__ gate_up, uint8_t *__restrict__ packed,
    uint8_t *__restrict__ scales, int64_t rows, int64_t cols,
    int64_t packed_cols, int64_t blocks_k, int64_t sf_per_row) {
  constexpr int kGroupsPerWarp = 4;
  constexpr int kLanesPerGroup = 8;
  int warp_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 0x1f;
  int group_in_warp = lane / kLanesPerGroup;
  int lane_in_group = lane & (kLanesPerGroup - 1);
  int64_t global_warp = blockIdx.x * (blockDim.x / warpSize) + warp_in_block;
  int64_t idx = global_warp * kGroupsPerWarp + group_in_warp;
  int64_t total = rows * sf_per_row;
  if (idx >= total) {
    return;
  }

  int64_t row = idx / sf_per_row;
  int64_t sf_idx = idx - row * sf_per_row;
  int64_t gate_idx = row * cols + sf_idx * kLanesPerGroup + lane_in_group;
  int64_t up_idx = gate_idx + cols / 2;
  if constexpr (kInterleaved) {
    constexpr int64_t kPairsPerBlock = 32;
    int64_t block = sf_idx / 4;
    int64_t pair_in_block = (sf_idx % 4) * kLanesPerGroup + lane_in_group;
    up_idx = row * cols + block * (2 * kPairsPerBlock) + pair_in_block;
    gate_idx = up_idx + kPairsPerBlock;
  }
  __nv_bfloat162 gate = gate_up[gate_idx];
  __nv_bfloat162 up = gate_up[up_idx];
  __nv_bfloat162 denominator =
      __hadd2(__float2bfloat162_rn(1.0f), h2exp(__hneg2(gate)));
  __nv_bfloat162 activated = __hmul2(__hmul2(gate, h2rcp(denominator)), up);
  float2 values = __bfloat1622float2(activated);
  float max_abs = fmaxf(fabsf(values.x), fabsf(values.y));

  unsigned group_mask = 0xffu << (group_in_warp * kLanesPerGroup);
#pragma unroll
  for (int offset = kLanesPerGroup / 2; offset > 0; offset >>= 1) {
    max_abs = fmaxf(
        max_abs, __shfl_xor_sync(group_mask, max_abs, offset, kLanesPerGroup));
  }

  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  cutlass::float_ue4m3_t scale_fp8(scale);
  float scale_deq = static_cast<float>(scale_fp8);
  if (!(scale_deq > 0.0f)) {
    scale_deq = scale;
  }
  float reciprocal_scale = 1.0f / scale_deq;
  __nv_fp4x2_storage_t pair = __nv_cvt_float2_to_fp4x2(
      make_float2(values.x * reciprocal_scale, values.y * reciprocal_scale),
      __NV_E2M1, cudaRoundNearest);
  uint32_t pair_bits = static_cast<uint8_t>(pair);

  uint64_t packed_bits = 0;
#pragma unroll
  for (int source_lane = 0; source_lane < kLanesPerGroup; ++source_lane) {
    uint32_t source_bits =
        __shfl_sync(group_mask, pair_bits, source_lane, kLanesPerGroup);
    if (lane_in_group == 0) {
      packed_bits |= static_cast<uint64_t>(source_bits) << (8 * source_lane);
    }
  }
  if (lane_in_group == 0) {
    reinterpret_cast<uint64_t *>(packed + row * packed_cols)[sf_idx] =
        packed_bits;
    scales[sfa_offset(row, sf_idx, blocks_k)] = scale_fp8.raw();
  }
}

template <bool kInterleaved>
__global__ void silu_mul_pack_fp4_ue_rowmajor_vector_kernel(
    const at::BFloat16 *__restrict__ gate_up, uint8_t *__restrict__ packed,
    uint8_t *__restrict__ scales, int64_t rows, int64_t cols,
    int64_t packed_cols, int64_t blocks_k, int64_t sf_per_row) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * sf_per_row;
  if (idx >= total) {
    return;
  }

  int64_t row = idx / sf_per_row;
  int64_t sf_idx = idx - row * sf_per_row;
  int64_t col0 = sf_idx * kSfVecSize;
  const at::BFloat16 *gate_group = gate_up + row * 2 * cols + col0;
  const at::BFloat16 *up_group = gate_group + cols;
  if constexpr (kInterleaved) {
    constexpr int64_t kBlockCols = 64;
    int64_t block = col0 / kBlockCols;
    int64_t col_in_block = col0 % kBlockCols;
    up_group =
        gate_up + row * 2 * cols + block * (2 * kBlockCols) + col_in_block;
    gate_group = up_group + kBlockCols;
  }
  const uint4 *gate_vec = reinterpret_cast<const uint4 *>(gate_group);
  const uint4 *up_vec = reinterpret_cast<const uint4 *>(up_group);
  uint4 gate_lower = gate_vec[0];
  uint4 gate_upper = gate_vec[1];
  uint4 up_lower = up_vec[0];
  uint4 up_upper = up_vec[1];
  const __nv_bfloat162 *gate_lower_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(&gate_lower);
  const __nv_bfloat162 *gate_upper_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(&gate_upper);
  const __nv_bfloat162 *up_lower_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(&up_lower);
  const __nv_bfloat162 *up_upper_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(&up_upper);
  __nv_bfloat162 activated[kSfVecSize / 2];

  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < kSfVecSize / 2; ++i) {
    __nv_bfloat162 gate = i < kSfVecSize / 4
                              ? gate_lower_pairs[i]
                              : gate_upper_pairs[i - kSfVecSize / 4];
    __nv_bfloat162 up = i < kSfVecSize / 4 ? up_lower_pairs[i]
                                           : up_upper_pairs[i - kSfVecSize / 4];
    __nv_bfloat162 denominator =
        __hadd2(__float2bfloat162_rn(1.0f), h2exp(__hneg2(gate)));
    activated[i] = __hmul2(__hmul2(gate, h2rcp(denominator)), up);
    float2 values = __bfloat1622float2(activated[i]);
    max_abs = fmaxf(max_abs, fabsf(values.x));
    max_abs = fmaxf(max_abs, fabsf(values.y));
  }

  float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
  cutlass::float_ue4m3_t scale_fp8(scale);
  float scale_deq = static_cast<float>(scale_fp8);
  if (!(scale_deq > 0.0f)) {
    scale_deq = scale;
  }
  float reciprocal_scale = 1.0f / scale_deq;
  uint64_t packed_bits = 0;
#pragma unroll
  for (int i = 0; i < kSfVecSize / 2; ++i) {
    float2 values = __bfloat1622float2(activated[i]);
    __nv_fp4x2_storage_t pair = __nv_cvt_float2_to_fp4x2(
        make_float2(values.x * reciprocal_scale, values.y * reciprocal_scale),
        __NV_E2M1, cudaRoundNearest);
    packed_bits |= static_cast<uint64_t>(static_cast<uint8_t>(pair)) << (8 * i);
  }
  reinterpret_cast<uint64_t *>(packed + row * packed_cols)[sf_idx] =
      packed_bits;
  scales[sfa_offset(row, sf_idx, blocks_k)] = scale_fp8.raw();
}

template <typename bias_t>
__global__ void dequantize_output_fp4_kernel(const uint8_t *__restrict__ packed,
                                             const uint8_t *__restrict__ scales,
                                             const bias_t *__restrict__ bias,
                                             at::BFloat16 *__restrict__ output,
                                             int64_t rows, float epilogue_st) {
  int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  uint8_t byte = packed[row >> 1];
  uint8_t raw = (row & 1) ? (byte >> 4) : (byte & 0x0f);
  cutlass::float_e2m1_t value_fp4 = cutlass::float_e2m1_t::bitcast(raw);
  int64_t sf_block = row / kSfVecSize;
  int64_t sf_offset = ((sf_block >> 2) << 9) + (sf_block & 0x3);
  cutlass::float_e4m3_t scale_fp8 =
      cutlass::float_e4m3_t::bitcast(scales[sf_offset]);
  float value = static_cast<float>(value_fp4) * static_cast<float>(scale_fp8) /
                epilogue_st;
  if (bias != nullptr) {
    value += static_cast<float>(bias[row]);
  }
  output[row] = at::BFloat16(value);
}

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementSFA = cutlass::float_e4m3_t;
using ElementSFB = cutlass::float_e4m3_t;
using ElementC = cutlass::float_e2m1_t;
using ElementD = cutlass::float_e2m1_t;
using ElementSFD = cutlass::float_e4m3_t;
using ElementAccumulatorMainloop = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using LayoutSFD = cutlass::layout::ColumnMajor;
using ThreadShape = cutlass::gemm::GemmShape<16, 8>;

using EpilogueOp =
    cutlass::epilogue::threadblock::GemvEpilogueWithScalingFactor<
        kSfVecSize, ThreadShape, ElementCompute, ElementAccumulator, ElementC,
        ElementD, ElementSFD, LayoutD, LayoutSFD>;

using GemvKernel =
    cutlass::gemm::kernel::GemvBlockScaled<ElementA, LayoutA, ElementB,
                                           ElementC, ElementAccumulatorMainloop,
                                           EpilogueOp, kElementsPerAccess>;

using Gemv = cutlass::gemm::device::GemvBlockScaled<GemvKernel>;
using TensorRefA = typename GemvKernel::TensorRefA;

namespace gemm_bf16 {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;

template <typename MmaTileShape_, typename ClusterShape_,
          typename EpilogueTile_, int AlignmentCD, typename EpilogueSchedule_,
          typename KernelSchedule_>
struct GemmSpec {
  static constexpr bool kFusesBias = false;
  using MmaTileShape = MmaTileShape_;
  using ClusterShape = ClusterShape_;
  using EpilogueTile = EpilogueTile_;
  using EpilogueSchedule = EpilogueSchedule_;
  using KernelSchedule = KernelSchedule_;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape, EpilogueTile,
          ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag,
          AlignmentCD, ElementD, LayoutDTag, AlignmentCD,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename MmaTileShape_, typename ClusterShape_,
          typename EpilogueTile_, int AlignmentCD, typename EpilogueSchedule_,
          typename KernelSchedule_>
struct BiasGemmSpec {
  static constexpr bool kFusesBias = true;
  using MmaTileShape = MmaTileShape_;
  using ClusterShape = ClusterShape_;
  using EpilogueTile = EpilogueTile_;
  using EpilogueSchedule = EpilogueSchedule_;
  using KernelSchedule = KernelSchedule_;
  using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBias<
      ElementD, ElementAccumulator, ElementD, ElementC, ElementAccumulator,
      AlignmentCD>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape, EpilogueTile,
          ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag,
          AlignmentCD, ElementD, LayoutDTag, AlignmentCD, EpilogueSchedule,
          FusionOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using PrefillSpec =
    GemmSpec<Shape<_256, _192, _256>, Shape<_2, _1, _1>,
             cutlass::epilogue::collective::EpilogueTileAuto, 8,
             cutlass::epilogue::TmaWarpSpecialized2Sm,
             cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>;

using BiasPrefillSpec =
    BiasGemmSpec<Shape<_256, _192, _256>, Shape<_2, _1, _1>,
                 cutlass::epilogue::collective::EpilogueTileAuto, 8,
                 cutlass::epilogue::TmaWarpSpecialized2Sm,
                 cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>;

using DecodeSpec =
    GemmSpec<Shape<_256, _256, _256>, Shape<_2, _1, _1>, Shape<_128, _64>, 8,
             cutlass::epilogue::TmaWarpSpecialized2Sm,
             cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>;

using DecodeWideSpec =
    GemmSpec<Shape<_256, _256, _256>, Shape<_4, _1, _1>, Shape<_128, _64>, 4,
             cutlass::epilogue::NoSmemWarpSpecialized2Sm,
             cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>;

template <typename Spec>
typename Spec::Gemm::Arguments
make_arguments(at::Tensor &output, at::Tensor const &packed_input,
               at::Tensor const &packed_weight, at::Tensor const &input_scale,
               at::Tensor const &weight_scale, int64_t m, int64_t n, int64_t k,
               const at::BFloat16 *bias) {
  using Gemm = typename Spec::Gemm;
  using CollectiveMainloop = typename Spec::CollectiveMainloop;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ArrayElementA = typename CollectiveMainloop::ElementA;
  using ArrayElementB = typename CollectiveMainloop::ElementB;
  using Sm100BlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int mi = static_cast<int>(m);
  int ni = static_cast<int>(n);
  int ki = static_cast<int>(k);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {mi, ki, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {ni, ki, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {mi, ni, 1});
  auto layout_SFA =
      Sm100BlkScaledConfig::tile_atom_to_shape_SFA(make_shape(mi, ni, ki, 1));
  auto layout_SFB =
      Sm100BlkScaledConfig::tile_atom_to_shape_SFB(make_shape(mi, ni, ki, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {mi, ni, ki, 1},
      {
          reinterpret_cast<ArrayElementA const *>(
              packed_input.data_ptr<uint8_t>()),
          stride_A,
          reinterpret_cast<ArrayElementB const *>(
              packed_weight.data_ptr<uint8_t>()),
          stride_B,
          reinterpret_cast<ElementSFA const *>(input_scale.data_ptr<uint8_t>()),
          layout_SFA,
          reinterpret_cast<ElementSFB const *>(
              weight_scale.data_ptr<uint8_t>()),
          layout_SFB,
      },
      {
          {},
          reinterpret_cast<ElementD const *>(output.data_ptr<at::BFloat16>()),
          stride_D,
          reinterpret_cast<ElementD *>(output.data_ptr<at::BFloat16>()),
          stride_D,
      }};
  arguments.epilogue.thread.alpha = 1.0f;
  if constexpr (Spec::kFusesBias) {
    arguments.epilogue.thread.bias_ptr =
        reinterpret_cast<cutlass::bfloat16_t const *>(bias);
  }
  return arguments;
}

template <typename Spec>
void run(at::Tensor &output, at::Tensor const &packed_input,
         at::Tensor const &packed_weight, at::Tensor const &input_scale,
         at::Tensor const &weight_scale, int64_t m, int64_t n, int64_t k,
         const at::BFloat16 *bias, cudaStream_t stream) {
  using Gemm = typename Spec::Gemm;
  Gemm gemm;
  auto arguments =
      make_arguments<Spec>(output, packed_input, packed_weight, input_scale,
                           weight_scale, m, n, k, bias);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                             packed_input.options().dtype(at::kByte));
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP4 BF16-output GEMM cannot implement this shape");
  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP4 BF16-output GEMM initialization failed");
  status = gemm.run(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP4 BF16-output GEMM run failed");
}

void run_selected(at::Tensor &output, at::Tensor const &packed_input,
                  at::Tensor const &packed_weight,
                  at::Tensor const &input_scale, at::Tensor const &weight_scale,
                  int64_t m, int64_t n, int64_t k, const at::BFloat16 *bias,
                  cudaStream_t stream) {
  if (bias != nullptr && m > 1) {
    run<BiasPrefillSpec>(output, packed_input, packed_weight, input_scale,
                         weight_scale, m, n, k, bias, stream);
  } else if (m == 1) {
    int64_t policy = g_decode_policy.load(std::memory_order_relaxed);
    bool fused_qkv = n > k && n < 8192;
    bool vocabulary_projection = n >= 65536;
    if (policy == kDecodePolicyLegacy) {
      if (n > k) {
        run<DecodeWideSpec>(output, packed_input, packed_weight, input_scale,
                            weight_scale, m, n, k, nullptr, stream);
      } else {
        run<DecodeSpec>(output, packed_input, packed_weight, input_scale,
                        weight_scale, m, n, k, nullptr, stream);
      }
    } else if (policy == kDecodePolicyLowClock && n > k) {
      run<DecodeSpec>(output, packed_input, packed_weight, input_scale,
                      weight_scale, m, n, k, nullptr, stream);
    } else if (fused_qkv) {
      run<DecodeSpec>(output, packed_input, packed_weight, input_scale,
                      weight_scale, m, n, k, nullptr, stream);
    } else if (policy == kDecodePolicyBalanced && vocabulary_projection) {
      run<DecodeWideSpec>(output, packed_input, packed_weight, input_scale,
                          weight_scale, m, n, k, nullptr, stream);
    } else {
      run<PrefillSpec>(output, packed_input, packed_weight, input_scale,
                       weight_scale, m, n, k, nullptr, stream);
    }
  } else {
    run<PrefillSpec>(output, packed_input, packed_weight, input_scale,
                     weight_scale, m, n, k, nullptr, stream);
  }
}

void run_tactic(at::Tensor &output, at::Tensor const &packed_input,
                at::Tensor const &packed_weight, at::Tensor const &input_scale,
                at::Tensor const &weight_scale, int64_t m, int64_t n, int64_t k,
                const at::BFloat16 *bias, int64_t tactic, cudaStream_t stream) {
  if (tactic == 1) {
    run<DecodeSpec>(output, packed_input, packed_weight, input_scale,
                    weight_scale, m, n, k, bias, stream);
  } else if (tactic == 2) {
    run<DecodeWideSpec>(output, packed_input, packed_weight, input_scale,
                        weight_scale, m, n, k, bias, stream);
  } else if (tactic == 3) {
    run<PrefillSpec>(output, packed_input, packed_weight, input_scale,
                     weight_scale, m, n, k, bias, stream);
  } else {
    TORCH_CHECK(false, "unknown CUTLASS FP4 GEMM tactic ", tactic);
  }
}

} // namespace gemm_bf16

void check_weight_metadata(at::Tensor const &metadata, int64_t &rows,
                           int64_t &cols) {
  at::Tensor metadata_cpu =
      metadata.device().is_cpu() ? metadata : metadata.cpu();
  TORCH_CHECK(metadata_cpu.scalar_type() == at::kLong,
              "metadata must be int64");
  TORCH_CHECK(metadata_cpu.numel() >= 2, "metadata must contain rows and cols");
  auto *data = metadata_cpu.data_ptr<int64_t>();
  rows = data[0];
  cols = data[1];
}

bool supports_device(int64_t cc) { return cc >= 100; }

void set_decode_policy(int64_t policy) {
  TORCH_CHECK(policy >= kDecodePolicyBalanced && policy <= kDecodePolicyLegacy,
              "FP4 decode policy must be 0 (balanced), 1 (low_clock), "
              "2 (mid_clock), or 3 (legacy), got ",
              policy);
  g_decode_policy.store(policy, std::memory_order_relaxed);
}

int64_t get_decode_policy() {
  return g_decode_policy.load(std::memory_order_relaxed);
}

std::vector<at::Tensor> pack_linear_weight(at::Tensor weight, std::string,
                                           std::string) {
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(weight.dim() == 2,
              "weight must be rank-2 [out_features, in_features]");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16 ||
                  weight.scalar_type() == at::kHalf,
              "weight must be bf16 or fp16");

  int64_t rows = weight.size(0);
  int64_t cols = weight.size(1);
  TORCH_CHECK(cols % kElementsPerAccess == 0,
              "CUTLASS FP4 GEMV requires in_features divisible by ",
              kElementsPerAccess, ", got ", cols);
  TORCH_CHECK(
      cols >= 1024,
      "CUTLASS 91_fp4_gemv prologue path requires in_features >= 1024, got ",
      cols);

  c10::cuda::CUDAGuard guard(weight.device());
  auto byte_options = weight.options().dtype(at::kByte);
  int64_t packed_cols = ceil_div_int64(cols, 2);
  int64_t sf_per_row = ceil_div_int64(cols, kSfVecSize);
  int64_t blocks_k = sf_blocks_k(cols);
  int64_t scale_elems = sf_blocks_m(rows) * blocks_k * 512;

  at::Tensor packed = at::empty({rows, packed_cols}, byte_options);
  at::Tensor scales = at::zeros({scale_elems}, byte_options);
  at::Tensor metadata =
      at::empty({2}, weight.options().device(at::kCPU).dtype(at::kLong));
  auto *metadata_ptr = metadata.data_ptr<int64_t>();
  metadata_ptr[0] = rows;
  metadata_ptr[1] = cols;

  int threads = 256;
  int64_t total_blocks = rows * sf_per_row;
  int blocks = static_cast<int>(ceil_div_int64(total_blocks, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(weight.get_device());

  if (weight.scalar_type() == at::kBFloat16) {
    pack_matrix_fp4_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
        weight.data_ptr<at::BFloat16>(), packed.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(), rows, cols, packed_cols, blocks_k,
        sf_per_row);
  } else {
    pack_matrix_fp4_kernel<at::Half><<<blocks, threads, 0, stream>>>(
        weight.data_ptr<at::Half>(), packed.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(), rows, cols, packed_cols, blocks_k,
        sf_per_row);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales, metadata};
}

std::vector<at::Tensor> pack_linear_weight_gemm_bf16(at::Tensor weight,
                                                     std::string, std::string) {
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(weight.dim() == 2,
              "weight must be rank-2 [out_features, in_features]");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16 ||
                  weight.scalar_type() == at::kHalf,
              "weight must be bf16 or fp16");

  int64_t rows = weight.size(0);
  int64_t cols = weight.size(1);
  TORCH_CHECK(rows % 2 == 0, "CUTLASS FP4 GEMM weight rows must be even");
  TORCH_CHECK(rows % 32 == 0,
              "CUTLASS FP4 GEMM requires out_features divisible by 32, got ",
              rows);
  TORCH_CHECK(cols % kElementsPerAccess == 0,
              "CUTLASS FP4 GEMM requires in_features divisible by ",
              kElementsPerAccess, ", got ", cols);

  c10::cuda::CUDAGuard guard(weight.device());
  auto byte_options = weight.options().dtype(at::kByte);
  int64_t packed_cols = cols / 2;
  int64_t sf_per_row = ceil_div_int64(cols, kSfVecSize);
  int64_t blocks_k = sf_blocks_k(cols);
  int64_t rounded_rows = round_up_int64(rows, 128);
  int64_t rounded_scale_cols = round_up_int64(sf_per_row, 4);

  at::Tensor packed = at::empty({rows, packed_cols}, byte_options);
  at::Tensor scales =
      at::zeros({rounded_rows, rounded_scale_cols}, byte_options);
  at::Tensor metadata =
      at::empty({2}, weight.options().device(at::kCPU).dtype(at::kLong));
  auto *metadata_ptr = metadata.data_ptr<int64_t>();
  metadata_ptr[0] = rows;
  metadata_ptr[1] = cols;

  int threads = 256;
  int64_t total_blocks = rows * sf_per_row;
  int blocks = static_cast<int>(ceil_div_int64(total_blocks, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(weight.get_device());

  if (weight.scalar_type() == at::kBFloat16) {
    pack_matrix_fp4_ue_rowmajor_kernel<at::BFloat16>
        <<<blocks, threads, 0, stream>>>(weight.data_ptr<at::BFloat16>(),
                                         packed.data_ptr<uint8_t>(),
                                         scales.data_ptr<uint8_t>(), rows, cols,
                                         packed_cols, blocks_k, sf_per_row);
  } else {
    pack_matrix_fp4_ue_rowmajor_kernel<at::Half>
        <<<blocks, threads, 0, stream>>>(weight.data_ptr<at::Half>(),
                                         packed.data_ptr<uint8_t>(),
                                         scales.data_ptr<uint8_t>(), rows, cols,
                                         packed_cols, blocks_k, sf_per_row);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales, metadata};
}

at::Tensor linear_forward(at::Tensor input, at::Tensor packed_weight,
                          at::Tensor weight_scale, at::Tensor metadata,
                          std::optional<at::Tensor> bias, std::string,
                          std::string) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "first CUTLASS FP4 Linear path supports bf16 input only");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(packed_weight.is_cuda() && weight_scale.is_cuda(),
              "packed weight and scale must be CUDA");
  TORCH_CHECK(packed_weight.scalar_type() == at::kByte &&
                  weight_scale.scalar_type() == at::kByte,
              "packed weight and scale must be uint8");

  int64_t rows = 0;
  int64_t cols = 0;
  check_weight_metadata(metadata, rows, cols);
  TORCH_CHECK(
      cols >= 1024,
      "CUTLASS 91_fp4_gemv prologue path requires in_features >= 1024, got ",
      cols);
  TORCH_CHECK(input.size(-1) == cols,
              "input last dimension does not match packed weight");

  int64_t leading = input.numel() / cols;
  TORCH_CHECK(leading == 1,
              "first CUTLASS FP4 Linear path supports decode only (one input "
              "row), got ",
              leading, " rows");

  c10::cuda::CUDAGuard guard(input.device());
  at::Tensor input_1d = input.reshape({cols});
  auto byte_options = input.options().dtype(at::kByte);
  at::Tensor packed_input = at::empty({ceil_div_int64(cols, 2)}, byte_options);
  int64_t sf_per_row = ceil_div_int64(cols, kSfVecSize);
  at::Tensor input_scale = at::zeros({sf_blocks_k(cols) * 512}, byte_options);

  int threads = 256;
  int blocks = static_cast<int>(ceil_div_int64(sf_per_row, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  pack_vector_fp4_kernel<at::BFloat16><<<blocks, threads, 0, stream>>>(
      input_1d.data_ptr<at::BFloat16>(), packed_input.data_ptr<uint8_t>(),
      input_scale.data_ptr<uint8_t>(), cols, sf_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  TORCH_CHECK(rows % kSfVecSize == 0,
              "official CUTLASS FP4 output epilogue path currently requires "
              "out_features divisible by ",
              kSfVecSize, ", got ", rows);

  at::Tensor output = at::empty({rows}, input.options());
  at::Tensor output_packed = at::empty({ceil_div_int64(rows, 2)}, byte_options);
  at::Tensor output_scale = at::zeros({sfd_elems(rows)}, byte_options);
  at::Tensor c_tensor = at::zeros({ceil_div_int64(rows, 2)}, byte_options);
  at::Tensor bias_tensor;
  const at::BFloat16 *bias_ptr = nullptr;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.value().device() == packed_input.device(),
                "bias and packed input must be on the same CUDA device");
    TORCH_CHECK(bias.value().scalar_type() == at::kBFloat16,
                "bias must be bf16");
    TORCH_CHECK(bias.value().numel() == rows,
                "bias size must match out_features");
    bias_tensor = bias.value().contiguous();
    bias_ptr = bias_tensor.data_ptr<at::BFloat16>();
  }

  Gemv gemv;
  TensorRefA ref_A(
      reinterpret_cast<ElementA *>(packed_weight.data_ptr<uint8_t>()),
      LayoutA(cols));
  constexpr float kOutputScaleTarget = 1.0f;
  typename Gemv::Arguments args{
      cutlass::MatrixCoord{static_cast<int>(rows), static_cast<int>(cols)},
      1,
      typename EpilogueOp::Params{
          typename EpilogueOp::TensorRefD(
              reinterpret_cast<ElementD *>(output_packed.data_ptr<uint8_t>()),
              LayoutD(rows)),
          reinterpret_cast<ElementSFD *>(output_scale.data_ptr<uint8_t>()),
          1.0f,
          0.0f,
          kOutputScaleTarget,
          sfd_elems(rows),
          rows,
      },
      ref_A,
      reinterpret_cast<ElementB const *>(packed_input.data_ptr<uint8_t>()),
      reinterpret_cast<ElementC const *>(c_tensor.data_ptr<uint8_t>()),
      reinterpret_cast<ElementC *>(output_packed.data_ptr<uint8_t>()),
      reinterpret_cast<ElementSFA const *>(weight_scale.data_ptr<uint8_t>()),
      reinterpret_cast<ElementSFB const *>(input_scale.data_ptr<uint8_t>()),
      cols,
      rows * cols,
      cols,
      rows,
      rows,
      sf_blocks_m(rows) * sf_blocks_k(cols) * 512,
      sf_blocks_k(cols) * 512,
      sfd_elems(rows),
  };

  auto status = gemv.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP4 GEMV cannot implement this shape");
  status = gemv.initialize(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP4 GEMV initialization failed");
  status = gemv.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP4 GEMV run failed");
  int dequant_threads = 256;
  int dequant_blocks = static_cast<int>(ceil_div_int64(rows, dequant_threads));
  dequantize_output_fp4_kernel<at::BFloat16>
      <<<dequant_blocks, dequant_threads, 0, stream>>>(
          output_packed.data_ptr<uint8_t>(), output_scale.data_ptr<uint8_t>(),
          bias_ptr, output.data_ptr<at::BFloat16>(), rows, kOutputScaleTarget);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end());
  out_shape.back() = rows;
  return output.reshape(out_shape);
}

std::vector<at::Tensor> pack_activation_gemm_bf16(at::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "CUTLASS FP4 activation packing supports bf16 input only");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  int64_t cols = input.size(-1);
  TORCH_CHECK(cols % kElementsPerAccess == 0,
              "CUTLASS FP4 GEMM requires in_features divisible by ",
              kElementsPerAccess, ", got ", cols);
  int64_t leading = input.numel() / cols;
  TORCH_CHECK(leading > 0, "input must contain at least one row");

  c10::cuda::CUDAGuard guard(input.device());
  auto byte_options = input.options().dtype(at::kByte);
  int64_t packed_cols = cols / 2;
  int64_t sf_per_row = ceil_div_int64(cols, kSfVecSize);
  int64_t rounded_m = round_up_int64(leading, 128);
  int64_t rounded_scale_cols = round_up_int64(sf_per_row, 4);
  at::Tensor input_2d = input.reshape({leading, cols});
  at::Tensor packed_input = at::empty({leading, packed_cols}, byte_options);
  at::Tensor input_scale =
      at::empty({rounded_m, rounded_scale_cols}, byte_options);

  int threads = 256;
  int64_t total_blocks = leading * sf_per_row;
  int blocks = static_cast<int>(ceil_div_int64(total_blocks, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  pack_matrix_fp4_ue_rowmajor_kernel<at::BFloat16>
      <<<blocks, threads, 0, stream>>>(
          input_2d.data_ptr<at::BFloat16>(), packed_input.data_ptr<uint8_t>(),
          input_scale.data_ptr<uint8_t>(), leading, cols, packed_cols,
          sf_blocks_k(cols), sf_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed_input, input_scale};
}

at::Tensor linear_forward_packed_gemm_bf16_impl(
    at::Tensor packed_input, at::Tensor input_scale, at::Tensor packed_weight,
    at::Tensor weight_scale, at::Tensor metadata,
    std::optional<at::Tensor> bias, std::string, std::string, int64_t tactic) {
  TORCH_CHECK(packed_input.is_cuda() && input_scale.is_cuda(),
              "packed input and scale must be CUDA");
  TORCH_CHECK(packed_weight.is_cuda() && weight_scale.is_cuda(),
              "packed weight and scale must be CUDA");
  TORCH_CHECK(packed_input.scalar_type() == at::kByte &&
                  input_scale.scalar_type() == at::kByte &&
                  packed_weight.scalar_type() == at::kByte &&
                  weight_scale.scalar_type() == at::kByte,
              "packed inputs, weights, and scales must be uint8");
  TORCH_CHECK(packed_input.is_contiguous() && input_scale.is_contiguous(),
              "packed input and scale must be contiguous");
  TORCH_CHECK(packed_input.device() == input_scale.device() &&
                  packed_input.device() == packed_weight.device() &&
                  packed_input.device() == weight_scale.device(),
              "packed input, weights, and scales must share a CUDA device");
  TORCH_CHECK(packed_input.dim() == 2,
              "packed input must be rank-2 [rows, packed_k]");

  int64_t rows = 0;
  int64_t cols = 0;
  check_weight_metadata(metadata, rows, cols);
  int64_t leading = packed_input.size(0);
  TORCH_CHECK(leading > 0, "packed input must contain at least one row");
  TORCH_CHECK(packed_input.size(1) == cols / 2,
              "packed input width does not match weight in_features");
  TORCH_CHECK(rows % 32 == 0 && cols % kElementsPerAccess == 0,
              "CUTLASS FP4 GEMM dimensions must be divisible by 32");
  int64_t required_scale_elems =
      round_up_int64(leading, 128) *
      round_up_int64(ceil_div_int64(cols, kSfVecSize), 4);
  TORCH_CHECK(input_scale.numel() >= required_scale_elems,
              "packed input scale buffer is too small");

  c10::cuda::CUDAGuard guard(packed_input.device());
  at::Tensor output =
      at::empty({leading, rows}, packed_input.options().dtype(at::kBFloat16));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(packed_input.get_device());

  at::Tensor bias_tensor;
  const at::BFloat16 *bias_ptr = nullptr;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.value().device() == packed_input.device(),
                "bias and packed input must be on the same CUDA device");
    TORCH_CHECK(bias.value().scalar_type() == at::kBFloat16,
                "bias must be bf16");
    TORCH_CHECK(bias.value().numel() == rows,
                "bias size must match out_features");
    bias_tensor = bias.value().contiguous();
    bias_ptr = bias_tensor.data_ptr<at::BFloat16>();
  }

  if (tactic == 0) {
    gemm_bf16::run_selected(output, packed_input, packed_weight, input_scale,
                            weight_scale, leading, rows, cols, bias_ptr,
                            stream);
  } else {
    gemm_bf16::run_tactic(output, packed_input, packed_weight, input_scale,
                          weight_scale, leading, rows, cols, bias_ptr, tactic,
                          stream);
  }

  if (bias_ptr != nullptr && leading == 1) {
    int bias_threads = 256;
    int64_t bias_elements = leading * rows;
    int bias_blocks =
        static_cast<int>(ceil_div_int64(bias_elements, bias_threads));
    add_bias_bf16_kernel<<<bias_blocks, bias_threads, 0, stream>>>(
        bias_ptr, output.data_ptr<at::BFloat16>(), leading, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return output;
}

at::Tensor linear_forward_packed_gemm_bf16(
    at::Tensor packed_input, at::Tensor input_scale, at::Tensor packed_weight,
    at::Tensor weight_scale, at::Tensor metadata,
    std::optional<at::Tensor> bias, std::string role, std::string target_name) {
  return linear_forward_packed_gemm_bf16_impl(
      packed_input, input_scale, packed_weight, weight_scale, metadata, bias,
      role, target_name, 0);
}

at::Tensor linear_forward_packed_gemm_bf16_tactic(
    at::Tensor packed_input, at::Tensor input_scale, at::Tensor packed_weight,
    at::Tensor weight_scale, at::Tensor metadata,
    std::optional<at::Tensor> bias, std::string role, std::string target_name,
    int64_t tactic) {
  return linear_forward_packed_gemm_bf16_impl(
      packed_input, input_scale, packed_weight, weight_scale, metadata, bias,
      role, target_name, tactic);
}

at::Tensor linear_forward_packed_greedy_gemm_bf16(
    at::Tensor packed_input, at::Tensor input_scale, at::Tensor packed_weight,
    at::Tensor weight_scale, at::Tensor metadata,
    std::optional<at::Tensor> bias, at::Tensor seen_tokens,
    at::Tensor eos_tokens, at::Tensor generated_count,
    double repetition_penalty, int64_t min_new_tokens, std::string role,
    std::string target_name) {
  TORCH_CHECK(packed_input.dim() == 2 && packed_input.size(0) == 1,
              "fused greedy lm_head requires one input row");
  int64_t vocab_size = 0;
  int64_t in_features = 0;
  check_weight_metadata(metadata, vocab_size, in_features);
  TORCH_CHECK(vocab_size > 0 && vocab_size <= INT_MAX,
              "vocabulary size is out of range");
  TORCH_CHECK(seen_tokens.is_cuda() && eos_tokens.is_cuda() &&
                  generated_count.is_cuda(),
              "sampler state must be CUDA");
  TORCH_CHECK(seen_tokens.device() == packed_input.device() &&
                  eos_tokens.device() == packed_input.device() &&
                  generated_count.device() == packed_input.device(),
              "sampler state and packed input must share a CUDA device");
  TORCH_CHECK(seen_tokens.scalar_type() == at::kByte &&
                  eos_tokens.scalar_type() == at::kByte,
              "seen/eos masks must be uint8");
  TORCH_CHECK(generated_count.scalar_type() == at::kInt &&
                  generated_count.numel() == 1,
              "generated_count must be a one-element int32 tensor");
  TORCH_CHECK(seen_tokens.is_contiguous() && eos_tokens.is_contiguous() &&
                  generated_count.is_contiguous(),
              "sampler state must be contiguous");
  TORCH_CHECK(seen_tokens.numel() == vocab_size &&
                  eos_tokens.numel() == vocab_size,
              "sampler masks must match vocabulary size");
  TORCH_CHECK(repetition_penalty > 0.0, "repetition_penalty must be positive");
  TORCH_CHECK(min_new_tokens >= 0, "min_new_tokens must be non-negative");

  at::Tensor logits = linear_forward_packed_gemm_bf16(
      packed_input, input_scale, packed_weight, weight_scale, metadata, bias,
      role, target_name);
  at::Tensor output = at::empty({1}, packed_input.options().dtype(at::kLong));
  constexpr int kThreads = 256;
  int partial_count = static_cast<int>(
      std::min<int64_t>(256, ceil_div_int64(vocab_size, kThreads)));
  at::Tensor partial_scores =
      at::empty({partial_count}, packed_input.options().dtype(at::kFloat));
  at::Tensor partial_indices =
      at::empty({partial_count}, packed_input.options().dtype(at::kInt));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(packed_input.get_device());
  processed_argmax_stage1_bf16_kernel<<<partial_count, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()),
      seen_tokens.data_ptr<uint8_t>(), eos_tokens.data_ptr<uint8_t>(),
      generated_count.data_ptr<int32_t>(), partial_scores.data_ptr<float>(),
      partial_indices.data_ptr<int32_t>(), static_cast<int>(vocab_size),
      static_cast<float>(repetition_penalty), static_cast<int>(min_new_tokens));
  processed_argmax_finalize_kernel<<<1, kThreads, 0, stream>>>(
      partial_scores.data_ptr<float>(), partial_indices.data_ptr<int32_t>(),
      seen_tokens.data_ptr<uint8_t>(), generated_count.data_ptr<int32_t>(),
      output.data_ptr<int64_t>(), partial_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<at::Tensor> linear_forward_packed_topk_gemm_bf16(
    at::Tensor packed_input, at::Tensor input_scale, at::Tensor packed_weight,
    at::Tensor weight_scale, at::Tensor metadata,
    std::optional<at::Tensor> bias, at::Tensor seen_tokens,
    at::Tensor eos_tokens, at::Tensor generated_count,
    double repetition_penalty, int64_t min_new_tokens, int64_t top_k,
    std::string role, std::string target_name) {
  TORCH_CHECK(packed_input.dim() == 2 && packed_input.size(0) == 1,
              "fused top-k lm_head requires one input row");
  int64_t vocab_size = 0;
  int64_t in_features = 0;
  check_weight_metadata(metadata, vocab_size, in_features);
  TORCH_CHECK(top_k > 0 && top_k <= vocab_size,
              "top_k must be in [1, vocab_size]");
  TORCH_CHECK(seen_tokens.is_cuda() && eos_tokens.is_cuda() &&
                  generated_count.is_cuda(),
              "sampler state must be CUDA");
  TORCH_CHECK(seen_tokens.device() == packed_input.device() &&
                  eos_tokens.device() == packed_input.device() &&
                  generated_count.device() == packed_input.device(),
              "sampler state and packed input must share a CUDA device");
  TORCH_CHECK(seen_tokens.scalar_type() == at::kByte &&
                  eos_tokens.scalar_type() == at::kByte &&
                  generated_count.scalar_type() == at::kInt,
              "sampler state has an invalid dtype");
  TORCH_CHECK(seen_tokens.numel() == vocab_size &&
                  eos_tokens.numel() == vocab_size &&
                  generated_count.numel() == 1,
              "sampler state has an invalid size");

  at::Tensor logits = linear_forward_packed_gemm_bf16(
      packed_input, input_scale, packed_weight, weight_scale, metadata, bias,
      role, target_name);
  at::Tensor processed =
      at::empty({1, vocab_size}, packed_input.options().dtype(at::kFloat));
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(packed_input.get_device());
  constexpr int kThreads = 256;
  int blocks = static_cast<int>(ceil_div_int64(vocab_size, kThreads));
  process_logits_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(logits.data_ptr<at::BFloat16>()),
      seen_tokens.data_ptr<uint8_t>(), eos_tokens.data_ptr<uint8_t>(),
      generated_count.data_ptr<int32_t>(), processed.data_ptr<float>(),
      static_cast<int>(vocab_size), static_cast<float>(repetition_penalty),
      static_cast<int>(min_new_tokens));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto result = at::topk(processed, top_k, -1, true, true);
  return {std::get<0>(result), std::get<1>(result)};
}

at::Tensor linear_forward_gemm_bf16(at::Tensor input, at::Tensor packed_weight,
                                    at::Tensor weight_scale,
                                    at::Tensor metadata,
                                    std::optional<at::Tensor> bias,
                                    std::string role, std::string target_name) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16,
              "CUTLASS FP4 BF16-output GEMM path supports bf16 input only");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  int64_t rows = 0;
  int64_t cols = 0;
  check_weight_metadata(metadata, rows, cols);
  TORCH_CHECK(input.size(-1) == cols,
              "input last dimension does not match packed weight");

  auto packed = pack_activation_gemm_bf16(input);
  at::Tensor output = linear_forward_packed_gemm_bf16(
      packed[0], packed[1], packed_weight, weight_scale, metadata, bias, role,
      target_name);

  std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end());
  out_shape.back() = rows;
  return output.reshape(out_shape);
}

at::Tensor silu_mul_bf16(at::Tensor gate_up, int64_t intermediate_size,
                         bool interleaved) {
  TORCH_CHECK(gate_up.is_cuda(), "gate_up must be CUDA");
  TORCH_CHECK(gate_up.scalar_type() == at::kBFloat16, "gate_up must be bf16");
  TORCH_CHECK(gate_up.is_contiguous(), "gate_up must be contiguous");
  TORCH_CHECK(gate_up.dim() >= 1, "gate_up must have at least one dimension");
  TORCH_CHECK(gate_up.numel() > 0, "gate_up must not be empty");
  TORCH_CHECK(intermediate_size > 0, "intermediate_size must be positive");
  TORCH_CHECK(intermediate_size % 2 == 0,
              "intermediate_size must be even for BF16x2 SwiGLU");
  TORCH_CHECK(gate_up.size(-1) == 2 * intermediate_size,
              "gate_up last dimension must equal 2 * intermediate_size");
  TORCH_CHECK(!interleaved || intermediate_size % 64 == 0,
              "interleaved gate/up layout requires intermediate_size "
              "divisible by 64");

  c10::cuda::CUDAGuard guard(gate_up.device());
  int64_t leading = gate_up.numel() / (2 * intermediate_size);
  std::vector<int64_t> out_shape(gate_up.sizes().begin(),
                                 gate_up.sizes().end());
  out_shape.back() = intermediate_size;
  at::Tensor output = at::empty(out_shape, gate_up.options());
  int threads = 256;
  int64_t pairs_per_row = intermediate_size / 2;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());
  if (leading == 1) {
    int blocks = static_cast<int>(ceil_div_int64(intermediate_size, threads));
    if (interleaved) {
      silu_mul_bf16_scalar_kernel<true><<<blocks, threads, 0, stream>>>(
          gate_up.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(),
          intermediate_size);
    } else {
      silu_mul_bf16_scalar_kernel<false><<<blocks, threads, 0, stream>>>(
          gate_up.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(),
          intermediate_size);
    }
  } else {
    int blocks =
        static_cast<int>(ceil_div_int64(leading * pairs_per_row, threads));
    if (interleaved) {
      silu_mul_bf16_kernel<true><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat162 *>(
              gate_up.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
          leading, pairs_per_row);
    } else {
      silu_mul_bf16_kernel<false><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat162 *>(
              gate_up.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat162 *>(output.data_ptr<at::BFloat16>()),
          leading, pairs_per_row);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor
linear_forward_swiglu_gemm_bf16(at::Tensor gate_up, int64_t intermediate_size,
                                at::Tensor packed_weight,
                                at::Tensor weight_scale, at::Tensor metadata,
                                std::optional<at::Tensor> bias, std::string,
                                std::string, bool interleaved) {
  TORCH_CHECK(gate_up.is_cuda(), "gate_up must be CUDA");
  TORCH_CHECK(gate_up.scalar_type() == at::kBFloat16, "gate_up must be bf16");
  TORCH_CHECK(gate_up.is_contiguous(), "gate_up must be contiguous");
  TORCH_CHECK(gate_up.dim() >= 1, "gate_up must have at least one dimension");
  TORCH_CHECK(gate_up.numel() > 0, "gate_up must not be empty");
  TORCH_CHECK(packed_weight.is_cuda() && weight_scale.is_cuda(),
              "packed weight and scale must be CUDA");
  TORCH_CHECK(packed_weight.device() == gate_up.device() &&
                  weight_scale.device() == gate_up.device(),
              "gate_up, packed weight, and weight scale must be on the same "
              "CUDA device");
  TORCH_CHECK(packed_weight.scalar_type() == at::kByte &&
                  weight_scale.scalar_type() == at::kByte,
              "packed weight and scale must be uint8");

  int64_t rows = 0;
  int64_t cols = 0;
  check_weight_metadata(metadata, rows, cols);
  TORCH_CHECK(intermediate_size == cols,
              "intermediate_size must match down projection in_features");
  TORCH_CHECK(gate_up.size(-1) == 2 * cols,
              "gate_up last dimension must equal 2 * down in_features");
  TORCH_CHECK(!interleaved || cols % 64 == 0,
              "interleaved gate/up layout requires down in_features "
              "divisible by 64");
  TORCH_CHECK(cols % kElementsPerAccess == 0,
              "CUTLASS FP4 GEMM requires in_features divisible by ",
              kElementsPerAccess, ", got ", cols);
  TORCH_CHECK(rows % 32 == 0,
              "CUTLASS FP4 GEMM requires out_features divisible by 32, got ",
              rows);

  int64_t leading = gate_up.numel() / (2 * cols);
  c10::cuda::CUDAGuard guard(gate_up.device());
  auto byte_options = gate_up.options().dtype(at::kByte);
  int64_t packed_cols = cols / 2;
  int64_t sf_per_row = cols / kSfVecSize;
  int64_t rounded_m = round_up_int64(leading, 128);
  int64_t rounded_scale_cols = round_up_int64(sf_per_row, 4);
  at::Tensor packed_input = at::empty({leading, packed_cols}, byte_options);
  at::Tensor input_scale =
      at::empty({rounded_m, rounded_scale_cols}, byte_options);
  at::Tensor output = at::empty({leading, rows}, gate_up.options());

  constexpr int threads = 256;
  int64_t total_groups = leading * sf_per_row;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());
  if (leading == 1) {
    constexpr int groups_per_block = (threads / 32) * 4;
    int blocks =
        static_cast<int>(ceil_div_int64(total_groups, groups_per_block));
    if (interleaved) {
      silu_mul_pack_fp4_ue_rowmajor_kernel<true>
          <<<blocks, threads, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat162 *>(
                  gate_up.data_ptr<at::BFloat16>()),
              packed_input.data_ptr<uint8_t>(), input_scale.data_ptr<uint8_t>(),
              leading, cols, packed_cols, sf_blocks_k(cols), sf_per_row);
    } else {
      silu_mul_pack_fp4_ue_rowmajor_kernel<false>
          <<<blocks, threads, 0, stream>>>(
              reinterpret_cast<const __nv_bfloat162 *>(
                  gate_up.data_ptr<at::BFloat16>()),
              packed_input.data_ptr<uint8_t>(), input_scale.data_ptr<uint8_t>(),
              leading, cols, packed_cols, sf_blocks_k(cols), sf_per_row);
    }
  } else {
    int blocks = static_cast<int>(ceil_div_int64(total_groups, threads));
    if (interleaved) {
      silu_mul_pack_fp4_ue_rowmajor_vector_kernel<true>
          <<<blocks, threads, 0, stream>>>(
              gate_up.data_ptr<at::BFloat16>(),
              packed_input.data_ptr<uint8_t>(), input_scale.data_ptr<uint8_t>(),
              leading, cols, packed_cols, sf_blocks_k(cols), sf_per_row);
    } else {
      silu_mul_pack_fp4_ue_rowmajor_vector_kernel<false>
          <<<blocks, threads, 0, stream>>>(
              gate_up.data_ptr<at::BFloat16>(),
              packed_input.data_ptr<uint8_t>(), input_scale.data_ptr<uint8_t>(),
              leading, cols, packed_cols, sf_blocks_k(cols), sf_per_row);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  at::Tensor bias_tensor;
  const at::BFloat16 *bias_ptr = nullptr;
  if (bias.has_value() && bias.value().defined()) {
    TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA");
    TORCH_CHECK(bias.value().device() == gate_up.device(),
                "bias and gate_up must be on the same CUDA device");
    TORCH_CHECK(bias.value().scalar_type() == at::kBFloat16,
                "bias must be bf16");
    TORCH_CHECK(bias.value().numel() == rows,
                "bias size must match out_features");
    bias_tensor = bias.value().contiguous();
    bias_ptr = bias_tensor.data_ptr<at::BFloat16>();
  }

  gemm_bf16::run_selected(output, packed_input, packed_weight, input_scale,
                          weight_scale, leading, rows, cols, bias_ptr, stream);

  if (bias_ptr != nullptr && leading == 1) {
    int64_t bias_elements = leading * rows;
    int bias_blocks = static_cast<int>(ceil_div_int64(bias_elements, threads));
    add_bias_bf16_kernel<<<bias_blocks, threads, 0, stream>>>(
        bias_ptr, output.data_ptr<at::BFloat16>(), leading, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  std::vector<int64_t> out_shape(gate_up.sizes().begin(),
                                 gate_up.sizes().end());
  out_shape.back() = rows;
  return output.reshape(out_shape);
}

void check_bf16_cuda_tensor(const at::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bf16");
}

void check_bhsd_tensor(const at::Tensor &tensor, const char *name) {
  check_bf16_cuda_tensor(tensor, name);
  TORCH_CHECK(tensor.dim() == 4, name,
              " must have shape [batch, heads, sequence, dim]");
  TORCH_CHECK(tensor.stride(3) == 1, name,
              " head dimension must be contiguous");
}

std::vector<at::Tensor> fused_rope_qk_bf16(at::Tensor query, at::Tensor key,
                                           at::Tensor cos, at::Tensor sin) {
  check_bhsd_tensor(query, "query");
  check_bhsd_tensor(key, "key");
  check_bf16_cuda_tensor(cos, "cos");
  check_bf16_cuda_tensor(sin, "sin");
  TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3,
              "cos and sin must have shape [batch, sequence, dim]");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
              "cos and sin must be contiguous");
  TORCH_CHECK(query.device() == key.device() &&
                  query.device() == cos.device() &&
                  query.device() == sin.device(),
              "query, key, cos, and sin must share a CUDA device");
  TORCH_CHECK(query.size(0) == key.size(0),
              "query and key batch sizes must match");
  TORCH_CHECK(query.size(3) == key.size(3),
              "query and key head dimensions must match");
  TORCH_CHECK(query.size(3) % 2 == 0, "RoPE head dimension must be even");
  TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin shapes must match");
  TORCH_CHECK(cos.size(0) == 1 || cos.size(0) == query.size(0),
              "cos batch must be one or match query batch");
  TORCH_CHECK(cos.size(1) >= query.size(2) && cos.size(1) >= key.size(2),
              "cos sequence must cover query and key sequences");
  TORCH_CHECK(cos.size(2) == query.size(3),
              "cos head dimension must match query head dimension");

  c10::cuda::CUDAGuard guard(query.device());
  at::Tensor query_output = at::empty(query.sizes(), query.options());
  at::Tensor key_output = at::empty(key.sizes(), key.options());
  int64_t total = query.numel() + key.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>(ceil_div_int64(total, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
  fused_rope_qk_bf16_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(cos.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(sin.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(query_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(key_output.data_ptr<at::BFloat16>()),
      query.numel(), key.numel(), query.size(1), key.size(1), query.size(2),
      key.size(2), query.size(3), query.stride(0), query.stride(1),
      query.stride(2), query.stride(3), key.stride(0), key.stride(1),
      key.stride(2), key.stride(3), cos.size(0), cos.size(1), cos.stride(0),
      cos.stride(1), cos.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {query_output, key_output};
}

std::vector<at::Tensor> fused_rope_qk_pair_bf16(at::Tensor query,
                                                at::Tensor key, at::Tensor cos,
                                                at::Tensor sin) {
  check_bhsd_tensor(query, "query");
  check_bhsd_tensor(key, "key");
  check_bf16_cuda_tensor(cos, "cos");
  check_bf16_cuda_tensor(sin, "sin");
  TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3,
              "cos and sin must have shape [batch, sequence, dim]");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
              "cos and sin must be contiguous");
  TORCH_CHECK(query.device() == key.device() &&
                  query.device() == cos.device() &&
                  query.device() == sin.device(),
              "query, key, cos, and sin must share a CUDA device");
  TORCH_CHECK(query.size(0) == key.size(0),
              "query and key batch sizes must match");
  TORCH_CHECK(query.size(3) == key.size(3),
              "query and key head dimensions must match");
  TORCH_CHECK(query.size(3) % 2 == 0, "RoPE head dimension must be even");
  TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin shapes must match");
  TORCH_CHECK(cos.size(0) == 1 || cos.size(0) == query.size(0),
              "cos batch must be one or match query batch");
  TORCH_CHECK(cos.size(1) >= query.size(2) && cos.size(1) >= key.size(2),
              "cos sequence must cover query and key sequences");
  TORCH_CHECK(cos.size(2) == query.size(3),
              "cos head dimension must match query head dimension");

  c10::cuda::CUDAGuard guard(query.device());
  at::Tensor query_output = at::empty(query.sizes(), query.options());
  at::Tensor key_output = at::empty(key.sizes(), key.options());
  int64_t query_pairs = query.numel() / 2;
  int64_t key_pairs = key.numel() / 2;
  constexpr int threads = 256;
  int blocks =
      static_cast<int>(ceil_div_int64(query_pairs + key_pairs, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
  fused_rope_qk_pair_bf16_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(cos.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(sin.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(query_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(key_output.data_ptr<at::BFloat16>()),
      query_pairs, key_pairs, query.size(1), key.size(1), query.size(2),
      key.size(2), query.size(3), query.stride(0), query.stride(1),
      query.stride(2), query.stride(3), key.stride(0), key.stride(1),
      key.stride(2), key.stride(3), cos.size(0), cos.size(1), cos.stride(0),
      cos.stride(1), cos.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {query_output, key_output};
}

std::vector<at::Tensor> fused_vision_rope_qk_bf16(at::Tensor query,
                                                  at::Tensor key,
                                                  at::Tensor cos,
                                                  at::Tensor sin) {
  check_bf16_cuda_tensor(query, "query");
  check_bf16_cuda_tensor(key, "key");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "cos and sin must be CUDA");
  TORCH_CHECK(cos.scalar_type() == at::kFloat &&
                  sin.scalar_type() == at::kFloat,
              "vision cos and sin must be float32");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3,
              "vision query and key must have shape [sequence, heads, dim]");
  TORCH_CHECK(query.sizes() == key.sizes(),
              "vision query and key shapes must match");
  TORCH_CHECK(query.stride(2) == 1 && key.stride(2) == 1,
              "vision query and key head dimensions must be contiguous");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2 && cos.is_contiguous() &&
                  sin.is_contiguous(),
              "vision cos and sin must be contiguous [sequence, dim] tensors");
  TORCH_CHECK(cos.sizes() == sin.sizes() && cos.size(0) == query.size(0) &&
                  cos.size(1) == query.size(2),
              "vision cos and sin shapes must match query sequence and dim");
  TORCH_CHECK(query.size(2) % 2 == 0,
              "vision RoPE head dimension must be even");
  TORCH_CHECK(query.device() == key.device() &&
                  query.device() == cos.device() &&
                  query.device() == sin.device(),
              "vision RoPE tensors must share a CUDA device");

  c10::cuda::CUDAGuard guard(query.device());
  at::Tensor query_output = at::empty(query.sizes(), query.options());
  at::Tensor key_output = at::empty(key.sizes(), key.options());
  constexpr int threads = 256;
  int64_t total = query.numel() / 2;
  int blocks = static_cast<int>(ceil_div_int64(total, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
  fused_vision_rope_qk_bf16_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(key.data_ptr<at::BFloat16>()),
      cos.data_ptr<float>(), sin.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16 *>(query_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(key_output.data_ptr<at::BFloat16>()),
      query.numel(), query.size(1), query.size(0), query.size(2),
      query.stride(0), query.stride(1), query.stride(2), key.stride(0),
      key.stride(1), key.stride(2), cos.stride(0), cos.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {query_output, key_output};
}

std::vector<at::Tensor> fused_vision_rope_qk_scalar_bf16(at::Tensor query,
                                                         at::Tensor key,
                                                         at::Tensor cos,
                                                         at::Tensor sin) {
  check_bf16_cuda_tensor(query, "query");
  check_bf16_cuda_tensor(key, "key");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "cos and sin must be CUDA");
  TORCH_CHECK(cos.scalar_type() == at::kFloat &&
                  sin.scalar_type() == at::kFloat,
              "vision cos and sin must be float32");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 &&
                  query.sizes() == key.sizes(),
              "vision query and key shapes must match at rank three");
  TORCH_CHECK(query.stride(2) == 1 && key.stride(2) == 1,
              "vision query and key head dimensions must be contiguous");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2 && cos.is_contiguous() &&
                  sin.is_contiguous() && cos.sizes() == sin.sizes() &&
                  cos.size(0) == query.size(0) && cos.size(1) == query.size(2),
              "vision cos and sin shapes must match query sequence and dim");
  TORCH_CHECK(query.size(2) % 2 == 0,
              "vision RoPE head dimension must be even");
  TORCH_CHECK(query.device() == key.device() &&
                  query.device() == cos.device() &&
                  query.device() == sin.device(),
              "vision RoPE tensors must share a CUDA device");

  c10::cuda::CUDAGuard guard(query.device());
  at::Tensor query_output = at::empty(query.sizes(), query.options());
  at::Tensor key_output = at::empty(key.sizes(), key.options());
  constexpr int threads = 256;
  int64_t total = 2 * query.numel();
  int blocks = static_cast<int>(ceil_div_int64(total, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
  fused_vision_rope_qk_scalar_bf16_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(key.data_ptr<at::BFloat16>()),
      cos.data_ptr<float>(), sin.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16 *>(query_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(key_output.data_ptr<at::BFloat16>()),
      query.numel(), query.size(1), query.size(2), query.stride(0),
      query.stride(1), query.stride(2), key.stride(0), key.stride(1),
      key.stride(2), cos.stride(0), cos.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {query_output, key_output};
}

at::Tensor fused_paged_cache_update_rope_bf16(
    at::Tensor query, at::Tensor new_key, at::Tensor new_value,
    at::Tensor raw_key_storage, at::Tensor rotated_key_storage,
    at::Tensor value_storage, at::Tensor cos, at::Tensor sin,
    int64_t old_sequence, bool rerotate_full, int64_t threads_per_block) {
  check_bhsd_tensor(query, "query");
  check_bhsd_tensor(new_key, "new_key");
  check_bhsd_tensor(new_value, "new_value");
  check_bf16_cuda_tensor(raw_key_storage, "raw_key_storage");
  check_bf16_cuda_tensor(rotated_key_storage, "rotated_key_storage");
  check_bf16_cuda_tensor(value_storage, "value_storage");
  check_bf16_cuda_tensor(cos, "cos");
  check_bf16_cuda_tensor(sin, "sin");
  TORCH_CHECK(raw_key_storage.is_contiguous() &&
                  rotated_key_storage.is_contiguous() &&
                  value_storage.is_contiguous(),
              "paged K/V storages must be contiguous");
  TORCH_CHECK(raw_key_storage.sizes() == rotated_key_storage.sizes() &&
                  raw_key_storage.sizes() == value_storage.sizes(),
              "paged K/V storage shapes must match");
  TORCH_CHECK(query.device() == new_key.device() &&
                  query.device() == new_value.device() &&
                  query.device() == raw_key_storage.device() &&
                  query.device() == rotated_key_storage.device() &&
                  query.device() == value_storage.device() &&
                  query.device() == cos.device() &&
                  query.device() == sin.device(),
              "all paged cache update tensors must share a CUDA device");
  TORCH_CHECK(new_key.sizes() == new_value.sizes(),
              "new key and value shapes must match");
  TORCH_CHECK(query.size(0) == 1 && new_key.size(0) == 1,
              "paged cache update supports batch size one");
  TORCH_CHECK(query.size(2) == new_key.size(2),
              "query and new KV sequence lengths must match");
  TORCH_CHECK(
      query.size(3) == new_key.size(3) && query.size(3) % 8 == 0,
      "query and KV head dimensions must match and be divisible by eight");
  TORCH_CHECK(old_sequence >= 0, "old sequence length must be nonnegative");
  TORCH_CHECK(threads_per_block == 64 || threads_per_block == 128 ||
                  threads_per_block == 256,
              "threads_per_block must be 64, 128, or 256");
  TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3 && cos.is_contiguous() &&
                  sin.is_contiguous() && cos.sizes() == sin.sizes(),
              "cos and sin must be contiguous matching rank-three tensors");
  TORCH_CHECK(cos.size(0) == 1 && cos.size(2) == query.size(3),
              "cos and sin must have shape [1, sequence, head_dim]");
  int64_t query_sequence = query.size(2);
  int64_t key_heads = new_key.size(1);
  int64_t head_dim = query.size(3);
  int64_t elements_per_sequence = key_heads * head_dim;
  TORCH_CHECK(raw_key_storage.numel() % elements_per_sequence == 0,
              "paged storage size must be divisible by KV heads * head_dim");
  int64_t capacity = raw_key_storage.numel() / elements_per_sequence;
  int64_t full_sequence = old_sequence + query_sequence;
  TORCH_CHECK(full_sequence <= capacity,
              "paged cache update exceeds storage capacity");
  int64_t required_cos = rerotate_full ? full_sequence : query_sequence;
  TORCH_CHECK(cos.size(1) == required_cos,
              "cos sequence does not match paged cache update mode");

  c10::cuda::CUDAGuard guard(query.device());
  at::Tensor query_output = at::empty(query.sizes(), query.options());
  int64_t rotated_sequence = rerotate_full ? full_sequence : query_sequence;
  int64_t rotated_key_elements = rotated_sequence * elements_per_sequence;
  constexpr int kVectorElements = sizeof(uint4) / sizeof(__nv_bfloat16);
  int64_t value_vectors =
      query_sequence * elements_per_sequence / kVectorElements;
  int64_t total = query.numel() + rotated_key_elements + value_vectors;
  int threads = static_cast<int>(threads_per_block);
  int blocks = static_cast<int>(ceil_div_int64(total, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
  fused_paged_cache_update_rope_bf16_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(new_key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(
          new_value.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(
          raw_key_storage.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(
          rotated_key_storage.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(value_storage.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(cos.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(sin.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(query_output.data_ptr<at::BFloat16>()),
      query.numel(), rotated_key_elements, value_vectors, query.size(1),
      key_heads, query_sequence, old_sequence, capacity, head_dim,
      query.stride(0), query.stride(1), query.stride(2), query.stride(3),
      new_key.stride(0), new_key.stride(1), new_key.stride(2),
      new_key.stride(3), new_value.stride(0), new_value.stride(1),
      new_value.stride(2), new_value.stride(3), cos.size(0), cos.size(1),
      cos.stride(0), cos.stride(1), cos.stride(2), rerotate_full);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return query_output;
}

at::Tensor rms_norm_bf16(at::Tensor input, at::Tensor weight, double epsilon) {
  check_bf16_cuda_tensor(input, "input");
  check_bf16_cuda_tensor(weight, "weight");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous() && weight.dim() == 1,
              "weight must be a contiguous vector");
  TORCH_CHECK(input.device() == weight.device(),
              "input and weight must share a CUDA device");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) == weight.numel(),
              "input last dimension must match weight size");
  TORCH_CHECK(input.numel() > 0, "input must not be empty");

  c10::cuda::CUDAGuard guard(input.device());
  int64_t columns = input.size(-1);
  int64_t rows = input.numel() / columns;
  at::Tensor output = at::empty_like(input);
  constexpr int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  rms_norm_bf16_kernel<false><<<static_cast<int>(rows), threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()),
      nullptr,
      reinterpret_cast<const __nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()),
      nullptr,
      reinterpret_cast<__nv_bfloat16 *>(output.data_ptr<at::BFloat16>()),
      columns, static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<at::Tensor> add_rms_norm_bf16(at::Tensor input, at::Tensor residual,
                                          at::Tensor weight, double epsilon) {
  check_bf16_cuda_tensor(input, "input");
  check_bf16_cuda_tensor(residual, "residual");
  check_bf16_cuda_tensor(weight, "weight");
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous(),
              "input and residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous() && weight.dim() == 1,
              "weight must be a contiguous vector");
  TORCH_CHECK(input.sizes() == residual.sizes(),
              "input and residual shapes must match");
  TORCH_CHECK(input.device() == residual.device() &&
                  input.device() == weight.device(),
              "input, residual, and weight must share a CUDA device");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) == weight.numel(),
              "input last dimension must match weight size");
  TORCH_CHECK(input.numel() > 0, "input must not be empty");

  c10::cuda::CUDAGuard guard(input.device());
  int64_t columns = input.size(-1);
  int64_t rows = input.numel() / columns;
  at::Tensor summed = at::empty_like(input);
  at::Tensor normalized = at::empty_like(input);
  constexpr int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  rms_norm_bf16_kernel<true><<<static_cast<int>(rows), threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(
          residual.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(summed.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(normalized.data_ptr<at::BFloat16>()),
      columns, static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {summed, normalized};
}

std::vector<at::Tensor> add_rms_norm_fp4_quant_bf16(at::Tensor input,
                                                    at::Tensor residual,
                                                    at::Tensor weight,
                                                    double epsilon,
                                                    bool return_normalized) {
  check_bf16_cuda_tensor(input, "input");
  check_bf16_cuda_tensor(residual, "residual");
  check_bf16_cuda_tensor(weight, "weight");
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous(),
              "input and residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous() && weight.dim() == 1,
              "weight must be a contiguous vector");
  TORCH_CHECK(input.sizes() == residual.sizes(),
              "input and residual shapes must match");
  TORCH_CHECK(input.device() == residual.device() &&
                  input.device() == weight.device(),
              "input, residual, and weight must share a CUDA device");
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) == weight.numel(),
              "input last dimension must match weight size");
  TORCH_CHECK(input.numel() > 0, "input must not be empty");

  int64_t columns = input.size(-1);
  TORCH_CHECK(columns % kElementsPerAccess == 0,
              "fused FP4 packing requires hidden size divisible by ",
              kElementsPerAccess, ", got ", columns);
  int64_t rows = input.numel() / columns;
  int64_t packed_cols = columns / 2;
  int64_t sf_per_row = ceil_div_int64(columns, kSfVecSize);
  int64_t rounded_m = round_up_int64(rows, 128);
  int64_t rounded_scale_cols = round_up_int64(sf_per_row, 4);

  c10::cuda::CUDAGuard guard(input.device());
  auto byte_options = input.options().dtype(at::kByte);
  at::Tensor summed = at::empty_like(input);
  at::Tensor normalized =
      return_normalized ? at::empty_like(input) : at::Tensor();
  at::Tensor packed = at::empty({rows, packed_cols}, byte_options);
  at::Tensor scales = at::empty({rounded_m, rounded_scale_cols}, byte_options);
  constexpr int threads = 256;
  size_t shared_bytes = static_cast<size_t>(columns) * sizeof(__nv_bfloat16);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  auto launch = [&](auto write_normalized) {
    constexpr bool kWriteNormalized = decltype(write_normalized)::value;
    rms_norm_fp4_quant_bf16_kernel<true, kWriteNormalized>
        <<<static_cast<int>(rows), threads, shared_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(
                input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16 *>(
                residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16 *>(
                weight.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16 *>(summed.data_ptr<at::BFloat16>()),
            kWriteNormalized ? reinterpret_cast<__nv_bfloat16 *>(
                                   normalized.data_ptr<at::BFloat16>())
                             : nullptr,
            packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), columns,
            packed_cols, sf_blocks_k(columns), sf_per_row,
            static_cast<float>(epsilon));
  };
  if (return_normalized) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (return_normalized) {
    return {summed, normalized, packed, scales};
  }
  return {summed, packed, scales};
}

std::vector<at::Tensor>
fused_cache_update_rope_bf16(at::Tensor query, at::Tensor old_key,
                             at::Tensor old_value, at::Tensor old_rotated_key,
                             at::Tensor new_key, at::Tensor new_value,
                             at::Tensor cos, at::Tensor sin) {
  check_bhsd_tensor(query, "query");
  check_bhsd_tensor(old_key, "old_key");
  check_bhsd_tensor(old_value, "old_value");
  check_bhsd_tensor(old_rotated_key, "old_rotated_key");
  check_bhsd_tensor(new_key, "new_key");
  check_bhsd_tensor(new_value, "new_value");
  check_bf16_cuda_tensor(cos, "cos");
  check_bf16_cuda_tensor(sin, "sin");
  TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3 && cos.is_contiguous() &&
                  sin.is_contiguous(),
              "cos and sin must be contiguous [batch, sequence, dim] tensors");
  TORCH_CHECK(query.device() == old_key.device() &&
                  query.device() == old_value.device() &&
                  query.device() == old_rotated_key.device() &&
                  query.device() == new_key.device() &&
                  query.device() == new_value.device() &&
                  query.device() == cos.device() &&
                  query.device() == sin.device(),
              "all fused cache inputs must share a CUDA device");
  TORCH_CHECK(old_key.sizes() == old_value.sizes() &&
                  old_key.sizes() == old_rotated_key.sizes(),
              "old key, value, and rotated-key cache shapes must match");
  TORCH_CHECK(new_key.sizes() == new_value.sizes(),
              "new key and value shapes must match");
  TORCH_CHECK(query.size(0) == old_key.size(0) &&
                  query.size(0) == new_key.size(0),
              "cache and query batch sizes must match");
  TORCH_CHECK(old_key.size(1) == new_key.size(1) &&
                  old_key.size(3) == new_key.size(3),
              "old and new KV head shapes must match");
  TORCH_CHECK(query.size(2) == new_key.size(2),
              "query and new KV sequence lengths must match");
  TORCH_CHECK(query.size(3) == new_key.size(3) && query.size(3) % 2 == 0,
              "query and key head dimensions must match and be even");
  TORCH_CHECK(query.size(3) % 8 == 0,
              "fused cache head dimension must be divisible by eight");
  TORCH_CHECK(cos.sizes() == sin.sizes() && cos.size(1) == new_key.size(2) &&
                  cos.size(2) == query.size(3),
              "cos and sin must cover exactly the appended sequence");
  TORCH_CHECK(cos.size(0) == 1 || cos.size(0) == query.size(0),
              "cos batch must be one or match query batch");

  c10::cuda::CUDAGuard guard(query.device());
  int64_t full_sequence = old_key.size(2) + new_key.size(2);
  at::Tensor query_output = at::empty(query.sizes(), query.options());
  at::Tensor key_output = at::empty(
      {old_key.size(0), old_key.size(1), full_sequence, old_key.size(3)},
      old_key.options());
  at::Tensor value_output = at::empty_like(key_output);
  at::Tensor rotated_key_output = at::empty_like(key_output);
  int64_t full_cache_elements = key_output.numel();
  int64_t full_cache_vectors = full_cache_elements / 8;
  int64_t total = query.numel() + 3 * full_cache_vectors;
  constexpr int threads = 256;
  int blocks = static_cast<int>(ceil_div_int64(total, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
  fused_cache_update_rope_bf16_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(old_key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(
          old_value.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(
          old_rotated_key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(new_key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(
          new_value.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(cos.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(sin.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(query_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(key_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(value_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(
          rotated_key_output.data_ptr<at::BFloat16>()),
      query.numel(), full_cache_vectors, query.size(1), new_key.size(1),
      query.size(2), old_key.size(2), new_key.size(2), query.size(3),
      query.stride(0), query.stride(1), query.stride(2), query.stride(3),
      old_key.stride(0), old_key.stride(1), old_key.stride(2),
      old_key.stride(3), old_value.stride(0), old_value.stride(1),
      old_value.stride(2), old_value.stride(3), old_rotated_key.stride(0),
      old_rotated_key.stride(1), old_rotated_key.stride(2),
      old_rotated_key.stride(3), new_key.stride(0), new_key.stride(1),
      new_key.stride(2), new_key.stride(3), new_value.stride(0),
      new_value.stride(1), new_value.stride(2), new_value.stride(3),
      cos.size(0), cos.stride(0), cos.stride(1), cos.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {query_output, key_output, value_output, rotated_key_output};
}

std::vector<at::Tensor> fused_prefill_cache_update_rope_bf16(
    at::Tensor query, at::Tensor old_key, at::Tensor old_value,
    at::Tensor new_key, at::Tensor new_value, at::Tensor cos, at::Tensor sin) {
  check_bhsd_tensor(query, "query");
  check_bhsd_tensor(old_key, "old_key");
  check_bhsd_tensor(old_value, "old_value");
  check_bhsd_tensor(new_key, "new_key");
  check_bhsd_tensor(new_value, "new_value");
  check_bf16_cuda_tensor(cos, "cos");
  check_bf16_cuda_tensor(sin, "sin");
  TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3 && cos.is_contiguous() &&
                  sin.is_contiguous(),
              "cos and sin must be contiguous [batch, sequence, dim] tensors");
  TORCH_CHECK(query.device() == old_key.device() &&
                  query.device() == old_value.device() &&
                  query.device() == new_key.device() &&
                  query.device() == new_value.device() &&
                  query.device() == cos.device() &&
                  query.device() == sin.device(),
              "all fused prefill inputs must share a CUDA device");
  TORCH_CHECK(old_key.sizes() == old_value.sizes(),
              "old key and value cache shapes must match");
  TORCH_CHECK(new_key.sizes() == new_value.sizes(),
              "new key and value shapes must match");
  TORCH_CHECK(query.size(0) == old_key.size(0) &&
                  query.size(0) == new_key.size(0),
              "cache and query batch sizes must match");
  TORCH_CHECK(old_key.size(1) == new_key.size(1) &&
                  old_key.size(3) == new_key.size(3),
              "old and new KV head shapes must match");
  TORCH_CHECK(query.size(2) == new_key.size(2),
              "query and new KV sequence lengths must match");
  TORCH_CHECK(
      query.size(3) == new_key.size(3) && query.size(3) % 8 == 0,
      "query and key head dimensions must match and be divisible by eight");
  int64_t full_sequence = old_key.size(2) + new_key.size(2);
  TORCH_CHECK(cos.sizes() == sin.sizes() && cos.size(1) == full_sequence &&
                  cos.size(2) == query.size(3),
              "cos and sin must cover the full prefill cache sequence");
  TORCH_CHECK(cos.size(0) == 1 || cos.size(0) == query.size(0),
              "cos batch must be one or match query batch");

  c10::cuda::CUDAGuard guard(query.device());
  at::Tensor query_output = at::empty(query.sizes(), query.options());
  at::Tensor key_output = at::empty(
      {old_key.size(0), old_key.size(1), full_sequence, old_key.size(3)},
      old_key.options());
  at::Tensor value_output = at::empty_like(key_output);
  at::Tensor rotated_key_output = at::empty_like(key_output);
  int64_t full_cache_elements = key_output.numel();
  int64_t full_cache_vectors = full_cache_elements / 8;
  int64_t total = query.numel() + full_cache_elements + full_cache_vectors;
  constexpr int threads = 256;
  int blocks = static_cast<int>(ceil_div_int64(total, threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
  fused_prefill_cache_update_rope_bf16_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(old_key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(
          old_value.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(new_key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(
          new_value.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(cos.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(sin.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(query_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(key_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(value_output.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(
          rotated_key_output.data_ptr<at::BFloat16>()),
      query.numel(), full_cache_elements, full_cache_vectors, query.size(1),
      new_key.size(1), query.size(2), old_key.size(2), new_key.size(2),
      query.size(3), query.stride(0), query.stride(1), query.stride(2),
      query.stride(3), old_key.stride(0), old_key.stride(1), old_key.stride(2),
      old_key.stride(3), old_value.stride(0), old_value.stride(1),
      old_value.stride(2), old_value.stride(3), new_key.stride(0),
      new_key.stride(1), new_key.stride(2), new_key.stride(3),
      new_value.stride(0), new_value.stride(1), new_value.stride(2),
      new_value.stride(3), cos.size(0), cos.size(1), cos.stride(0),
      cos.stride(1), cos.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {query_output, key_output, value_output, rotated_key_output};
}

} // namespace

TORCH_LIBRARY(robotics_cutlass_fp4, m) {
  m.def("supports_device(int cc) -> bool");
  m.def("set_decode_policy(int policy) -> ()");
  m.def("get_decode_policy() -> int");
  m.def("pack_linear_weight(Tensor weight, str role, str target_name) -> "
        "Tensor[]");
  m.def("pack_linear_weight_gemm_bf16(Tensor weight, str role, str "
        "target_name) -> Tensor[]");
  m.def(
      "linear_forward(Tensor input, Tensor packed_weight, Tensor weight_scale, "
      "Tensor metadata, Tensor? bias, str role, str target_name) -> Tensor");
  m.def("linear_forward_gemm_bf16(Tensor input, Tensor packed_weight, "
        "Tensor weight_scale, Tensor metadata, Tensor? bias, str role, "
        "str target_name) -> Tensor");
  m.def("pack_activation_gemm_bf16(Tensor input) -> Tensor[]");
  m.def("linear_forward_packed_gemm_bf16(Tensor packed_input, "
        "Tensor input_scale, Tensor packed_weight, Tensor weight_scale, "
        "Tensor metadata, Tensor? bias, str role, str target_name) -> Tensor");
  m.def("linear_forward_packed_gemm_bf16_tactic(Tensor packed_input, "
        "Tensor input_scale, Tensor packed_weight, Tensor weight_scale, "
        "Tensor metadata, Tensor? bias, str role, str target_name, "
        "int tactic) -> Tensor");
  m.def("linear_forward_packed_greedy_gemm_bf16(Tensor packed_input, "
        "Tensor input_scale, Tensor packed_weight, Tensor weight_scale, "
        "Tensor metadata, Tensor? bias, Tensor seen_tokens, "
        "Tensor eos_tokens, Tensor generated_count, float repetition_penalty, "
        "int min_new_tokens, str role, str target_name) -> Tensor");
  m.def(
      "linear_forward_packed_topk_gemm_bf16(Tensor packed_input, "
      "Tensor input_scale, Tensor packed_weight, Tensor weight_scale, "
      "Tensor metadata, Tensor? bias, Tensor seen_tokens, "
      "Tensor eos_tokens, Tensor generated_count, float repetition_penalty, "
      "int min_new_tokens, int top_k, str role, str target_name) -> Tensor[]");
  m.def("silu_mul_bf16(Tensor gate_up, int intermediate_size, "
        "bool interleaved=False) -> Tensor");
  m.def("linear_forward_swiglu_gemm_bf16(Tensor gate_up, "
        "int intermediate_size, Tensor packed_weight, Tensor weight_scale, "
        "Tensor metadata, Tensor? bias, str role, str target_name, "
        "bool interleaved=False) -> Tensor");
  m.def("fused_rope_qk_bf16(Tensor query, Tensor key, Tensor cos, Tensor sin) "
        "-> Tensor[]");
  m.def("fused_rope_qk_pair_bf16(Tensor query, Tensor key, Tensor cos, "
        "Tensor sin) -> Tensor[]");
  m.def("fused_vision_rope_qk_bf16(Tensor query, Tensor key, Tensor cos, "
        "Tensor sin) -> Tensor[]");
  m.def("fused_vision_rope_qk_scalar_bf16(Tensor query, Tensor key, "
        "Tensor cos, Tensor sin) -> Tensor[]");
  m.def("fused_paged_cache_update_rope_bf16(Tensor query, Tensor new_key, "
        "Tensor new_value, Tensor raw_key_storage, "
        "Tensor rotated_key_storage, Tensor value_storage, Tensor cos, "
        "Tensor sin, int old_sequence, bool rerotate_full, "
        "int threads_per_block) -> Tensor");
  m.def("rms_norm_bf16(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def("add_rms_norm_bf16(Tensor input, Tensor residual, Tensor weight, "
        "float epsilon) -> Tensor[]");
  m.def(
      "add_rms_norm_fp4_quant_bf16(Tensor input, Tensor residual, "
      "Tensor weight, float epsilon, bool return_normalized=True) -> Tensor[]");
  m.def("fused_cache_update_rope_bf16(Tensor query, Tensor old_key, "
        "Tensor old_value, Tensor old_rotated_key, Tensor new_key, "
        "Tensor new_value, Tensor cos, Tensor sin) -> Tensor[]");
  m.def("fused_prefill_cache_update_rope_bf16(Tensor query, Tensor old_key, "
        "Tensor old_value, Tensor new_key, Tensor new_value, Tensor cos, "
        "Tensor sin) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(robotics_cutlass_fp4, CUDA, m) {
  m.impl("pack_linear_weight", &pack_linear_weight);
  m.impl("pack_linear_weight_gemm_bf16", &pack_linear_weight_gemm_bf16);
  m.impl("linear_forward", &linear_forward);
  m.impl("linear_forward_gemm_bf16", &linear_forward_gemm_bf16);
  m.impl("pack_activation_gemm_bf16", &pack_activation_gemm_bf16);
  m.impl("linear_forward_packed_gemm_bf16", &linear_forward_packed_gemm_bf16);
  m.impl("linear_forward_packed_gemm_bf16_tactic",
         &linear_forward_packed_gemm_bf16_tactic);
  m.impl("linear_forward_packed_greedy_gemm_bf16",
         &linear_forward_packed_greedy_gemm_bf16);
  m.impl("linear_forward_packed_topk_gemm_bf16",
         &linear_forward_packed_topk_gemm_bf16);
  m.impl("silu_mul_bf16", &silu_mul_bf16);
  m.impl("linear_forward_swiglu_gemm_bf16", &linear_forward_swiglu_gemm_bf16);
  m.impl("fused_rope_qk_bf16", &fused_rope_qk_bf16);
  m.impl("fused_rope_qk_pair_bf16", &fused_rope_qk_pair_bf16);
  m.impl("fused_vision_rope_qk_bf16", &fused_vision_rope_qk_bf16);
  m.impl("fused_vision_rope_qk_scalar_bf16", &fused_vision_rope_qk_scalar_bf16);
  m.impl("fused_paged_cache_update_rope_bf16",
         &fused_paged_cache_update_rope_bf16);
  m.impl("rms_norm_bf16", &rms_norm_bf16);
  m.impl("add_rms_norm_bf16", &add_rms_norm_bf16);
  m.impl("add_rms_norm_fp4_quant_bf16", &add_rms_norm_fp4_quant_bf16);
  m.impl("fused_cache_update_rope_bf16", &fused_cache_update_rope_bf16);
  m.impl("fused_prefill_cache_update_rope_bf16",
         &fused_prefill_cache_update_rope_bf16);
}

TORCH_LIBRARY_IMPL(robotics_cutlass_fp4, CatchAll, m) {
  m.impl("supports_device", &supports_device);
  m.impl("set_decode_policy", &set_decode_policy);
  m.impl("get_decode_policy", &get_decode_policy);
}
