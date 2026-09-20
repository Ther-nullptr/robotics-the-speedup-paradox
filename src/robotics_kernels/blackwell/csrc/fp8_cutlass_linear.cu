#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/library.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

#include "sampling_bf16.cuh"

namespace {

constexpr float kFp8Max = 448.0f;
constexpr float kMinScale = 1.0e-12f;
constexpr int64_t kDecodePolicyBalanced = 0;
constexpr int64_t kDecodePolicyLowClock = 1;
constexpr int64_t kDecodePolicyLegacy = 2;
std::atomic<int64_t> g_decode_policy{kDecodePolicyBalanced};
std::atomic<bool> g_swap_ab_enabled{true};

int64_t ceil_div_int64(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

void check_bf16_cuda_contiguous(const at::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name,
              " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_fp8_cuda_contiguous(const at::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float8_e4m3fn, name,
              " must have dtype torch.float8_e4m3fn");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__global__ void bf16_amax_kernel(const __nv_bfloat16 *input, float *output,
                                 int64_t elements) {
  float local_max = 0.0f;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float value = fabsf(__bfloat162float(input[index]));
    if (isfinite(value)) {
      local_max = fmaxf(local_max, value);
    }
  }

  __shared__ float block_max[256];
  block_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_max[threadIdx.x] =
          fmaxf(block_max[threadIdx.x], block_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<unsigned int *>(output),
              __float_as_uint(block_max[0]));
  }
}

__global__ void finalize_scale_kernel(float *scale) {
  scale[0] = fmaxf(scale[0] / kFp8Max, kMinScale);
}

__global__ void bf16_to_fp8_kernel(const __nv_bfloat16 *input, uint8_t *output,
                                   const float *scale, int64_t elements) {
  float inverse_scale = 1.0f / scale[0];
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    float value = __bfloat162float(input[index]) * inverse_scale;
    value = fminf(kFp8Max, fmaxf(-kFp8Max, value));
    output[index] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
}

__global__ void bf16_to_fp8_single_row_kernel(const __nv_bfloat16 *input,
                                              uint8_t *output,
                                              float *output_scale,
                                              int64_t elements) {
  float local_max = 0.0f;
  for (int64_t index = threadIdx.x; index < elements; index += blockDim.x) {
    float value = fabsf(__bfloat162float(input[index]));
    if (isfinite(value)) {
      local_max = fmaxf(local_max, value);
    }
  }
  __shared__ float block_max[256];
  block_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_max[threadIdx.x] =
          fmaxf(block_max[threadIdx.x], block_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    block_max[0] = fmaxf(block_max[0] / kFp8Max, kMinScale);
    output_scale[0] = block_max[0];
  }
  __syncthreads();
  float inverse_scale = 1.0f / block_max[0];
  for (int64_t index = threadIdx.x; index < elements; index += blockDim.x) {
    float value = __bfloat162float(input[index]) * inverse_scale;
    value = fminf(kFp8Max, fmaxf(-kFp8Max, value));
    output[index] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
}

__device__ __forceinline__ float warp_sum_fp8(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void add_rms_norm_amax_bf16_kernel(const __nv_bfloat16 *input,
                                              const __nv_bfloat16 *residual,
                                              const __nv_bfloat16 *weight,
                                              __nv_bfloat16 *summed,
                                              __nv_bfloat16 *normalized,
                                              float *output_scale,
                                              int64_t columns, float epsilon) {
  int64_t row_offset = static_cast<int64_t>(blockIdx.x) * columns;
  float sum_squares = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    __nv_bfloat16 value =
        __float2bfloat16_rn(__bfloat162float(input[row_offset + column]) +
                            __bfloat162float(residual[row_offset + column]));
    summed[row_offset + column] = value;
    float value_float = __bfloat162float(value);
    sum_squares = fmaf(value_float, value_float, sum_squares);
  }
  sum_squares = warp_sum_fp8(sum_squares);
  __shared__ float warp_values[256];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_values[warp] = sum_squares;
  }
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < (blockDim.x + 31) / 32 ? warp_values[lane] : 0.0f;
    block_sum = warp_sum_fp8(block_sum);
    if (lane == 0) {
      warp_values[0] =
          rsqrtf(block_sum / static_cast<float>(columns) + epsilon);
    }
  }
  __syncthreads();
  float inverse_rms = warp_values[0];
  float local_max = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = __bfloat162float(summed[row_offset + column]);
    __nv_bfloat16 normalized_value = __float2bfloat16_rn(value * inverse_rms);
    normalized_value = __float2bfloat16_rn(__bfloat162float(normalized_value) *
                                           __bfloat162float(weight[column]));
    normalized[row_offset + column] = normalized_value;
    float normalized_float = fabsf(__bfloat162float(normalized_value));
    if (isfinite(normalized_float)) {
      local_max = fmaxf(local_max, normalized_float);
    }
  }
  warp_values[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      warp_values[threadIdx.x] =
          fmaxf(warp_values[threadIdx.x], warp_values[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<unsigned int *>(output_scale),
              __float_as_uint(warp_values[0]));
  }
}

__global__ void add_rms_norm_quant_single_row_bf16_kernel(
    const __nv_bfloat16 *input, const __nv_bfloat16 *residual,
    const __nv_bfloat16 *weight, __nv_bfloat16 *summed,
    __nv_bfloat16 *normalized, uint8_t *packed, float *output_scale,
    int64_t columns, float epsilon) {
  float sum_squares = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    __nv_bfloat16 value = __float2bfloat16_rn(
        __bfloat162float(input[column]) + __bfloat162float(residual[column]));
    summed[column] = value;
    float value_float = __bfloat162float(value);
    sum_squares = fmaf(value_float, value_float, sum_squares);
  }
  sum_squares = warp_sum_fp8(sum_squares);
  __shared__ float values[256];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) {
    values[warp] = sum_squares;
  }
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < (blockDim.x + 31) / 32 ? values[lane] : 0.0f;
    block_sum = warp_sum_fp8(block_sum);
    if (lane == 0) {
      values[0] = rsqrtf(block_sum / static_cast<float>(columns) + epsilon);
    }
  }
  __syncthreads();
  float inverse_rms = values[0];
  float local_max = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = __bfloat162float(summed[column]);
    __nv_bfloat16 normalized_value = __float2bfloat16_rn(value * inverse_rms);
    normalized_value = __float2bfloat16_rn(__bfloat162float(normalized_value) *
                                           __bfloat162float(weight[column]));
    normalized[column] = normalized_value;
    float normalized_float = fabsf(__bfloat162float(normalized_value));
    if (isfinite(normalized_float)) {
      local_max = fmaxf(local_max, normalized_float);
    }
  }
  values[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      values[threadIdx.x] =
          fmaxf(values[threadIdx.x], values[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    values[0] = fmaxf(values[0] / kFp8Max, kMinScale);
    output_scale[0] = values[0];
  }
  __syncthreads();
  float inverse_scale = 1.0f / values[0];
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = __bfloat162float(normalized[column]) * inverse_scale;
    value = fminf(kFp8Max, fmaxf(-kFp8Max, value));
    packed[column] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
}

__device__ __forceinline__ float swiglu_value(const __nv_bfloat16 *gate_up,
                                              int64_t row, int64_t column,
                                              int64_t columns) {
  int64_t row_offset = row * 2 * columns;
  float gate = __bfloat162float(gate_up[row_offset + column]);
  float up = __bfloat162float(gate_up[row_offset + columns + column]);
  return (gate / (1.0f + __expf(-gate))) * up;
}

__global__ void swiglu_to_fp8_with_scale_kernel(const __nv_bfloat16 *gate_up,
                                                uint8_t *output,
                                                const float *scale,
                                                int64_t rows, int64_t columns) {
  int64_t elements = rows * columns;
  float inverse_scale = 1.0f / scale[0];
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t row = index / columns;
    int64_t column = index - row * columns;
    __nv_bfloat16 rounded =
        __float2bfloat16_rn(swiglu_value(gate_up, row, column, columns));
    float value = __bfloat162float(rounded) * inverse_scale;
    value = fminf(kFp8Max, fmaxf(-kFp8Max, value));
    output[index] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
}

__global__ void swiglu_to_fp8_single_row_kernel(const __nv_bfloat16 *gate_up,
                                                uint8_t *output,
                                                float *output_scale,
                                                int64_t columns) {
  float local_max = 0.0f;
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = swiglu_value(gate_up, 0, column, columns);
    if (isfinite(value)) {
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  __shared__ float block_max[256];
  block_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_max[threadIdx.x] =
          fmaxf(block_max[threadIdx.x], block_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    block_max[0] = fmaxf(block_max[0] / kFp8Max, kMinScale);
    output_scale[0] = block_max[0];
  }
  __syncthreads();
  float inverse_scale = 1.0f / block_max[0];
  for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) {
    float value = swiglu_value(gate_up, 0, column, columns) * inverse_scale;
    value = fminf(kFp8Max, fmaxf(-kFp8Max, value));
    output[column] = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
}

__global__ void swiglu_to_bf16_amax_kernel(const __nv_bfloat16 *gate_up,
                                           __nv_bfloat16 *activated,
                                           float *output_scale, int64_t rows,
                                           int64_t columns) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  int64_t row_offset = row * 2 * columns;
  int64_t output_offset = row * columns;
  float local_max = 0.0f;
  int64_t pairs = columns / 2;
  auto gate_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(gate_up + row_offset);
  auto up_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(gate_up + row_offset + columns);
  auto output_pairs =
      reinterpret_cast<__nv_bfloat162 *>(activated + output_offset);
  for (int64_t pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
    float2 gate = __bfloat1622float2(gate_pairs[pair]);
    float2 up = __bfloat1622float2(up_pairs[pair]);
    float value0 = (gate.x / (1.0f + __expf(-gate.x))) * up.x;
    float value1 = (gate.y / (1.0f + __expf(-gate.y))) * up.y;
    __nv_bfloat162 rounded = __floats2bfloat162_rn(value0, value1);
    output_pairs[pair] = rounded;
    float2 rounded_values = __bfloat1622float2(rounded);
    if (isfinite(rounded_values.x)) {
      local_max = fmaxf(local_max, fabsf(rounded_values.x));
    }
    if (isfinite(rounded_values.y)) {
      local_max = fmaxf(local_max, fabsf(rounded_values.y));
    }
  }
  if ((columns & 1) != 0 && threadIdx.x == 0) {
    int64_t column = columns - 1;
    __nv_bfloat16 rounded =
        __float2bfloat16_rn(swiglu_value(gate_up, row, column, columns));
    activated[output_offset + column] = rounded;
    float value = __bfloat162float(rounded);
    if (isfinite(value)) {
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  __shared__ float block_max[256];
  block_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_max[threadIdx.x] =
          fmaxf(block_max[threadIdx.x], block_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<unsigned int *>(output_scale),
              __float_as_uint(block_max[0]));
  }
}

std::vector<at::Tensor> quantize_bf16_impl(const at::Tensor &input) {
  check_bf16_cuda_contiguous(input, "input");
  TORCH_CHECK(input.numel() > 0, "input must not be empty");
  c10::cuda::CUDAGuard guard(input.device());
  auto fp8 = at::empty(input.sizes(),
                       input.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto scale = at::empty({}, input.options().dtype(at::kFloat));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

  constexpr int threads = 256;
  if (input.dim() > 0 && input.numel() == input.size(-1)) {
    bf16_to_fp8_single_row_kernel<<<1, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()),
        static_cast<uint8_t *>(fp8.data_ptr()), scale.data_ptr<float>(),
        input.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {fp8, scale};
  }

  C10_CUDA_CHECK(
      cudaMemsetAsync(scale.data_ptr<float>(), 0, sizeof(float), stream));
  int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div_int64(input.numel(), threads), 4096));
  bf16_amax_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()),
      scale.data_ptr<float>(), input.numel());
  finalize_scale_kernel<<<1, 1, 0, stream>>>(scale.data_ptr<float>());
  bf16_to_fp8_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()),
      static_cast<uint8_t *>(fp8.data_ptr()), scale.data_ptr<float>(),
      input.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {fp8, scale};
}

std::vector<at::Tensor> quantize_swiglu_bf16_impl(const at::Tensor &gate_up,
                                                  int64_t intermediate_size) {
  check_bf16_cuda_contiguous(gate_up, "gate_up");
  TORCH_CHECK(gate_up.dim() >= 2, "gate_up must have at least two dimensions");
  TORCH_CHECK(intermediate_size > 0, "intermediate_size must be positive");
  TORCH_CHECK(gate_up.size(-1) == 2 * intermediate_size,
              "gate_up last dimension must equal 2 * intermediate_size");
  int64_t rows = gate_up.numel() / gate_up.size(-1);
  TORCH_CHECK(rows > 0, "gate_up must not be empty");

  std::vector<int64_t> output_shape(gate_up.sizes().begin(),
                                    gate_up.sizes().end());
  output_shape.back() = intermediate_size;
  auto fp8 = at::empty(output_shape,
                       gate_up.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto scale = at::empty({}, gate_up.options().dtype(at::kFloat));
  c10::cuda::CUDAGuard guard(gate_up.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());
  constexpr int threads = 256;
  auto input_ptr =
      reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr<at::BFloat16>());
  auto output_ptr = static_cast<uint8_t *>(fp8.data_ptr());
  auto scale_ptr = scale.data_ptr<float>();
  if (rows == 1) {
    swiglu_to_fp8_single_row_kernel<<<1, threads, 0, stream>>>(
        input_ptr, output_ptr, scale_ptr, intermediate_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {fp8, scale};
  }

  auto activated = at::empty(output_shape, gate_up.options());
  int64_t elements = rows * intermediate_size;
  int blocks = static_cast<int>(rows);
  C10_CUDA_CHECK(cudaMemsetAsync(scale_ptr, 0, sizeof(float), stream));
  swiglu_to_bf16_amax_kernel<<<blocks, threads, 0, stream>>>(
      input_ptr,
      reinterpret_cast<__nv_bfloat16 *>(activated.data_ptr<at::BFloat16>()),
      scale_ptr, rows, intermediate_size);
  finalize_scale_kernel<<<1, 1, 0, stream>>>(scale_ptr);
  bf16_to_fp8_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          activated.data_ptr<at::BFloat16>()),
      output_ptr, scale_ptr, elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {fp8, scale};
}

namespace fp8_gemm {

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentCD = 8;

template <typename MmaTileShape_, typename ClusterShape_,
          typename EpilogueSchedule_, typename KernelSchedule_>
struct GemmSpec {
  using MmaTileShape = MmaTileShape_;
  using ClusterShape = ClusterShape_;
  using EpilogueSchedule = EpilogueSchedule_;
  using KernelSchedule = KernelSchedule_;
  using FusionOperation =
      cutlass::epilogue::fusion::ScaledLinCombPerColBiasEltAct<
          cutlass::epilogue::thread::Identity, ElementD, ElementCompute,
          ElementD, ElementC>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutC, AlignmentCD, ElementD, LayoutD,
          AlignmentCD, EpilogueSchedule, FusionOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB,
          LayoutB, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using DecodeSpec = GemmSpec<Shape<_128, _128, _64>, Shape<_1, _1, _1>,
                            cutlass::epilogue::TmaWarpSpecialized1Sm,
                            cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;

using DecodeWideSpec =
    GemmSpec<Shape<_128, _256, _128>, Shape<_1, _1, _1>,
             cutlass::epilogue::NoSmemWarpSpecialized1Sm,
             cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;

using PrefillSpec = GemmSpec<Shape<_256, _128, _64>, Shape<_2, _1, _1>,
                             cutlass::epilogue::TmaWarpSpecialized2Sm,
                             cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>;

template <typename ClusterShape_, typename EpilogueSchedule_,
          typename KernelSchedule_>
struct DecodeSwapABGemmSpec {
  using MmaTileShape = Shape<_128, _32, _128>;
  using ClusterShape = ClusterShape_;
  using EpilogueSchedule = EpilogueSchedule_;
  using KernelSchedule = KernelSchedule_;
  using LayoutATranspose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutBTranspose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  using LayoutCTranspose =
      typename cutlass::layout::LayoutTranspose<LayoutC>::type;
  using LayoutDTranspose =
      typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  using FusionOperation =
      cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltAct<
          cutlass::epilogue::thread::Identity, ElementD, ElementCompute,
          ElementD, ElementC>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutCTranspose, AlignmentCD, ElementD,
          LayoutDTranspose, AlignmentCD, EpilogueSchedule,
          FusionOperation>::CollectiveOp;

  // Compute (A W^T)^T = W A^T. Original B becomes row-major operand A,
  // while original A becomes column-major operand B.
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementB, LayoutBTranspose, AlignmentB,
          ElementA, LayoutATranspose, AlignmentA, ElementAccumulator,
          MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using DecodeSwapABSpec =
    DecodeSwapABGemmSpec<Shape<_4, _1, _1>,
                         cutlass::epilogue::collective::EpilogueScheduleAuto,
                         cutlass::gemm::collective::KernelScheduleAuto>;
using DecodeSwapABCluster2Spec =
    DecodeSwapABGemmSpec<Shape<_2, _1, _1>,
                         cutlass::epilogue::collective::EpilogueScheduleAuto,
                         cutlass::gemm::collective::KernelScheduleAuto>;
using DecodeSwapABCluster1Spec =
    DecodeSwapABGemmSpec<Shape<_1, _1, _1>,
                         cutlass::epilogue::collective::EpilogueScheduleAuto,
                         cutlass::gemm::collective::KernelScheduleAuto>;
using DecodeSwapABNoSmemSpec =
    DecodeSwapABGemmSpec<Shape<_4, _1, _1>,
                         cutlass::epilogue::NoSmemWarpSpecialized1Sm,
                         cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;
using DecodeSwapABCluster2NoSmemSpec =
    DecodeSwapABGemmSpec<Shape<_2, _1, _1>,
                         cutlass::epilogue::NoSmemWarpSpecialized1Sm,
                         cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;
using DecodeSwapABCluster1NoSmemSpec =
    DecodeSwapABGemmSpec<Shape<_1, _1, _1>,
                         cutlass::epilogue::NoSmemWarpSpecialized1Sm,
                         cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;

template <typename Spec>
void run(at::Tensor &output, const at::Tensor &input, const at::Tensor &weight,
         const at::Tensor &input_scale, const at::Tensor &weight_scale,
         const at::Tensor *bias, int64_t m, int64_t n, int64_t k,
         cudaStream_t stream) {
  using Gemm = typename Spec::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  int mi = static_cast<int>(m);
  int ni = static_cast<int>(n);
  int ki = static_cast<int>(k);
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {mi, ki, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {ni, ki, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {mi, ni, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {mi, ni, 1});

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {mi, ni, ki, 1},
      {reinterpret_cast<const ElementA *>(input.data_ptr()), stride_a,
       reinterpret_cast<const ElementB *>(weight.data_ptr()), stride_b},
      {{},
       reinterpret_cast<const ElementC *>(output.data_ptr<at::BFloat16>()),
       stride_c,
       reinterpret_cast<ElementD *>(output.data_ptr<at::BFloat16>()),
       stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  arguments.epilogue.thread.scale_a_ptr = input_scale.data_ptr<float>();
  arguments.epilogue.thread.scale_b_ptr = weight_scale.data_ptr<float>();
  arguments.epilogue.thread.bias_ptr =
      bias == nullptr
          ? nullptr
          : reinterpret_cast<const ElementD *>(bias->data_ptr<at::BFloat16>());

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP8 GEMM cannot implement shape M=", m, ", N=", n,
              ", K=", k);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void *workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          input.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP8 GEMM initialization failed");
  status = gemm.run(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP8 GEMM launch failed");
}

template <typename Spec>
void run_swap_ab(at::Tensor &output, const at::Tensor &input,
                 const at::Tensor &weight, const at::Tensor &input_scale,
                 const at::Tensor &weight_scale, const at::Tensor *bias,
                 int64_t m, int64_t n, int64_t k, cudaStream_t stream) {
  using Gemm = typename Spec::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  int mi = static_cast<int>(m);
  int ni = static_cast<int>(n);
  int ki = static_cast<int>(k);
  auto weight_stride = cutlass::make_cute_packed_stride(StrideA{}, {ni, ki, 1});
  auto input_stride = cutlass::make_cute_packed_stride(StrideB{}, {mi, ki, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {ni, mi, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {ni, mi, 1});

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {ni, mi, ki, 1},
      {reinterpret_cast<const ElementB *>(weight.data_ptr()), weight_stride,
       reinterpret_cast<const ElementA *>(input.data_ptr()), input_stride},
      {{},
       reinterpret_cast<const ElementC *>(output.data_ptr<at::BFloat16>()),
       stride_c,
       reinterpret_cast<ElementD *>(output.data_ptr<at::BFloat16>()),
       stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  arguments.epilogue.thread.scale_a_ptr = weight_scale.data_ptr<float>();
  arguments.epilogue.thread.scale_b_ptr = input_scale.data_ptr<float>();
  arguments.epilogue.thread.bias_ptr =
      bias == nullptr
          ? nullptr
          : reinterpret_cast<const ElementD *>(bias->data_ptr<at::BFloat16>());

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP8 Swap-AB GEMM cannot implement shape M=", m,
              ", N=", n, ", K=", k);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void *workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          input.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP8 Swap-AB GEMM initialization failed");
  status = gemm.run(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP8 Swap-AB GEMM launch failed");
}

} // namespace fp8_gemm

namespace fp8_producer_gemm {

using namespace cute;

using ElementA = cutlass::bfloat16_t;
using ElementAMma = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentA = 8;
constexpr int AlignmentB = 16;
constexpr int AlignmentCD = 8;

template <typename MmaTileShape_, typename ClusterShape_,
          typename EpilogueSchedule_, typename KernelSchedule_>
struct GemmSpec {
  using MmaTileShape = MmaTileShape_;
  using ClusterShape = ClusterShape_;
  using EpilogueSchedule = EpilogueSchedule_;
  using KernelSchedule = KernelSchedule_;
  using FusionOperation =
      cutlass::epilogue::fusion::ScaledLinCombPerColBiasEltAct<
          cutlass::epilogue::thread::Identity, ElementD, ElementCompute,
          ElementD, ElementC>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementC, LayoutC, AlignmentCD, ElementD, LayoutD,
          AlignmentCD, EpilogueSchedule, FusionOperation>::CollectiveOp;

  // An explicit one-element tuple selects CUTLASS's mixed-input transform
  // pipeline. The producer loads BF16 A, converts it to E4M3 in registers,
  // and writes the converted tile directly to the UMMA operand buffer.
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, cute::tuple<ElementA>, LayoutA, AlignmentA,
          ElementB, LayoutB, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape, cutlass::gemm::collective::StageCount<2>,
          KernelSchedule>::CollectiveOp;

  static_assert(
      cute::is_same_v<typename CollectiveMainloop::ElementAMma, ElementAMma>,
      "mixed-input producer must feed E4M3 to UMMA");

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using DecodeSpec =
    GemmSpec<Shape<_128, _128, _64>, Shape<_1, _1, _1>,
             cutlass::epilogue::TmaWarpSpecialized1Sm,
             cutlass::gemm::KernelTmaWarpSpecialized1SmMixedInputSm100>;

using PrefillSpec =
    GemmSpec<Shape<_256, _128, _64>, Shape<_2, _1, _1>,
             cutlass::epilogue::TmaWarpSpecialized2Sm,
             cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100>;

template <typename Spec>
void run(at::Tensor &output, const at::Tensor &input, const at::Tensor &weight,
         const at::Tensor &weight_scale, const at::Tensor *bias, int64_t m,
         int64_t n, int64_t k, cudaStream_t stream) {
  using Gemm = typename Spec::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  int mi = static_cast<int>(m);
  int ni = static_cast<int>(n);
  int ki = static_cast<int>(k);
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {mi, ki, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {ni, ki, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {mi, ni, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {mi, ni, 1});

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {mi, ni, ki, 1},
      {reinterpret_cast<const ElementA *>(input.data_ptr<at::BFloat16>()),
       stride_a, reinterpret_cast<const ElementB *>(weight.data_ptr()),
       stride_b},
      {{},
       reinterpret_cast<const ElementC *>(output.data_ptr<at::BFloat16>()),
       stride_c,
       reinterpret_cast<ElementD *>(output.data_ptr<at::BFloat16>()),
       stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  arguments.epilogue.thread.scale_a = 1.0f;
  arguments.epilogue.thread.scale_a_ptr = nullptr;
  arguments.epilogue.thread.scale_b_ptr = weight_scale.data_ptr<float>();
  arguments.epilogue.thread.bias_ptr =
      bias == nullptr
          ? nullptr
          : reinterpret_cast<const ElementD *>(bias->data_ptr<at::BFloat16>());

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS BF16-to-FP8 producer GEMM cannot implement shape M=", m,
              ", N=", n, ", K=", k);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void *workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          input.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS BF16-to-FP8 producer GEMM initialization failed");
  status = gemm.run(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS BF16-to-FP8 producer GEMM launch failed");
}

} // namespace fp8_producer_gemm

bool supports_device(int64_t cc) { return cc >= 100; }

void set_decode_policy(int64_t policy) {
  TORCH_CHECK(policy >= kDecodePolicyBalanced && policy <= kDecodePolicyLegacy,
              "FP8 decode policy must be 0 (balanced), 1 (low_clock), "
              "or 2 (legacy), got ",
              policy);
  g_decode_policy.store(policy, std::memory_order_relaxed);
}

int64_t get_decode_policy() {
  return g_decode_policy.load(std::memory_order_relaxed);
}

void set_swap_ab_enabled(bool enabled) {
  g_swap_ab_enabled.store(enabled, std::memory_order_relaxed);
}

bool get_swap_ab_enabled() {
  return g_swap_ab_enabled.load(std::memory_order_relaxed);
}

std::vector<at::Tensor> quantize_bf16(const at::Tensor &input) {
  auto contiguous = input.contiguous();
  return quantize_bf16_impl(contiguous);
}

at::Tensor quantize_bf16_with_scale(const at::Tensor &input,
                                    const at::Tensor &scale) {
  auto contiguous = input.contiguous();
  check_bf16_cuda_contiguous(contiguous, "input");
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == at::kFloat &&
                  scale.numel() == 1,
              "scale must be a one-element CUDA float32 tensor");
  TORCH_CHECK(contiguous.device() == scale.device(),
              "input and scale must share a CUDA device");
  TORCH_CHECK(contiguous.numel() > 0, "input must not be empty");

  auto fp8 =
      at::empty(contiguous.sizes(),
                contiguous.options().dtype(at::ScalarType::Float8_e4m3fn));
  c10::cuda::CUDAGuard guard(contiguous.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(contiguous.get_device());
  constexpr int threads = 256;
  int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div_int64(contiguous.numel(), threads), 4096));
  bf16_to_fp8_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          contiguous.data_ptr<at::BFloat16>()),
      static_cast<uint8_t *>(fp8.data_ptr()), scale.data_ptr<float>(),
      contiguous.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return fp8;
}

at::Tensor quantize_swiglu_bf16_with_scale(const at::Tensor &gate_up,
                                           int64_t intermediate_size,
                                           const at::Tensor &scale) {
  check_bf16_cuda_contiguous(gate_up, "gate_up");
  TORCH_CHECK(gate_up.dim() >= 2, "gate_up must have at least two dimensions");
  TORCH_CHECK(intermediate_size > 0, "intermediate_size must be positive");
  TORCH_CHECK(gate_up.size(-1) == 2 * intermediate_size,
              "gate_up last dimension must equal 2 * intermediate_size");
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == at::kFloat &&
                  scale.numel() == 1,
              "scale must be a one-element CUDA float32 tensor");
  TORCH_CHECK(gate_up.device() == scale.device(),
              "gate_up and scale must share a CUDA device");
  int64_t rows = gate_up.numel() / gate_up.size(-1);
  TORCH_CHECK(rows > 0, "gate_up must not be empty");

  std::vector<int64_t> output_shape(gate_up.sizes().begin(),
                                    gate_up.sizes().end());
  output_shape.back() = intermediate_size;
  auto fp8 = at::empty(output_shape,
                       gate_up.options().dtype(at::ScalarType::Float8_e4m3fn));
  c10::cuda::CUDAGuard guard(gate_up.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());
  constexpr int threads = 256;
  int64_t elements = rows * intermediate_size;
  int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div_int64(elements, threads), 4096));
  swiglu_to_fp8_with_scale_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr<at::BFloat16>()),
      static_cast<uint8_t *>(fp8.data_ptr()), scale.data_ptr<float>(), rows,
      intermediate_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return fp8;
}

std::vector<at::Tensor> add_rms_norm_quantize_bf16(const at::Tensor &input,
                                                   const at::Tensor &residual,
                                                   const at::Tensor &weight,
                                                   double epsilon) {
  check_bf16_cuda_contiguous(input, "input");
  check_bf16_cuda_contiguous(residual, "residual");
  check_bf16_cuda_contiguous(weight, "weight");
  TORCH_CHECK(input.sizes() == residual.sizes(),
              "input and residual shapes must match");
  TORCH_CHECK(weight.dim() == 1 && input.size(-1) == weight.numel(),
              "RMSNorm weight must match the input hidden dimension");
  TORCH_CHECK(input.device() == residual.device() &&
                  input.device() == weight.device(),
              "input, residual, and weight must share a CUDA device");
  TORCH_CHECK(input.numel() > 0, "input must not be empty");

  int64_t columns = input.size(-1);
  int64_t rows = input.numel() / columns;
  auto summed = at::empty_like(input);
  auto normalized = at::empty_like(input);
  auto packed = at::empty(input.sizes(),
                          input.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto scale = at::empty({}, input.options().dtype(at::kFloat));
  c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  constexpr int threads = 256;
  if (rows == 1) {
    add_rms_norm_quant_single_row_bf16_kernel<<<1, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(
            residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(
            weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(summed.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(normalized.data_ptr<at::BFloat16>()),
        static_cast<uint8_t *>(packed.data_ptr()), scale.data_ptr<float>(),
        columns, static_cast<float>(epsilon));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {summed, normalized, packed, scale};
  }

  C10_CUDA_CHECK(
      cudaMemsetAsync(scale.data_ptr<float>(), 0, sizeof(float), stream));
  add_rms_norm_amax_bf16_kernel<<<static_cast<int>(rows), threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(
          residual.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(summed.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16 *>(normalized.data_ptr<at::BFloat16>()),
      scale.data_ptr<float>(), columns, static_cast<float>(epsilon));
  finalize_scale_kernel<<<1, 1, 0, stream>>>(scale.data_ptr<float>());
  int blocks = static_cast<int>(
      std::min<int64_t>(ceil_div_int64(input.numel(), threads), 4096));
  bf16_to_fp8_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(
          normalized.data_ptr<at::BFloat16>()),
      static_cast<uint8_t *>(packed.data_ptr()), scale.data_ptr<float>(),
      input.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {summed, normalized, packed, scale};
}

std::vector<at::Tensor> pack_linear_weight(const at::Tensor &weight,
                                           const std::string &role,
                                           const std::string &target_name) {
  (void)role;
  (void)target_name;
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
  auto contiguous = weight.contiguous();
  auto packed = quantize_bf16_impl(contiguous);
  auto metadata =
      at::empty({2}, weight.options().device(at::kCPU).dtype(at::kLong));
  metadata.data_ptr<int64_t>()[0] = weight.size(0);
  metadata.data_ptr<int64_t>()[1] = weight.size(1);
  return {packed[0], packed[1], metadata};
}

at::Tensor linear_forward_packed_tactic(const at::Tensor &input,
                                        const at::Tensor &input_scale,
                                        const at::Tensor &weight,
                                        const at::Tensor &weight_scale,
                                        const std::optional<at::Tensor> &bias,
                                        int64_t tactic) {
  check_fp8_cuda_contiguous(input, "input");
  check_fp8_cuda_contiguous(weight, "weight");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
  TORCH_CHECK(input.dim() >= 2, "input must have at least two dimensions");
  TORCH_CHECK(input.size(-1) == weight.size(1),
              "input and weight K dimensions must match");
  TORCH_CHECK(input.device() == weight.device() &&
                  input.device() == input_scale.device() &&
                  input.device() == weight_scale.device(),
              "input, weight, and scales must be on the same CUDA device");
  TORCH_CHECK(weight.size(0) % 16 == 0 && weight.size(1) % 16 == 0,
              "FP8 CUTLASS GEMM requires N and K divisible by 16");
  TORCH_CHECK(input_scale.is_cuda() &&
                  input_scale.scalar_type() == at::kFloat &&
                  input_scale.numel() == 1,
              "input_scale must be a one-element CUDA float32 tensor");
  TORCH_CHECK(weight_scale.is_cuda() &&
                  weight_scale.scalar_type() == at::kFloat &&
                  weight_scale.numel() == 1,
              "weight_scale must be a one-element CUDA float32 tensor");
  const at::Tensor *bias_ptr = nullptr;
  if (bias.has_value()) {
    check_bf16_cuda_contiguous(*bias, "bias");
    TORCH_CHECK(bias->dim() == 1 && bias->numel() == weight.size(0),
                "bias must have shape [N]");
    TORCH_CHECK(bias->device() == input.device(),
                "bias must be on the same CUDA device as input");
    bias_ptr = &*bias;
  }

  int64_t k = weight.size(1);
  int64_t n = weight.size(0);
  int64_t m = input.numel() / k;
  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  output_shape.back() = n;
  auto output = at::empty(output_shape, input.options().dtype(at::kBFloat16));
  c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (tactic == 0) {
    int64_t policy = get_decode_policy();
    if (get_swap_ab_enabled() && policy != kDecodePolicyLegacy && m <= 16 &&
        n >= 2048 && k >= 2048 && k < 8192 && n < 65536) {
      if (n >= 16384) {
        tactic = policy == kDecodePolicyLowClock ? 10 : 7;
      } else {
        tactic = 11;
      }
    } else if (policy == kDecodePolicyLegacy) {
      tactic = m <= 8 && n < 8192 && k < 8192 ? 1 : 2;
    } else if (m > 8) {
      tactic = 2;
    } else if (policy == kDecodePolicyLowClock) {
      // At 315 MHz, the wider one-SM tile amortizes launch and mainloop
      // overhead. Tiny K/V projections retain the two-SM tile.
      tactic = n < 1024 ? 2 : 6;
    } else {
      // At full clock, only the vocabulary projection benefits from two SMs.
      tactic = n >= 65536 ? 2 : 6;
    }
  }
  if (tactic == 1) {
    fp8_gemm::run<fp8_gemm::DecodeSpec>(output, input, weight, input_scale,
                                        weight_scale, bias_ptr, m, n, k,
                                        stream);
  } else if (tactic == 6) {
    fp8_gemm::run<fp8_gemm::DecodeWideSpec>(output, input, weight, input_scale,
                                            weight_scale, bias_ptr, m, n, k,
                                            stream);
  } else if (tactic == 2) {
    fp8_gemm::run<fp8_gemm::PrefillSpec>(output, input, weight, input_scale,
                                         weight_scale, bias_ptr, m, n, k,
                                         stream);
  } else if (tactic == 7) {
    TORCH_CHECK(m >= 1 && m <= 16,
                "FP8 Swap-AB tactic requires M in [1, 16], got ", m);
    fp8_gemm::run_swap_ab<fp8_gemm::DecodeSwapABSpec>(
        output, input, weight, input_scale, weight_scale, bias_ptr, m, n, k,
        stream);
  } else if (tactic == 8) {
    TORCH_CHECK(m >= 1 && m <= 16,
                "FP8 Swap-AB tactic requires M in [1, 16], got ", m);
    fp8_gemm::run_swap_ab<fp8_gemm::DecodeSwapABCluster2Spec>(
        output, input, weight, input_scale, weight_scale, bias_ptr, m, n, k,
        stream);
  } else if (tactic == 9) {
    TORCH_CHECK(m >= 1 && m <= 16,
                "FP8 Swap-AB tactic requires M in [1, 16], got ", m);
    fp8_gemm::run_swap_ab<fp8_gemm::DecodeSwapABCluster1Spec>(
        output, input, weight, input_scale, weight_scale, bias_ptr, m, n, k,
        stream);
  } else if (tactic == 10) {
    TORCH_CHECK(m >= 1 && m <= 16,
                "FP8 Swap-AB tactic requires M in [1, 16], got ", m);
    fp8_gemm::run_swap_ab<fp8_gemm::DecodeSwapABNoSmemSpec>(
        output, input, weight, input_scale, weight_scale, bias_ptr, m, n, k,
        stream);
  } else if (tactic == 11) {
    TORCH_CHECK(m >= 1 && m <= 16,
                "FP8 Swap-AB tactic requires M in [1, 16], got ", m);
    fp8_gemm::run_swap_ab<fp8_gemm::DecodeSwapABCluster2NoSmemSpec>(
        output, input, weight, input_scale, weight_scale, bias_ptr, m, n, k,
        stream);
  } else if (tactic == 12) {
    TORCH_CHECK(m >= 1 && m <= 16,
                "FP8 Swap-AB tactic requires M in [1, 16], got ", m);
    fp8_gemm::run_swap_ab<fp8_gemm::DecodeSwapABCluster1NoSmemSpec>(
        output, input, weight, input_scale, weight_scale, bias_ptr, m, n, k,
        stream);
  } else {
    TORCH_CHECK(false, "unknown FP8 GEMM tactic ", tactic);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor linear_forward_packed(const at::Tensor &input,
                                 const at::Tensor &input_scale,
                                 const at::Tensor &weight,
                                 const at::Tensor &weight_scale,
                                 const std::optional<at::Tensor> &bias) {
  return linear_forward_packed_tactic(input, input_scale, weight, weight_scale,
                                      bias, 0);
}

at::Tensor linear_forward_packed_greedy(
    const at::Tensor &input, const at::Tensor &input_scale,
    const at::Tensor &weight, const at::Tensor &weight_scale,
    const std::optional<at::Tensor> &bias, at::Tensor seen_tokens,
    at::Tensor eos_tokens, at::Tensor generated_count,
    double repetition_penalty, int64_t min_new_tokens) {
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 1,
              "fused greedy lm_head requires one input row");
  int64_t vocab_size = weight.size(0);
  TORCH_CHECK(vocab_size > 0 && vocab_size <= INT_MAX,
              "vocabulary size is out of range");
  TORCH_CHECK(seen_tokens.is_cuda() && eos_tokens.is_cuda() &&
                  generated_count.is_cuda(),
              "sampler state must be CUDA");
  TORCH_CHECK(seen_tokens.device() == input.device() &&
                  eos_tokens.device() == input.device() &&
                  generated_count.device() == input.device(),
              "sampler state and input must share a CUDA device");
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

  at::Tensor logits =
      linear_forward_packed(input, input_scale, weight, weight_scale, bias);
  at::Tensor output = at::empty({1}, input.options().dtype(at::kLong));
  constexpr int kThreads = 256;
  int partial_count = static_cast<int>(
      std::min<int64_t>(256, ceil_div_int64(vocab_size, kThreads)));
  at::Tensor partial_scores =
      at::empty({partial_count}, input.options().dtype(at::kFloat));
  at::Tensor partial_indices =
      at::empty({partial_count}, input.options().dtype(at::kInt));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
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

std::vector<at::Tensor> linear_forward_packed_topk(
    const at::Tensor &input, const at::Tensor &input_scale,
    const at::Tensor &weight, const at::Tensor &weight_scale,
    const std::optional<at::Tensor> &bias, at::Tensor seen_tokens,
    at::Tensor eos_tokens, at::Tensor generated_count,
    double repetition_penalty, int64_t min_new_tokens, int64_t top_k) {
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 1,
              "fused top-k lm_head requires one input row");
  int64_t vocab_size = weight.size(0);
  TORCH_CHECK(top_k > 0 && top_k <= vocab_size,
              "top_k must be in [1, vocab_size]");
  TORCH_CHECK(seen_tokens.is_cuda() && eos_tokens.is_cuda() &&
                  generated_count.is_cuda(),
              "sampler state must be CUDA");
  TORCH_CHECK(seen_tokens.device() == input.device() &&
                  eos_tokens.device() == input.device() &&
                  generated_count.device() == input.device(),
              "sampler state and input must share a CUDA device");
  TORCH_CHECK(seen_tokens.scalar_type() == at::kByte &&
                  eos_tokens.scalar_type() == at::kByte &&
                  generated_count.scalar_type() == at::kInt,
              "sampler state has an invalid dtype");
  TORCH_CHECK(seen_tokens.numel() == vocab_size &&
                  eos_tokens.numel() == vocab_size &&
                  generated_count.numel() == 1,
              "sampler state has an invalid size");

  at::Tensor logits =
      linear_forward_packed(input, input_scale, weight, weight_scale, bias);
  at::Tensor processed =
      at::empty({1, vocab_size}, input.options().dtype(at::kFloat));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
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

at::Tensor linear_forward(const at::Tensor &input, const at::Tensor &weight,
                          const at::Tensor &weight_scale,
                          const std::optional<at::Tensor> &bias) {
  auto contiguous = input.contiguous();
  auto packed = quantize_bf16_impl(contiguous);
  return linear_forward_packed(packed[0], packed[1], weight, weight_scale,
                               bias);
}

at::Tensor linear_forward_producer(const at::Tensor &input,
                                   const at::Tensor &weight,
                                   const at::Tensor &weight_scale,
                                   const std::optional<at::Tensor> &bias) {
  check_bf16_cuda_contiguous(input, "input");
  check_fp8_cuda_contiguous(weight, "weight");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
  TORCH_CHECK(input.dim() >= 2, "input must have at least two dimensions");
  TORCH_CHECK(input.size(-1) == weight.size(1),
              "input and weight K dimensions must match");
  TORCH_CHECK(input.device() == weight.device() &&
                  input.device() == weight_scale.device(),
              "input, weight, and weight_scale must share a CUDA device");
  TORCH_CHECK(weight.size(0) % 16 == 0 && weight.size(1) % 16 == 0,
              "FP8 CUTLASS GEMM requires N and K divisible by 16");
  TORCH_CHECK(weight_scale.is_cuda() &&
                  weight_scale.scalar_type() == at::kFloat &&
                  weight_scale.numel() == 1,
              "weight_scale must be a one-element CUDA float32 tensor");
  const at::Tensor *bias_ptr = nullptr;
  if (bias.has_value()) {
    check_bf16_cuda_contiguous(*bias, "bias");
    TORCH_CHECK(bias->dim() == 1 && bias->numel() == weight.size(0),
                "bias must have shape [N]");
    TORCH_CHECK(bias->device() == input.device(),
                "bias must be on the same CUDA device as input");
    bias_ptr = &*bias;
  }

  int64_t k = weight.size(1);
  int64_t n = weight.size(0);
  int64_t m = input.numel() / k;
  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  output_shape.back() = n;
  auto output = at::empty(output_shape, input.options());
  c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  bool use_decode_tactic = m <= 8 && n < 8192 && k < 8192;
  if (use_decode_tactic) {
    fp8_producer_gemm::run<fp8_producer_gemm::DecodeSpec>(
        output, input, weight, weight_scale, bias_ptr, m, n, k, stream);
  } else {
    fp8_producer_gemm::run<fp8_producer_gemm::PrefillSpec>(
        output, input, weight, weight_scale, bias_ptr, m, n, k, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace

TORCH_LIBRARY(robotics_cutlass_fp8, m) {
  m.def("supports_device(int cc) -> bool");
  m.def("set_decode_policy(int policy) -> ()");
  m.def("get_decode_policy() -> int");
  m.def("set_swap_ab_enabled(bool enabled) -> ()");
  m.def("get_swap_ab_enabled() -> bool");
  m.def("quantize_bf16(Tensor input) -> Tensor[]");
  m.def("quantize_bf16_with_scale(Tensor input, Tensor scale) -> Tensor");
  m.def("add_rms_norm_quantize_bf16(Tensor input, Tensor residual, "
        "Tensor weight, float epsilon) -> Tensor[]");
  m.def("quantize_swiglu_bf16(Tensor gate_up, int intermediate_size) -> "
        "Tensor[]");
  m.def("quantize_swiglu_bf16_with_scale(Tensor gate_up, "
        "int intermediate_size, Tensor scale) -> Tensor");
  m.def("pack_linear_weight(Tensor weight, str role, str target_name) -> "
        "Tensor[]");
  m.def("linear_forward(Tensor input, Tensor weight, Tensor weight_scale, "
        "Tensor? bias) -> Tensor");
  m.def("linear_forward_producer(Tensor input, Tensor weight, "
        "Tensor weight_scale, Tensor? bias) -> Tensor");
  m.def(
      "linear_forward_packed(Tensor input, Tensor input_scale, Tensor weight, "
      "Tensor weight_scale, Tensor? bias) -> Tensor");
  m.def("linear_forward_packed_greedy(Tensor input, Tensor input_scale, "
        "Tensor weight, Tensor weight_scale, Tensor? bias, "
        "Tensor seen_tokens, Tensor eos_tokens, Tensor generated_count, "
        "float repetition_penalty, int min_new_tokens) -> Tensor");
  m.def("linear_forward_packed_topk(Tensor input, Tensor input_scale, "
        "Tensor weight, Tensor weight_scale, Tensor? bias, "
        "Tensor seen_tokens, Tensor eos_tokens, Tensor generated_count, "
        "float repetition_penalty, int min_new_tokens, int top_k) -> Tensor[]");
  m.def("linear_forward_packed_tactic(Tensor input, Tensor input_scale, "
        "Tensor weight, Tensor weight_scale, Tensor? bias, int tactic) -> "
        "Tensor");
}

TORCH_LIBRARY_IMPL(robotics_cutlass_fp8, CUDA, m) {
  m.impl("quantize_bf16", &quantize_bf16);
  m.impl("quantize_bf16_with_scale", &quantize_bf16_with_scale);
  m.impl("add_rms_norm_quantize_bf16", &add_rms_norm_quantize_bf16);
  m.impl("quantize_swiglu_bf16", &quantize_swiglu_bf16_impl);
  m.impl("quantize_swiglu_bf16_with_scale", &quantize_swiglu_bf16_with_scale);
  m.impl("pack_linear_weight", &pack_linear_weight);
  m.impl("linear_forward", &linear_forward);
  m.impl("linear_forward_producer", &linear_forward_producer);
  m.impl("linear_forward_packed", &linear_forward_packed);
  m.impl("linear_forward_packed_greedy", &linear_forward_packed_greedy);
  m.impl("linear_forward_packed_topk", &linear_forward_packed_topk);
  m.impl("linear_forward_packed_tactic", &linear_forward_packed_tactic);
}

TORCH_LIBRARY_IMPL(robotics_cutlass_fp8, CatchAll, m) {
  m.impl("supports_device", &supports_device);
  m.impl("set_decode_policy", &set_decode_policy);
  m.impl("get_decode_policy", &get_decode_policy);
  m.impl("set_swap_ab_enabled", &set_swap_ab_enabled);
  m.impl("get_swap_ab_enabled", &get_swap_ab_enabled);
}
