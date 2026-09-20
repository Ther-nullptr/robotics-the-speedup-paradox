// Fused BF16 SwiGLU to CUTLASS MXFP8 block32 activation packing.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

#include <vector>

#include "mxfp8_common.cuh"

namespace {

using F8 = cutlass::float_e4m3_t;

__global__ void silu_mul_pack_mxfp8_kernel(const __nv_bfloat16 *gate_up,
                                           const __nv_bfloat16 *silu_table,
                                           uint32_t *packed, uint8_t *scales,
                                           int64_t rows, int64_t columns) {
  int64_t groups_padded = ((columns + 127) / 128) * 4;
  int64_t thread = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t group_index = thread / 8;
  int64_t row = group_index / groups_padded;
  int64_t group = group_index % groups_padded;
  int lane = threadIdx.x % 8;
  if (row >= ((rows + 127) / 128) * 128) {
    return;
  }
  int64_t sf_index = scale_offset(row, group, groups_padded / 4);
  if (row >= rows || group * 32 >= columns) {
    if (lane == 0) {
      scales[sf_index] = 0;
    }
    return;
  }

  cutlass::Array<float, 4> values;
  float maximum = 0.0f;
  unsigned invalid = 0;
  unsigned mask = __activemask();
  int64_t row_offset = row * 2 * columns;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    int64_t column = group * 32 + lane * 4 + item;
    __nv_bfloat16 up = gate_up[row_offset + columns + column];
    uint16_t gate_bits =
        reinterpret_cast<const uint16_t *>(gate_up)[row_offset + column];
    __nv_bfloat16 activated = __float2bfloat16_rn(
        __bfloat162float(silu_table[gate_bits]) * __bfloat162float(up));
    float value = __bfloat162float(activated);
    values[item] = value;
    maximum = fmaxf(maximum, fabsf(value));
    invalid |= !isfinite(value);
  }
#pragma unroll
  for (int offset = 4; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, offset, 8));
    invalid |= __shfl_xor_sync(mask, invalid, offset, 8);
  }
  SF scale = invalid ? SF::bitcast(255) : block_scale(maximum);
  if (lane == 0) {
    scales[sf_index] = scale.raw();
  }
  float inverse = inverse_scale(scale);
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    values[item] *= inverse;
  }
  auto output = cutlass::NumericArrayConverter<F8, float, 4>{}(values);
  packed[(row * columns + group * 32) / 4 + lane] =
      *reinterpret_cast<const uint32_t *>(output.data());
}

std::vector<at::Tensor> silu_mul_pack_mxfp8(const at::Tensor &gate_up,
                                            int64_t intermediate_size,
                                            const at::Tensor &silu_table) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == at::kBFloat16 &&
                  gate_up.is_contiguous() && gate_up.dim() >= 2,
              "gate_up must be contiguous CUDA BF16");
  TORCH_CHECK(
      intermediate_size > 0 && intermediate_size % 32 == 0 &&
          gate_up.size(-1) == 2 * intermediate_size,
      "gate_up last dimension must be twice a K32-aligned intermediate size");
  TORCH_CHECK(silu_table.is_cuda() && silu_table.device() == gate_up.device() &&
                  silu_table.scalar_type() == at::kBFloat16 &&
                  silu_table.is_contiguous() && silu_table.numel() == 65536,
              "silu_table must be contiguous CUDA BF16 with 65536 entries");

  int64_t rows = gate_up.numel() / gate_up.size(-1);
  std::vector<int64_t> output_shape(gate_up.sizes().begin(),
                                    gate_up.sizes().end());
  output_shape.back() = intermediate_size;
  auto packed = at::empty(
      output_shape, gate_up.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto scales = at::empty({scale_storage_size(rows, intermediate_size)},
                          gate_up.options().dtype(at::kByte));

  int64_t groups_padded = ((intermediate_size + 127) / 128) * 4;
  int64_t groups = ((rows + 127) / 128) * 128 * groups_padded;
  constexpr int threads = 128;
  TORCH_CHECK(groups % 16 == 0, "invalid MXFP8 group launch geometry");
  int blocks = static_cast<int>(groups / 16);
  c10::cuda::CUDAGuard guard(gate_up.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());
  silu_mul_pack_mxfp8_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16 *>(
          silu_table.data_ptr<at::BFloat16>()),
      reinterpret_cast<uint32_t *>(packed.data_ptr()),
      scales.data_ptr<uint8_t>(), rows, intermediate_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}

} // namespace

TORCH_LIBRARY(robotics_mxfp8_swiglu, m) {
  m.def("silu_mul_pack(Tensor gate_up, int intermediate_size, "
        "Tensor silu_table) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(robotics_mxfp8_swiglu, CUDA, m) {
  m.impl("silu_mul_pack", &silu_mul_pack_mxfp8);
}
