// CUTLASS E4M3/E8M0 block32 layout and conversion helpers.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

namespace {

using SF = cutlass::float_ue8m0_t;

int64_t scale_storage_size(int64_t rows, int64_t columns) {
  return ((rows + 127) / 128) * ((columns + 127) / 128) * 512;
}

__host__ __device__ int64_t scale_offset(int64_t row, int64_t group,
                                         int64_t k_tiles) {
  return ((row / 128) * k_tiles + group / 4) * 512 + (row % 32) * 16 +
         ((row % 128) / 32) * 4 + group % 4;
}

__device__ SF block_scale(float amax) {
  float multiplier = cutlass::reciprocal_approximate_ftz<float>{}(448.0f);
  return cutlass::NumericConverter<SF, float>{}(amax * multiplier);
}

__device__ float inverse_scale(SF scale) {
  SF inverse = SF::bitcast(static_cast<uint8_t>(254 - scale.raw()));
  return static_cast<float>(inverse);
}

} // namespace
