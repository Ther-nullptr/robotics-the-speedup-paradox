#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
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
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace {

constexpr int64_t kSfVecSize = 32;

int64_t ceil_div_int64(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

int64_t scale_blocks_k(int64_t k) {
  return ceil_div_int64(ceil_div_int64(k, kSfVecSize), 4);
}

int64_t scale_blocks_mn(int64_t mn) { return ceil_div_int64(mn, 128); }

__device__ __forceinline__ int64_t scale_offset(int64_t row, int64_t scale_k,
                                                int64_t blocks_k) {
  return ((row >> 7) * blocks_k << 9) + ((row & 0x1f) << 4) +
         (((row & 0x7f) >> 5) << 2) + ((scale_k >> 2) << 9) + (scale_k & 0x3);
}

__device__ __forceinline__ float ceil_power_of_two(float value) {
  if (!(value > 0.0f)) {
    return 1.0f;
  }
  int exponent = 0;
  float mantissa = frexpf(value, &exponent);
  if (mantissa == 0.5f) {
    return ldexpf(1.0f, exponent - 1);
  }
  return ldexpf(1.0f, exponent);
}

__global__ void pack_mxfp4_weight_kernel(const __nv_bfloat16 *input,
                                         uint8_t *packed, uint8_t *scales,
                                         int64_t rows, int64_t columns,
                                         int64_t packed_columns,
                                         int64_t blocks_k,
                                         int64_t scale_groups_k) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t groups = rows * scale_groups_k;
  if (index >= groups) {
    return;
  }
  int64_t row = index / scale_groups_k;
  int64_t scale_k = index - row * scale_groups_k;
  int64_t column0 = scale_k * kSfVecSize;
  float values[kSfVecSize];
  float max_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < kSfVecSize; ++i) {
    int64_t column = column0 + i;
    float value =
        column < columns
            ? __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(
                  input)[row * columns + column])
            : 0.0f;
    values[i] = value;
    max_abs = fmaxf(max_abs, fabsf(value));
  }
  float scale_value = ceil_power_of_two(max_abs / 6.0f);
  cutlass::float_ue8m0_t scale(scale_value);
  float scale_dequant = static_cast<float>(scale);
  float inverse_scale = 1.0f / scale_dequant;
#pragma unroll
  for (int i = 0; i < kSfVecSize; i += 2) {
    __nv_fp4x2_storage_t pair = __nv_cvt_float2_to_fp4x2(
        make_float2(values[i] * inverse_scale, values[i + 1] * inverse_scale),
        __NV_E2M1, cudaRoundNearest);
    packed[row * packed_columns + (column0 + i) / 2] =
        static_cast<uint8_t>(pair);
  }
  scales[scale_offset(row, scale_k, blocks_k)] = scale.raw();
}

__global__ void pack_mxfp8_activation_warp_kernel(
    const __nv_bfloat16 *input, uint8_t *packed, uint8_t *scales, int64_t rows,
    int64_t columns, int64_t blocks_k, int64_t scale_groups_k) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;
  int64_t group = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  int64_t groups = rows * scale_groups_k;
  if (group >= groups) {
    return;
  }
  int64_t row = group / scale_groups_k;
  int64_t scale_k = group - row * scale_groups_k;
  int64_t column = scale_k * kSfVecSize + lane;
  float value =
      column < columns ? __bfloat162float(input[row * columns + column]) : 0.0f;
  float max_abs = fabsf(value);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  max_abs = __shfl_sync(0xffffffff, max_abs, 0);
  float scale_value = ceil_power_of_two(max_abs / 448.0f);
  cutlass::float_ue8m0_t scale(scale_value);
  float inverse_scale = 1.0f / static_cast<float>(scale);
  float quantized = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
  if (column < columns) {
    packed[row * columns + column] =
        __nv_cvt_float_to_fp8(quantized, __NV_SATFINITE, __NV_E4M3);
  }
  if (lane == 0) {
    scales[scale_offset(row, scale_k, blocks_k)] = scale.raw();
  }
}

template <bool Interleaved>
__global__ void
pack_swiglu_mxfp8_warp_kernel(const __nv_bfloat16 *gate_up, uint8_t *packed,
                              uint8_t *scales, int64_t rows, int64_t columns,
                              int64_t blocks_k, int64_t scale_groups_k) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;
  int64_t group = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  int64_t groups = rows * scale_groups_k;
  if (group >= groups) {
    return;
  }
  int64_t row = group / scale_groups_k;
  int64_t scale_k = group - row * scale_groups_k;
  int64_t column = scale_k * kSfVecSize + lane;
  int64_t row_offset = row * columns * 2;
  int64_t gate_index = row_offset + column;
  int64_t up_index = row_offset + columns + column;
  if constexpr (Interleaved) {
    constexpr int64_t kBlockColumns = 64;
    int64_t block = column / kBlockColumns;
    int64_t column_in_block = column % kBlockColumns;
    up_index = row_offset + block * (2 * kBlockColumns) + column_in_block;
    gate_index = up_index + kBlockColumns;
  }
  float gate = __bfloat162float(gate_up[gate_index]);
  float up = __bfloat162float(gate_up[up_index]);
  float value = gate / (1.0f + expf(-gate)) * up;
  float max_abs = fabsf(value);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
  }
  max_abs = __shfl_sync(0xffffffff, max_abs, 0);
  float scale_value = ceil_power_of_two(max_abs / 448.0f);
  cutlass::float_ue8m0_t scale(scale_value);
  float quantized = value / static_cast<float>(scale);
  packed[row * columns + column] = __nv_cvt_float_to_fp8(
      fminf(448.0f, fmaxf(-448.0f, quantized)), __NV_SATFINITE, __NV_E4M3);
  if (lane == 0) {
    scales[scale_offset(row, scale_k, blocks_k)] = scale.raw();
  }
}

void check_metadata(const at::Tensor &metadata, int64_t &n, int64_t &k) {
  TORCH_CHECK(metadata.device().is_cpu() &&
                  metadata.scalar_type() == at::kLong && metadata.numel() >= 2,
              "metadata must be CPU int64 [N,K]");
  n = metadata.data_ptr<int64_t>()[0];
  k = metadata.data_ptr<int64_t>()[1];
}

std::vector<at::Tensor> pack_mxfp4_weight(const at::Tensor &weight,
                                          const std::string &,
                                          const std::string &) {
  TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                  weight.scalar_type() == at::kBFloat16 && weight.dim() == 2,
              "MXFP4 weight must be contiguous CUDA BF16 [N,K]");
  int64_t n = weight.size(0);
  int64_t k = weight.size(1);
  TORCH_CHECK(n % 32 == 0 && k % kSfVecSize == 0,
              "MXFP4 requires N divisible by 32 and K divisible by 32");
  int64_t scale_groups_k = k / kSfVecSize;
  int64_t blocks_k = scale_blocks_k(k);
  auto bytes = weight.options().dtype(at::kByte);
  auto packed = at::empty({n, k / 2}, bytes);
  auto scales = at::zeros({scale_blocks_mn(n) * blocks_k * 512}, bytes);
  auto metadata =
      at::empty({2}, weight.options().device(at::kCPU).dtype(at::kLong));
  metadata.data_ptr<int64_t>()[0] = n;
  metadata.data_ptr<int64_t>()[1] = k;
  constexpr int threads = 256;
  int blocks = static_cast<int>(ceil_div_int64(n * scale_groups_k, threads));
  c10::cuda::CUDAGuard guard(weight.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(weight.get_device());
  pack_mxfp4_weight_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()),
      packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), n, k, k / 2,
      blocks_k, scale_groups_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales, metadata};
}

std::vector<at::Tensor> pack_mxfp8_activation(const at::Tensor &input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                  input.scalar_type() == at::kBFloat16 && input.dim() >= 2,
              "MXFP8 input must be contiguous CUDA BF16");
  int64_t k = input.size(-1);
  int64_t m = input.numel() / k;
  TORCH_CHECK(k % kSfVecSize == 0, "MXFP8 requires K divisible by 32");
  int64_t scale_groups_k = k / kSfVecSize;
  int64_t blocks_k = scale_blocks_k(k);
  auto bytes = input.options().dtype(at::kByte);
  auto packed = at::empty(input.sizes(), bytes);
  auto scales = at::zeros({scale_blocks_mn(m) * blocks_k * 512}, bytes);
  constexpr int threads = 256;
  constexpr int warps_per_block = threads / 32;
  int blocks =
      static_cast<int>(ceil_div_int64(m * scale_groups_k, warps_per_block));
  c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  pack_mxfp8_activation_warp_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()),
      packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), m, k, blocks_k,
      scale_groups_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}

std::vector<at::Tensor> pack_swiglu_mxfp8(const at::Tensor &gate_up,
                                          int64_t intermediate_size,
                                          bool interleaved) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.is_contiguous() &&
                  gate_up.scalar_type() == at::kBFloat16 &&
                  gate_up.size(-1) == 2 * intermediate_size,
              "gate_up must be contiguous CUDA BF16 [...,2K]");
  TORCH_CHECK(intermediate_size % kSfVecSize == 0,
              "MXFP8 SwiGLU requires K divisible by 32");
  int64_t rows = gate_up.numel() / (2 * intermediate_size);
  int64_t scale_groups_k = intermediate_size / kSfVecSize;
  int64_t blocks_k = scale_blocks_k(intermediate_size);
  auto bytes = gate_up.options().dtype(at::kByte);
  std::vector<int64_t> packed_shape(gate_up.sizes().begin(),
                                    gate_up.sizes().end());
  packed_shape.back() = intermediate_size;
  auto packed = at::empty(packed_shape, bytes);
  auto scales = at::zeros({scale_blocks_mn(rows) * blocks_k * 512}, bytes);
  constexpr int threads = 256;
  constexpr int warps_per_block = threads / 32;
  int blocks =
      static_cast<int>(ceil_div_int64(rows * scale_groups_k, warps_per_block));
  c10::cuda::CUDAGuard guard(gate_up.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(gate_up.get_device());
  auto *input =
      reinterpret_cast<const __nv_bfloat16 *>(gate_up.data_ptr<at::BFloat16>());
  if (interleaved) {
    pack_swiglu_mxfp8_warp_kernel<true><<<blocks, threads, 0, stream>>>(
        input, packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), rows,
        intermediate_size, blocks_k, scale_groups_k);
  } else {
    pack_swiglu_mxfp8_warp_kernel<false><<<blocks, threads, 0, stream>>>(
        input, packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), rows,
        intermediate_size, blocks_k, scale_groups_k);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}

namespace mx_gemm {

using namespace cute;

using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
constexpr int AlignmentB = 128;
constexpr int AlignmentCD = 8;
using MmaTileShape = Shape<_256, _128, _256>;
using ClusterShape = Shape<_2, _1, _1>;
using FusionOperation = cutlass::epilogue::fusion::LinCombPerColBias<
    ElementD, ElementAccumulator, ElementD, ElementC, ElementAccumulator,
    AlignmentCD>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutC, AlignmentCD, ElementD, LayoutD,
        AlignmentCD, cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB,
        LayoutB, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

void run(at::Tensor &output, const at::Tensor &packed_input,
         const at::Tensor &input_scale, const at::Tensor &packed_weight,
         const at::Tensor &weight_scale, const at::Tensor *bias, int64_t m,
         int64_t n, int64_t k, cudaStream_t stream) {
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using ArrayElementA = typename CollectiveMainloop::ElementA;
  using ArrayElementB = typename CollectiveMainloop::ElementB;
  using ScaleTypeA = ElementA::ScaleFactorType;
  using ScaleTypeB = ElementB::ScaleFactorType;
  using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
  int mi = static_cast<int>(m);
  int ni = static_cast<int>(n);
  int ki = static_cast<int>(k);
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {mi, ki, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {ni, ki, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {mi, ni, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {mi, ni, 1});
  auto layout_sfa =
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(mi, ni, ki, 1));
  auto layout_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(mi, ni, ki, 1));
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {mi, ni, ki, 1},
      {reinterpret_cast<const ArrayElementA *>(
           packed_input.data_ptr<uint8_t>()),
       stride_a,
       reinterpret_cast<const ArrayElementB *>(
           packed_weight.data_ptr<uint8_t>()),
       stride_b,
       reinterpret_cast<const ScaleTypeA *>(input_scale.data_ptr<uint8_t>()),
       layout_sfa,
       reinterpret_cast<const ScaleTypeB *>(weight_scale.data_ptr<uint8_t>()),
       layout_sfb},
      {{},
       reinterpret_cast<const ElementC *>(output.data_ptr<at::BFloat16>()),
       stride_c,
       reinterpret_cast<ElementD *>(output.data_ptr<at::BFloat16>()),
       stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  arguments.epilogue.thread.bias_ptr =
      bias == nullptr
          ? nullptr
          : reinterpret_cast<const ElementD *>(bias->data_ptr<at::BFloat16>());
  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS MX W4A8 cannot implement M=", m, ", N=", n, ", K=", k);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                             packed_input.options().dtype(at::kByte));
  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS MX W4A8 initialization failed");
  status = gemm.run(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS MX W4A8 launch failed");
}

} // namespace mx_gemm

namespace mx_swap_gemm {

using namespace cute;

using ElementA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutCD = cutlass::layout::ColumnMajor;
constexpr int AlignmentA = 128;
constexpr int AlignmentB = 16;
constexpr int AlignmentCD = 8;
using MmaTileShape = Shape<_128, _64, _128>;
using ClusterShape = Shape<_2, _1, _1>;
using FusionOperation = cutlass::epilogue::fusion::LinCombPerRowBias<
    ElementD, ElementAccumulator, ElementD, ElementC, ElementAccumulator,
    AlignmentCD>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutCD, AlignmentCD, ElementD, LayoutCD,
        AlignmentCD, cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB,
        LayoutB, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

void run(at::Tensor &output, const at::Tensor &packed_input,
         const at::Tensor &input_scale, const at::Tensor &packed_weight,
         const at::Tensor &weight_scale, const at::Tensor *bias, int64_t m,
         int64_t n, int64_t k, cudaStream_t stream) {
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using ArrayElementA = typename CollectiveMainloop::ElementA;
  using ArrayElementB = typename CollectiveMainloop::ElementB;
  using ScaleTypeA = ElementA::ScaleFactorType;
  using ScaleTypeB = ElementB::ScaleFactorType;
  using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
  int ni = static_cast<int>(n);
  int mi = static_cast<int>(m);
  int ki = static_cast<int>(k);
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {ni, ki, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {mi, ki, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {ni, mi, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {ni, mi, 1});
  auto layout_sfa =
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(ni, mi, ki, 1));
  auto layout_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(ni, mi, ki, 1));
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {ni, mi, ki, 1},
      {reinterpret_cast<const ArrayElementA *>(
           packed_weight.data_ptr<uint8_t>()),
       stride_a,
       reinterpret_cast<const ArrayElementB *>(
           packed_input.data_ptr<uint8_t>()),
       stride_b,
       reinterpret_cast<const ScaleTypeA *>(weight_scale.data_ptr<uint8_t>()),
       layout_sfa,
       reinterpret_cast<const ScaleTypeB *>(input_scale.data_ptr<uint8_t>()),
       layout_sfb},
      {{},
       reinterpret_cast<const ElementC *>(output.data_ptr<at::BFloat16>()),
       stride_c,
       reinterpret_cast<ElementD *>(output.data_ptr<at::BFloat16>()),
       stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  arguments.epilogue.thread.bias_ptr =
      bias == nullptr
          ? nullptr
          : reinterpret_cast<const ElementD *>(bias->data_ptr<at::BFloat16>());
  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS MX W4A8 Swap-AB cannot implement M=", m, ", N=", n,
              ", K=", k);
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = at::empty({static_cast<int64_t>(workspace_size)},
                             packed_input.options().dtype(at::kByte));
  status = gemm.initialize(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS MX W4A8 Swap-AB initialization failed");
  status = gemm.run(arguments, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS MX W4A8 Swap-AB launch failed");
}

} // namespace mx_swap_gemm

const at::Tensor *check_bias(const std::optional<at::Tensor> &bias,
                             const at::Tensor &input, int64_t n) {
  if (!bias.has_value()) {
    return nullptr;
  }
  TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                  bias->scalar_type() == at::kBFloat16 && bias->numel() == n &&
                  bias->device() == input.device(),
              "bias must be contiguous CUDA BF16 [N]");
  return &*bias;
}

at::Tensor linear_w4a8_packed(const at::Tensor &packed_input,
                              const at::Tensor &input_scale,
                              const at::Tensor &packed_weight,
                              const at::Tensor &weight_scale,
                              const at::Tensor &metadata,
                              const std::optional<at::Tensor> &bias) {
  int64_t n = 0;
  int64_t k = 0;
  check_metadata(metadata, n, k);
  TORCH_CHECK(packed_input.is_cuda() && packed_input.is_contiguous() &&
                  packed_input.scalar_type() == at::kByte &&
                  packed_input.dim() >= 2 && packed_input.size(-1) == k,
              "packed MXFP8 input has an invalid shape");
  TORCH_CHECK(input_scale.is_cuda() && input_scale.scalar_type() == at::kByte,
              "MXFP8 scale must be CUDA uint8");
  TORCH_CHECK(packed_weight.is_cuda() && packed_weight.is_contiguous() &&
                  packed_weight.scalar_type() == at::kByte &&
                  packed_weight.numel() == n * k / 2,
              "MXFP4 weight has an invalid shape");
  TORCH_CHECK(weight_scale.is_cuda() && weight_scale.scalar_type() == at::kByte,
              "MXFP4 scale must be CUDA uint8");
  int64_t m = packed_input.numel() / k;
  std::vector<int64_t> output_shape(packed_input.sizes().begin(),
                                    packed_input.sizes().end());
  output_shape.back() = n;
  auto output =
      at::empty(output_shape, packed_input.options().dtype(at::kBFloat16));
  const at::Tensor *bias_ptr = check_bias(bias, packed_input, n);
  c10::cuda::CUDAGuard guard(packed_input.device());
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(packed_input.get_device());
  if (m <= 16) {
    mx_swap_gemm::run(output, packed_input, input_scale, packed_weight,
                      weight_scale, bias_ptr, m, n, k, stream);
  } else {
    mx_gemm::run(output, packed_input, input_scale, packed_weight, weight_scale,
                 bias_ptr, m, n, k, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor linear_w4a8(const at::Tensor &input, const at::Tensor &packed_weight,
                       const at::Tensor &weight_scale,
                       const at::Tensor &metadata,
                       const std::optional<at::Tensor> &bias) {
  auto packed = pack_mxfp8_activation(input.contiguous());
  return linear_w4a8_packed(packed[0], packed[1], packed_weight, weight_scale,
                            metadata, bias);
}

bool supports_device(int64_t cc) { return cc >= 100; }

} // namespace

TORCH_LIBRARY(robotics_cutlass_fp4_mx, m) {
  m.def("supports_device(int cc) -> bool");
  m.def("pack_mxfp4_weight(Tensor weight, str role, str target_name) -> "
        "Tensor[]");
  m.def("pack_mxfp8_activation(Tensor input) -> Tensor[]");
  m.def("pack_swiglu_mxfp8(Tensor gate_up, int intermediate_size, "
        "bool interleaved=False) -> Tensor[]");
  m.def("linear_w4a8_packed(Tensor packed_input, Tensor input_scale, "
        "Tensor packed_weight, Tensor weight_scale, Tensor metadata, "
        "Tensor? bias) -> Tensor");
  m.def("linear_w4a8(Tensor input, Tensor packed_weight, Tensor weight_scale, "
        "Tensor metadata, Tensor? bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(robotics_cutlass_fp4_mx, CUDA, m) {
  m.impl("pack_mxfp4_weight", &pack_mxfp4_weight);
  m.impl("pack_mxfp8_activation", &pack_mxfp8_activation);
  m.impl("pack_swiglu_mxfp8", &pack_swiglu_mxfp8);
  m.impl("linear_w4a8_packed", &linear_w4a8_packed);
  m.impl("linear_w4a8", &linear_w4a8);
}

TORCH_LIBRARY_IMPL(robotics_cutlass_fp4_mx, CatchAll, m) {
  m.impl("supports_device", &supports_device);
}
