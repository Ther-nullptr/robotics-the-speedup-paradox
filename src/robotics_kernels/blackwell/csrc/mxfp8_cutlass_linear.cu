// Native CUTLASS MXFP8 pack and BF16-output GEMM for StreamingVLM.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/functional.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/util/packed_stride.hpp"

#include <vector>

#include "mxfp8_common.cuh"

namespace {

using F8 = cutlass::float_e4m3_t;
using MX = cutlass::mx_float8_t<F8>;
using BF16 = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

__global__ void pack_bf16_mxfp8_kernel(const __nv_bfloat16 *input,
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
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    int64_t index = row * columns + group * 32 + lane * 4 + item;
    float value = __bfloat162float(input[index]);
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

std::vector<at::Tensor> pack_mxfp8(const at::Tensor &input) {
  TORCH_CHECK(
      input.is_cuda() && input.scalar_type() == at::kBFloat16 &&
          input.is_contiguous() && input.dim() == 2 && input.size(0) > 0 &&
          input.size(1) % 32 == 0,
      "MXFP8 pack requires contiguous CUDA BF16 [M,K] with K divisible by 32");
  auto packed = at::empty(input.sizes(),
                          input.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto scales = at::empty({scale_storage_size(input.size(0), input.size(1))},
                          input.options().dtype(at::kByte));
  int64_t groups_padded = ((input.size(1) + 127) / 128) * 4;
  int64_t groups = ((input.size(0) + 127) / 128) * 128 * groups_padded;
  constexpr int threads = 128;
  TORCH_CHECK(groups % 16 == 0, "invalid MXFP8 pack geometry");
  int blocks = static_cast<int>(groups / 16);
  c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  pack_bf16_mxfp8_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<uint32_t *>(packed.data_ptr()),
      scales.data_ptr<uint8_t>(), input.size(0), input.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales};
}

template <class Tile, class Cluster, bool TwoSm> struct Mxfp8Spec {
  using Fusion = cutlass::epilogue::fusion::LinCombPerColBias<BF16, float, BF16,
                                                              void, float, 8>;
  using Schedule =
      std::conditional_t<TwoSm, cutlass::epilogue::TmaWarpSpecialized2Sm,
                         cutlass::epilogue::TmaWarpSpecialized1Sm>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, OpClass, Tile, Cluster,
      cute::Shape<cute::_128, cute::_128>, float, float, void, Row, 16, BF16,
      Row, 16, Schedule, Fusion>::CollectiveOp;
  using KernelSchedule = std::conditional_t<
      TwoSm, cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100,
      cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, OpClass, MX, Row, 16, MX,
      cutlass::layout::ColumnMajor, 16, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(
          typename Epilogue::SharedStorage)>,
      KernelSchedule>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                           Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class Tile, class Cluster, bool TwoSm>
void run_mxfp8(at::Tensor &output, const at::Tensor &input,
               const at::Tensor &input_scales, const at::Tensor &weight,
               const at::Tensor &weight_scales,
               const c10::optional<at::Tensor> &bias, int raster, int swizzle,
               cudaStream_t stream) {
  using Spec = Mxfp8Spec<Tile, Cluster, TwoSm>;
  using Gemm = typename Spec::Gemm;
  using Kernel = typename Spec::Kernel;
  using Mainloop = typename Spec::Mainloop;
  using SfConfig = typename Mainloop::Sm1xxBlkScaledConfig;
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);
  auto stride_a =
      cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1});
  auto stride_b =
      cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1});
  auto stride_c =
      cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {m, n, 1});
  auto stride_d =
      cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1});
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<typename Mainloop::ElementA const *>(input.data_ptr()),
       stride_a,
       reinterpret_cast<typename Mainloop::ElementB const *>(weight.data_ptr()),
       stride_b, reinterpret_cast<SF const *>(input_scales.data_ptr()),
       SfConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1)),
       reinterpret_cast<SF const *>(weight_scales.data_ptr()),
       SfConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1))},
      {{},
       nullptr,
       stride_c,
       reinterpret_cast<BF16 *>(output.data_ptr<at::BFloat16>()),
       stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  arguments.epilogue.thread.bias_ptr =
      bias ? reinterpret_cast<BF16 const *>(bias->data_ptr<at::BFloat16>())
           : nullptr;
  if (raster == 1) {
    arguments.scheduler.raster_order =
        cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
  }
  if (raster == 2) {
    arguments.scheduler.raster_order =
        cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;
  }
  arguments.scheduler.max_swizzle_size = swizzle;
  Gemm gemm;
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
              "MXFP8 GEMM cannot implement this shape/tactic");
  size_t bytes = Gemm::get_workspace_size(arguments);
  auto workspace = at::empty({static_cast<int64_t>(bytes)},
                             input.options().dtype(at::kByte));
  TORCH_CHECK(gemm.initialize(arguments, bytes ? workspace.data_ptr() : nullptr,
                              stream) == cutlass::Status::kSuccess,
              "MXFP8 GEMM initialization failed");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
              "MXFP8 GEMM launch failed");
}

at::Tensor linear_mxfp8(const at::Tensor &input, const at::Tensor &input_scales,
                        const at::Tensor &weight,
                        const at::Tensor &weight_scales,
                        const c10::optional<at::Tensor> &bias, int64_t tactic) {
  for (const auto &tensor : {input, input_scales, weight, weight_scales}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == input.device() &&
                    tensor.is_contiguous(),
                "MXFP8 tensors must be contiguous on one CUDA device");
  }
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && input.size(0) > 0 &&
                  input.size(1) == weight.size(1) && weight.size(0) > 0,
              "MXFP8 operands must be [M,K] and [N,K]");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                  weight.scalar_type() == input.scalar_type() &&
                  input.size(1) % 32 == 0 && weight.size(0) % 32 == 0,
              "MXFP8 GEMM requires aligned E4M3 operands");
  TORCH_CHECK(input_scales.scalar_type() == at::kByte &&
                  weight_scales.scalar_type() == at::kByte &&
                  input_scales.numel() ==
                      scale_storage_size(input.size(0), input.size(1)) &&
                  weight_scales.numel() ==
                      scale_storage_size(weight.size(0), weight.size(1)),
              "invalid MXFP8 E8M0 scale storage");
  if (bias) {
    TORCH_CHECK(bias->is_cuda() && bias->device() == input.device() &&
                    bias->scalar_type() == at::kBFloat16 &&
                    bias->is_contiguous() && bias->numel() == weight.size(0),
                "MXFP8 bias must be contiguous CUDA BF16 [N]");
  }
  TORCH_CHECK(tactic > 0 && tactic < 4000, "invalid MXFP8 tactic");
  int base = tactic % 100;
  int raster = (tactic % 1000) / 100;
  int swizzle = 1 << (tactic / 1000);
  TORCH_CHECK(base >= 1 && base <= 6 && raster <= 2, "invalid MXFP8 tactic");

  auto output = at::empty({input.size(0), weight.size(0)},
                          input.options().dtype(at::kBFloat16));
  c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  using C1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using C2 = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using C4 = cute::Shape<cute::_4, cute::_1, cute::_1>;
#define RUN_MX(M, N, K, C, TWO_SM)                                             \
  run_mxfp8<cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>, C, TWO_SM>( \
      output, input, input_scales, weight, weight_scales, bias, raster,        \
      swizzle, stream)
  if (base == 1) {
    RUN_MX(128, 128, 128, C1, false);
  } else if (base == 2) {
    RUN_MX(256, 256, 128, C2, true);
  } else if (base == 3) {
    RUN_MX(256, 256, 128, C4, true);
  } else if (base == 4) {
    RUN_MX(256, 256, 256, C2, true);
  } else if (base == 5) {
    RUN_MX(256, 256, 256, C4, true);
  } else {
    RUN_MX(128, 256, 128, C1, false);
  }
#undef RUN_MX
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

} // namespace

TORCH_LIBRARY(robotics_mxfp8, m) {
  m.def("supports_device(int capability) -> bool");
  m.def("pack(Tensor input) -> Tensor[]");
  m.def("linear(Tensor input, Tensor input_scales, Tensor weight, "
        "Tensor weight_scales, Tensor? bias, int tactic) -> Tensor");
}

TORCH_LIBRARY_IMPL(robotics_mxfp8, CUDA, m) {
  m.impl("pack", &pack_mxfp8);
  m.impl("linear", &linear_mxfp8);
}

TORCH_LIBRARY_IMPL(robotics_mxfp8, CatchAll, m) {
  m.impl("supports_device", [](int64_t capability) {
    return capability == 100 || capability == 103 || capability == 110;
  });
}
