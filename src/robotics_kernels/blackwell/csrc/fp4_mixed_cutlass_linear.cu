#include <cstdint>
#include <optional>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_mixed_dtype_blockwise_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace {

constexpr int64_t kScaleGroupK = 16;

int64_t ceil_div_int64(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

__device__ __forceinline__ int64_t nvfp4_scale_offset(int64_t row,
                                                      int64_t scale_k,
                                                      int64_t blocks_k) {
  return ((row >> 7) * blocks_k << 9) + ((row & 0x1f) << 4) +
         (((row & 0x7f) >> 5) << 2) + ((scale_k >> 2) << 9) + (scale_k & 0x3);
}

__global__ void repack_scale_bf16_kernel(const uint8_t *input,
                                         at::BFloat16 *output, int64_t rows,
                                         int64_t scale_groups_k,
                                         int64_t nv_blocks_k) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t elements = rows * scale_groups_k;
  if (index >= elements) {
    return;
  }
  int64_t row = index % rows;
  int64_t scale_k = index / rows;
  uint8_t raw = input[nvfp4_scale_offset(row, scale_k, nv_blocks_k)];
  cutlass::float_ue4m3_t scale = cutlass::float_ue4m3_t::bitcast(raw);
  output[index] = at::BFloat16(static_cast<float>(scale));
}

void check_weight_inputs(const at::Tensor &packed_weight,
                         const at::Tensor &metadata, int64_t &n, int64_t &k) {
  TORCH_CHECK(packed_weight.is_cuda(), "packed_weight must be CUDA");
  TORCH_CHECK(packed_weight.scalar_type() == at::kByte,
              "packed_weight must be uint8");
  TORCH_CHECK(packed_weight.is_contiguous(),
              "packed_weight must be contiguous");
  TORCH_CHECK(metadata.device().is_cpu() &&
                  metadata.scalar_type() == at::kLong && metadata.numel() >= 2,
              "metadata must be a CPU int64 tensor containing N and K");
  n = metadata.data_ptr<int64_t>()[0];
  k = metadata.data_ptr<int64_t>()[1];
  TORCH_CHECK(n > 0 && k > 0 && k % kScaleGroupK == 0,
              "invalid FP4 weight shape N=", n, ", K=", k);
  TORCH_CHECK(packed_weight.numel() == n * k / 2,
              "packed_weight size does not match metadata");
}

at::Tensor repack_weight_scales(const at::Tensor &nvfp4_scales,
                                const at::Tensor &metadata) {
  TORCH_CHECK(metadata.device().is_cpu() &&
                  metadata.scalar_type() == at::kLong && metadata.numel() >= 2,
              "metadata must be a CPU int64 tensor containing N and K");
  int64_t n = metadata.data_ptr<int64_t>()[0];
  int64_t k = metadata.data_ptr<int64_t>()[1];
  TORCH_CHECK(n > 0 && k > 0 && k % kScaleGroupK == 0,
              "invalid FP4 weight shape N=", n, ", K=", k);
  TORCH_CHECK(nvfp4_scales.is_cuda() &&
                  nvfp4_scales.scalar_type() == at::kByte &&
                  nvfp4_scales.is_contiguous(),
              "nvfp4_scales must be contiguous CUDA uint8");
  int64_t scale_groups_k = k / kScaleGroupK;
  int64_t nv_blocks_k = ceil_div_int64(scale_groups_k, 4);
  auto bf16 = at::empty({scale_groups_k, n},
                        nvfp4_scales.options().dtype(at::kBFloat16));
  constexpr int threads = 256;
  int blocks = static_cast<int>(ceil_div_int64(n * scale_groups_k, threads));
  c10::cuda::CUDAGuard guard(nvfp4_scales.device());
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(nvfp4_scales.get_device());
  repack_scale_bf16_kernel<<<blocks, threads, 0, stream>>>(
      nvfp4_scales.data_ptr<uint8_t>(), bf16.data_ptr<at::BFloat16>(), n,
      scale_groups_k, nv_blocks_k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return bf16;
}

__global__ void
materialize_weight_bf16_kernel(const uint8_t *__restrict__ packed_weight,
                               const at::BFloat16 *__restrict__ mixed_scale,
                               at::BFloat16 *__restrict__ output, int64_t n,
                               int64_t k) {
  constexpr int kValuesPerThread = 8;
  int64_t vector_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t vectors_per_row = k / kValuesPerThread;
  int64_t vectors = n * vectors_per_row;
  if (vector_index >= vectors) {
    return;
  }
  int64_t row = vector_index / vectors_per_row;
  int64_t vector_column = vector_index - row * vectors_per_row;
  int64_t column = vector_column * kValuesPerThread;
  uint32_t packed = reinterpret_cast<const uint32_t *>(
      packed_weight + row * (k / 2))[vector_column];
  float scale =
      static_cast<float>(mixed_scale[(column / kScaleGroupK) * n + row]);
  uint4 output_vector;
  auto *output_bits = reinterpret_cast<unsigned short *>(&output_vector);
#pragma unroll
  for (int index = 0; index < kValuesPerThread; ++index) {
    uint8_t raw = static_cast<uint8_t>((packed >> (4 * index)) & 0x0f);
    cutlass::float_e2m1_t fp4 = cutlass::float_e2m1_t::bitcast(raw);
    output_bits[index] =
        __bfloat16_as_ushort(__float2bfloat16(static_cast<float>(fp4) * scale));
  }
  reinterpret_cast<uint4 *>(output)[vector_index] = output_vector;
}

void launch_materialize_weight_bf16(const at::Tensor &packed_weight,
                                    const at::Tensor &mixed_scale_bf16,
                                    const at::Tensor &metadata,
                                    at::Tensor &output) {
  int64_t n = 0;
  int64_t k = 0;
  check_weight_inputs(packed_weight, metadata, n, k);
  TORCH_CHECK(mixed_scale_bf16.is_cuda() && mixed_scale_bf16.is_contiguous() &&
                  mixed_scale_bf16.scalar_type() == at::kBFloat16 &&
                  mixed_scale_bf16.numel() == n * (k / kScaleGroupK),
              "W4A16 mixed scales have an invalid layout");
  TORCH_CHECK(packed_weight.device() == mixed_scale_bf16.device(),
              "W4A16 tensors must share one CUDA device");
  TORCH_CHECK(output.is_cuda() && output.is_contiguous() &&
                  output.scalar_type() == at::kBFloat16 &&
                  output.numel() == n * k &&
                  output.device() == packed_weight.device(),
              "W4A16 persistent output must be contiguous CUDA BF16 [N,K]");
  constexpr int threads = 256;
  constexpr int values_per_thread = 8;
  TORCH_CHECK(k % values_per_thread == 0,
              "W4A16 persistent materialization requires K divisible by 8");
  int blocks =
      static_cast<int>(ceil_div_int64(n * (k / values_per_thread), threads));
  c10::cuda::CUDAGuard guard(packed_weight.device());
  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(packed_weight.get_device());
  materialize_weight_bf16_kernel<<<blocks, threads, 0, stream>>>(
      packed_weight.data_ptr<uint8_t>(),
      mixed_scale_bf16.data_ptr<at::BFloat16>(),
      output.data_ptr<at::BFloat16>(), n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor materialize_weight_bf16(const at::Tensor &packed_weight,
                                   const at::Tensor &mixed_scale_bf16,
                                   const at::Tensor &metadata) {
  int64_t n = 0;
  int64_t k = 0;
  check_weight_inputs(packed_weight, metadata, n, k);
  auto output = at::empty({n, k}, packed_weight.options().dtype(at::kBFloat16));
  launch_materialize_weight_bf16(packed_weight, mixed_scale_bf16, metadata,
                                 output);
  return output;
}

at::Tensor materialize_weight_bf16_out(const at::Tensor &packed_weight,
                                       const at::Tensor &mixed_scale_bf16,
                                       const at::Tensor &metadata,
                                       at::Tensor output) {
  launch_materialize_weight_bf16(packed_weight, mixed_scale_bf16, metadata,
                                 output);
  return output;
}

namespace mixed_gemm {

using namespace cute;

using ElementWeight = cutlass::float_e2m1_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using LayoutWeight = cutlass::layout::RowMajor;
using LayoutActivation = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;
constexpr int AlignmentWeight = 32;
constexpr int AlignmentOutput = 8;
using ScaleConfig =
    cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<1, kScaleGroupK>;
using LayoutScale = decltype(ScaleConfig::deduce_layout_scale());

template <typename ElementActivation_, typename ElementWeightScale_,
          typename MmaTileShape_, typename ClusterShape_,
          typename EpilogueSchedule_, typename KernelSchedule_>
struct GemmSpec {
  using ElementActivation = ElementActivation_;
  using ElementWeightScale = ElementWeightScale_;
  using MmaTileShape = MmaTileShape_;
  using ClusterShape = ClusterShape_;
  using EpilogueSchedule = EpilogueSchedule_;
  using KernelSchedule = KernelSchedule_;
  static constexpr int AlignmentActivation =
      128 / cutlass::sizeof_bits<ElementActivation>::value;
  using FusionOperation = cutlass::epilogue::fusion::LinCombPerRowBiasEltAct<
      cutlass::epilogue::thread::Identity, ElementOutput, ElementCompute,
      ElementOutput, ElementOutput>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementCompute, ElementOutput, LayoutOutput, AlignmentOutput,
          ElementOutput, LayoutOutput, AlignmentOutput, EpilogueSchedule,
          FusionOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          cute::tuple<ElementWeight, ElementWeightScale>,
          cute::tuple<LayoutWeight, LayoutScale>, AlignmentWeight,
          ElementActivation, LayoutActivation, AlignmentActivation,
          ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using W4A16PrefillSpec =
    GemmSpec<cutlass::bfloat16_t, cutlass::bfloat16_t, Shape<_256, _128, _128>,
             Shape<_2, _1, _1>, cutlass::epilogue::TmaWarpSpecialized2Sm,
             cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100>;

using W4A16DecodeSpec =
    GemmSpec<cutlass::bfloat16_t, cutlass::bfloat16_t, Shape<_128, _128, _64>,
             Shape<_1, _1, _1>, cutlass::epilogue::TmaWarpSpecialized1Sm,
             cutlass::gemm::KernelTmaWarpSpecialized1SmMixedInputSm100>;

using W4A16DecodeN64Spec =
    GemmSpec<cutlass::bfloat16_t, cutlass::bfloat16_t, Shape<_128, _64, _128>,
             Shape<_1, _1, _1>, cutlass::epilogue::TmaWarpSpecialized1Sm,
             cutlass::gemm::KernelTmaWarpSpecialized1SmMixedInputSm100>;

using W4A16DecodeNoSmemSpec =
    GemmSpec<cutlass::bfloat16_t, cutlass::bfloat16_t, Shape<_128, _128, _64>,
             Shape<_1, _1, _1>, cutlass::epilogue::NoSmemWarpSpecialized1Sm,
             cutlass::gemm::KernelTmaWarpSpecialized1SmMixedInputSm100>;

template <typename Spec>
void run(at::Tensor &output, const at::Tensor &input,
         const at::Tensor &packed_weight, const at::Tensor &mixed_scale,
         const at::Tensor *input_scale, const at::Tensor *bias, int64_t m,
         int64_t n, int64_t k, cudaStream_t stream) {
  using Gemm = typename Spec::Gemm;
  using ElementActivation = typename Spec::ElementActivation;
  using ElementWeightScale = typename Spec::ElementWeightScale;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  int ni = static_cast<int>(n);
  int mi = static_cast<int>(m);
  int ki = static_cast<int>(k);
  auto stride_weight = cutlass::make_cute_packed_stride(StrideA{}, {ni, ki, 1});
  auto stride_input = cutlass::make_cute_packed_stride(StrideB{}, {mi, ki, 1});
  auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {ni, mi, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {ni, mi, 1});
  auto layout_scale =
      ScaleConfig::tile_atom_to_shape_scale(make_shape(ni, ki, 1));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {ni, mi, ki, 1},
      {reinterpret_cast<const ElementWeight *>(
           packed_weight.data_ptr<uint8_t>()),
       stride_weight,
       reinterpret_cast<const ElementActivation *>(input.data_ptr()),
       stride_input,
       reinterpret_cast<const ElementWeightScale *>(mixed_scale.data_ptr()),
       layout_scale},
      {{},
       reinterpret_cast<const ElementOutput *>(output.data_ptr<at::BFloat16>()),
       stride_c,
       reinterpret_cast<ElementOutput *>(output.data_ptr<at::BFloat16>()),
       stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  arguments.epilogue.thread.bias_ptr =
      bias == nullptr ? nullptr
                      : reinterpret_cast<const ElementOutput *>(
                            bias->data_ptr<at::BFloat16>());
  (void)input_scale;

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP4 mixed-input GEMM cannot implement M=", m, ", N=", n,
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
              "CUTLASS FP4 mixed-input GEMM initialization failed");
  status = gemm.run(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP4 mixed-input GEMM launch failed");
}

} // namespace mixed_gemm

const at::Tensor *check_bias(const std::optional<at::Tensor> &bias,
                             const at::Tensor &input, int64_t n) {
  if (!bias.has_value()) {
    return nullptr;
  }
  TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() &&
                  bias->scalar_type() == at::kBFloat16 && bias->dim() == 1 &&
                  bias->numel() == n && bias->device() == input.device(),
              "bias must be contiguous CUDA BF16 with shape [N]");
  return &*bias;
}

at::Tensor linear_w4a16_tactic(const at::Tensor &input,
                               const at::Tensor &packed_weight,
                               const at::Tensor &mixed_scale_bf16,
                               const at::Tensor &metadata,
                               const std::optional<at::Tensor> &bias,
                               int64_t tactic) {
  int64_t n = 0;
  int64_t k = 0;
  check_weight_inputs(packed_weight, metadata, n, k);
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                  input.scalar_type() == at::kBFloat16 && input.dim() >= 2 &&
                  input.size(-1) == k,
              "W4A16 input must be contiguous CUDA BF16 with last dim K");
  TORCH_CHECK(mixed_scale_bf16.is_cuda() && mixed_scale_bf16.is_contiguous() &&
                  mixed_scale_bf16.scalar_type() == at::kBFloat16 &&
                  mixed_scale_bf16.numel() == n * (k / kScaleGroupK),
              "W4A16 mixed scales have an invalid layout");
  TORCH_CHECK(input.device() == packed_weight.device() &&
                  input.device() == mixed_scale_bf16.device(),
              "W4A16 tensors must share one CUDA device");
  const at::Tensor *bias_ptr = check_bias(bias, input, n);
  int64_t m = input.numel() / k;
  std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end());
  output_shape.back() = n;
  auto output = at::empty(output_shape, input.options());
  c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  if (tactic == 0) {
    tactic = m <= 16 ? 3 : 2;
  }
  if (tactic == 1) {
    mixed_gemm::run<mixed_gemm::W4A16DecodeSpec>(output, input, packed_weight,
                                                 mixed_scale_bf16, nullptr,
                                                 bias_ptr, m, n, k, stream);
  } else if (tactic == 2) {
    mixed_gemm::run<mixed_gemm::W4A16PrefillSpec>(output, input, packed_weight,
                                                  mixed_scale_bf16, nullptr,
                                                  bias_ptr, m, n, k, stream);
  } else if (tactic == 3) {
    mixed_gemm::run<mixed_gemm::W4A16DecodeN64Spec>(
        output, input, packed_weight, mixed_scale_bf16, nullptr, bias_ptr, m, n,
        k, stream);
  } else if (tactic == 4) {
    mixed_gemm::run<mixed_gemm::W4A16DecodeNoSmemSpec>(
        output, input, packed_weight, mixed_scale_bf16, nullptr, bias_ptr, m, n,
        k, stream);
  } else {
    TORCH_CHECK(false, "unknown W4A16 tactic ", tactic);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor linear_w4a16(const at::Tensor &input,
                        const at::Tensor &packed_weight,
                        const at::Tensor &mixed_scale_bf16,
                        const at::Tensor &metadata,
                        const std::optional<at::Tensor> &bias) {
  return linear_w4a16_tactic(input, packed_weight, mixed_scale_bf16, metadata,
                             bias, 0);
}

bool supports_device(int64_t cc) { return cc >= 100; }

} // namespace

TORCH_LIBRARY(robotics_cutlass_fp4_mixed, m) {
  m.def("supports_device(int cc) -> bool");
  m.def("repack_weight_scales(Tensor nvfp4_scales, Tensor metadata) -> Tensor");
  m.def("materialize_weight_bf16(Tensor packed_weight, "
        "Tensor mixed_scale_bf16, Tensor metadata) -> Tensor");
  m.def("materialize_weight_bf16_out(Tensor packed_weight, "
        "Tensor mixed_scale_bf16, Tensor metadata, Tensor(a!) output) "
        "-> Tensor(a!)");
  m.def("linear_w4a16(Tensor input, Tensor packed_weight, "
        "Tensor mixed_scale_bf16, Tensor metadata, Tensor? bias) -> Tensor");
  m.def("linear_w4a16_tactic(Tensor input, Tensor packed_weight, "
        "Tensor mixed_scale_bf16, Tensor metadata, Tensor? bias, "
        "int tactic) -> Tensor");
}

TORCH_LIBRARY_IMPL(robotics_cutlass_fp4_mixed, CUDA, m) {
  m.impl("repack_weight_scales", &repack_weight_scales);
  m.impl("materialize_weight_bf16", &materialize_weight_bf16);
  m.impl("materialize_weight_bf16_out", &materialize_weight_bf16_out);
  m.impl("linear_w4a16", &linear_w4a16);
  m.impl("linear_w4a16_tactic", &linear_w4a16_tactic);
}

TORCH_LIBRARY_IMPL(robotics_cutlass_fp4_mixed, CatchAll, m) {
  m.impl("supports_device", &supports_device);
}
