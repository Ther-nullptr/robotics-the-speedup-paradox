// Package-owned BF16 fprop candidate. CUTLASS is the pinned local dependency.
// Tensor semantics are NDHWC activations and KTRSC filters, FP32 accumulation.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cutlass/conv/conv3d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv3d_fprop.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>

namespace {
using Element = cutlass::bfloat16_t;
using Layout = cutlass::layout::TensorNDHWC;

template <int TileM, int TileN, int TileK, int WarpM, int WarpN, int Stages = 3>
at::Tensor run(const at::Tensor &input, const at::Tensor &weight,
               const c10::optional<at::Tensor> &bias,
               const cutlass::conv::Conv3dProblemSize &problem) {
  using Kernel = typename cutlass::conv::kernel::DefaultConv3dFprop<
      Element, Layout, Element, Layout, Element, Layout, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TileM, TileN, TileK>,
      cutlass::gemm::GemmShape<WarpM, WarpN, TileK>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
  using Conv = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
  auto output = at::empty(
      {problem.N, problem.Z, problem.P, problem.Q, problem.K}, input.options());
  auto output_ptr = reinterpret_cast<Element *>(output.data_ptr());
  auto source_ptr =
      bias ? reinterpret_cast<Element *>(bias->data_ptr()) : output_ptr;
  // Zero row stride broadcasts channel bias without allocating a full tensor.
  Layout source_layout(0, 0, 0, 0);
  typename Conv::Arguments args(
      problem,
      {reinterpret_cast<Element *>(input.data_ptr()),
       Layout::packed(problem.activation_extent())},
      {reinterpret_cast<Element *>(weight.data_ptr()),
       Layout::packed(problem.filter_extent())},
      {source_ptr, source_layout},
      {output_ptr, Layout::packed(problem.output_extent())},
      {1.0f, bias ? 1.0f : 0.0f});
  Conv op;
  auto status = op.can_implement(args);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "Unsupported CUTLASS convolution: ", cutlassGetStatusString(status));
  // split_k_slices=1 needs no workspace. Keep the check explicit for later
  // tactics.
  TORCH_CHECK(Conv::get_workspace_size(args) == 0,
              "Convolution tactic unexpectedly requires workspace");
  status =
      op(args, nullptr, at::cuda::getCurrentCUDAStream(input.get_device()));
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor fprop(const at::Tensor &input, const at::Tensor &weight,
                 const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
                 at::IntArrayRef padding, int64_t tactic) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() &&
                  input.device() == weight.device(),
              "Convolution tensors must share one CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "Convolution requires BF16 tensors");
  TORCH_CHECK(input.dim() == 5 && weight.dim() == 5,
              "Expected 5D NDHWC and KTRSC tensors");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
              "Packed convolution tensors must be contiguous");
  TORCH_CHECK(stride.size() == 3 && padding.size() == 3,
              "Expected explicit 3D stride and padding");
  TORCH_CHECK(input.size(4) == weight.size(4) && input.size(4) % 8 == 0 &&
                  weight.size(0) % 8 == 0,
              "Convolution channels must match and be divisible by 8");
  for (int i = 0; i < 5; ++i) {
    TORCH_CHECK(input.size(i) > 0 && weight.size(i) > 0,
                "Convolution dimensions must be positive");
  }
  TORCH_CHECK(input.numel() < (1LL << 30) && weight.numel() < (1LL << 30),
              "Convolution operands must be smaller than 2 GiB");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
              "Convolution operands must be 16-byte aligned");
  if (bias) {
    TORCH_CHECK(
        bias->device() == input.device() &&
            bias->scalar_type() == at::kBFloat16 && bias->is_contiguous() &&
            bias->dim() == 1 && bias->numel() == weight.size(0),
        "Expected contiguous BF16 output-channel bias on the same device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(bias->data_ptr()) % 16 == 0,
                "Convolution bias must be 16-byte aligned");
  }
  int output_spatial[3];
  int64_t output_elements = input.size(0) * weight.size(0);
  TORCH_CHECK(output_elements < (1LL << 30),
              "Convolution output must be smaller than 2 GiB");
  for (int i = 0; i < 3; ++i) {
    TORCH_CHECK(stride[i] > 0 && stride[i] <= 32 && padding[i] >= 0 &&
                    padding[i] <= 32 && weight.size(i + 1) <= 32,
                "Unsupported convolution stride, padding or kernel size");
    const int64_t extent =
        input.size(i + 1) + 2 * padding[i] - weight.size(i + 1);
    TORCH_CHECK(extent >= 0, "Convolution kernel exceeds padded input");
    output_spatial[i] = int(extent / stride[i] + 1);
    TORCH_CHECK(output_spatial[i] < (1LL << 30) / output_elements,
                "Convolution output must be smaller than 2 GiB");
    output_elements *= output_spatial[i];
  }
  c10::cuda::CUDAGuard guard(input.device());
  const auto *props = at::cuda::getDeviceProperties(input.get_device());
  TORCH_CHECK(props->major == 8,
              "Convolution candidate supports SM80-SM89 only");
  cutlass::conv::Conv3dProblemSize problem(
      int(input.size(0)), int(input.size(1)), int(input.size(2)),
      int(input.size(3)), int(input.size(4)), int(weight.size(0)),
      int(weight.size(1)), int(weight.size(2)), int(weight.size(3)),
      output_spatial[0], output_spatial[1], output_spatial[2], int(padding[0]),
      int(padding[1]), int(padding[2]), int(stride[0]), int(stride[1]),
      int(stride[2]), 1, 1, 1, cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  switch (tactic) {
  case 0:
    return run<128, 128, 32, 64, 64>(input, weight, bias, problem);
  case 1:
    return run<128, 64, 32, 64, 32>(input, weight, bias, problem);
  case 2:
    return run<64, 64, 64, 32, 32>(input, weight, bias, problem);
  case 3:
    return run<256, 64, 32, 64, 32>(input, weight, bias, problem);
  case 4:
    return run<128, 32, 32, 64, 32, 3>(input, weight, bias, problem);
  case 5:
    return run<128, 32, 64, 64, 32, 3>(input, weight, bias, problem);
  case 6:
    return run<64, 32, 64, 32, 32, 3>(input, weight, bias, problem);
  case 7:
    return run<128, 32, 32, 64, 32, 4>(input, weight, bias, problem);
  default:
    TORCH_CHECK(false, "Unknown convolution tactic");
  }
}
} // namespace

TORCH_LIBRARY(robotics_convolution, m) {
  m.def("tactic_count() -> int", []() { return int64_t(8); });
  m.def("fprop(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] "
        "padding, int tactic) -> Tensor");
}
TORCH_LIBRARY_IMPL(robotics_convolution, CUDA, m) { m.impl("fprop", fprop); }
