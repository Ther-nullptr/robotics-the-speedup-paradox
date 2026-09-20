// Robotics integer GEMM. CUTLASS visitor organization follows the reviewed
// QuaRot / Mini QServe implementation; see THIRD_PARTY_NOTICES.md.
// Independently parameterized formats, current-stream execution, and
// framework-owned workspace replace the original model-specific dispatch.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/arch/memory.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_residual_block.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/numeric_conversion.h>
#include <torch/extension.h>

namespace {
using namespace cute;

template <int Bits, int TileM, int TileN, int Stages>
at::Tensor run(const at::Tensor &a, const at::Tensor &sa, const at::Tensor &b,
               const at::Tensor &sb, const c10::optional<at::Tensor> &bias) {
  using Element =
      typename std::conditional<Bits == 4, cutlass::int4b_t, int8_t>::type;
  constexpr int TileK = Bits == 4 ? 128 : 64;
  using TB = cutlass::gemm::GemmShape<TileM, TileN, TileK>;
  using Warp = cutlass::gemm::GemmShape<32, 64, TileK>;
  using Inst = cutlass::gemm::GemmShape<16, 8, Bits == 4 ? 64 : 32>;
  using Output = cutlass::bfloat16_t;
  using Base = cutlass::gemm::device::Gemm<
      Element, cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor,
      int32_t, cutlass::layout::RowMajor, int32_t,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Inst>;
  using Map =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, Output,
                                                             8, 1>;
  using Acc = cutlass::epilogue::threadblock::VisitorAccFetch;
  using Row = cutlass::epilogue::threadblock::VisitorColBroadcast<
      Map, float, Stride<_1, _0, int64_t>>;
  using Col = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      Map, float, Stride<_0, _1, int64_t>>;
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      Map, Output, Stride<_0, _1, int64_t>>;
  using Mul = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using Add = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using Mul0 = cutlass::epilogue::threadblock::Sm80EVT<Mul, Acc, Row>;
  using Mul1 = cutlass::epilogue::threadblock::Sm80EVT<Mul, Mul0, Col>;
  using Sum = cutlass::epilogue::threadblock::Sm80EVT<Add, Mul1, Bias>;
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
      Map, Output, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, _1, int64_t>>;
  using Tree = cutlass::epilogue::threadblock::Sm80EVT<Store, Sum>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
      Base::kAlignmentA, Element, cutlass::layout::ColumnMajor,
      cutlass::ComplexTransform::kNone, Base::kAlignmentB, Output,
      cutlass::layout::RowMajor, 8, int32_t, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Inst, Tree,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
      typename Base::Operator, 1>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  int64_t m = a.size(0), n = b.size(0), k = a.size(1) * (Bits == 4 ? 2 : 1);
  auto output = at::empty({m, n}, a.options().dtype(at::kBFloat16));
  typename Row::Arguments row{sa.data_ptr<float>(), 0.0f, {_1{}, _0{}, m}};
  typename Col::Arguments col{sb.data_ptr<float>(), 0.0f, {_0{}, _1{}, n}};
  typename Bias::Arguments bias_args{
      bias ? reinterpret_cast<const Output *>(bias->data_ptr()) : nullptr,
      Output(0),
      {_0{}, _1{}, n}};
  typename Mul0::Arguments mul0{{}, row, {}};
  typename Mul1::Arguments mul1{mul0, col, {}};
  typename Sum::Arguments sum{mul1, bias_args, {}};
  typename Tree::Arguments callbacks{
      sum, {reinterpret_cast<Output *>(output.data_ptr()), {n, _1{}, m * n}}};
  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,
                                {int(m), int(n), int(k)}, 1, callbacks,
                                a.data_ptr(), b.data_ptr(), nullptr, nullptr, 0,
                                0, 0, 0, k, k, 0, 0, nullptr, nullptr, nullptr);
  Gemm op;
  TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess,
              "Unsupported integer GEMM alignment");
  auto bytes = Gemm::get_workspace_size(args);
  auto workspace = at::empty({int64_t(bytes)}, a.options().dtype(at::kByte));
  auto status = op(args, bytes ? workspace.data_ptr() : nullptr,
                   at::cuda::getCurrentCUDAStream(a.get_device()));
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status));
  return output;
}

at::Tensor integer_gemm_dispatch(const at::Tensor &a, const at::Tensor &sa,
                                 const at::Tensor &b, const at::Tensor &sb,
                                 const c10::optional<at::Tensor> &bias,
                                 int64_t bits, int64_t tactic) {
  TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && sa.is_cuda() && sb.is_cuda(),
              "CUDA tensors required");
  TORCH_CHECK(a.device() == b.device() && a.device() == sa.device() &&
                  a.device() == sb.device(),
              "Devices must match");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1),
              "Invalid packed matrix shapes");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && sa.is_contiguous() &&
                  sb.is_contiguous(),
              "Contiguous tensors required");
  TORCH_CHECK(a.scalar_type() == (bits == 4 ? at::kByte : at::kChar) &&
                  b.scalar_type() == a.scalar_type(),
              "Invalid integer storage dtype");
  TORCH_CHECK(sa.scalar_type() == at::kFloat &&
                  sb.scalar_type() == at::kFloat && sa.dim() == 1 &&
                  sb.dim() == 1 && sa.size(0) == a.size(0) &&
                  sb.size(0) == b.size(0),
              "Invalid scale buffers");
  TORCH_CHECK(a.size(0) > 0 && b.size(0) > 0 && b.size(0) % 8 == 0 &&
                  (a.size(1) * (bits == 4 ? 2 : 1)) % 128 == 0,
              "GEMM requires padded K%128=0, N%8=0 and nonempty matrices");
  if (bias)
    TORCH_CHECK(bias->device() == a.device() &&
                    bias->scalar_type() == at::kBFloat16 &&
                    bias->is_contiguous() && bias->dim() == 1 &&
                    bias->size(0) == b.size(0),
                "Invalid BF16 bias");
  c10::cuda::CUDAGuard guard(a.device());
  if (bits == 4) {
    if (tactic == 0)
      return run<4, 64, 128, 3>(a, sa, b, sb, bias);
    if (tactic == 1)
      return run<4, 128, 128, 3>(a, sa, b, sb, bias);
  } else {
    if (tactic == 0)
      return run<8, 64, 128, 3>(a, sa, b, sb, bias);
    if (tactic == 1)
      return run<8, 128, 128, 3>(a, sa, b, sb, bias);
  }
  TORCH_CHECK(false, "Unknown integer GEMM tactic");
}
} // namespace
TORCH_LIBRARY(robotics_integer, m) {
  m.def("gemm(Tensor a, Tensor sa, Tensor b, Tensor sb, Tensor? bias, int "
        "bits, int tactic) -> Tensor");
}
TORCH_LIBRARY_IMPL(robotics_integer, CUDA, m) {
  m.impl("gemm", &integer_gemm_dispatch);
}
