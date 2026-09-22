# CUTLASS convolution candidates

The implementation is maintained in this package: [convolution.py](convolution.py)
owns packing and layout handling, [convolution_fprop.cu](csrc/convolution_fprop.cu)
owns the CUDA dispatch, and [the VAE context](../../robotics_bench/optimizations/cosmos_convolution.py)
owns selective application and restoration. The public causal wrapper retains
cache concatenation and asymmetric temporal padding. `forward_packed` consumes
NDHWC activations and KTRSC filters; complete calls include required activation
layout copies and preserve the input memory format. Bias is broadcast within
the BF16-output, FP32-accumulator epilogue.

This candidate is numerically approximate relative to native PyTorch convolution,
even though both expose BF16 tensors. The pinned
[PyTorch cuDNN dispatch](https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/Convolution.cpp)
adds bias after the convolution output has been rounded to BF16. This candidate
adds bias to the FP32 accumulator before the final BF16 conversion. That removes
a bias pass but changes a rounding boundary, in addition to possible reduction-order
differences. Operator tests use explicit tolerances; full-policy action drift is
reported separately. Do not classify this convolution replacement as lossless.

CUTLASS headers are an unmodified, independently vendored copy of NVIDIA/CUTLASS
revision `982748aa7356fa838c2ea4994ddcb0b2a4b4cefa`. Its BSD-3-Clause license and
file-specific notices remain in [LICENSE.txt](third_party/cutlass/LICENSE.txt)
and the headers. [PROVENANCE.json](third_party/cutlass/PROVENANCE.json) records
the revision and file hashes. No external project supplies runtime kernels or
build headers.

## What the implementation takes from CUTLASS

Implicit GEMM constructs activation tiles during loads rather than materializing
an `im2col` tensor. Here the logical GEMM dimensions are `M=batch*Z*P*Q`,
`N=output_channels`, and `K=input_channels*T*R*S`.
See the [official convolution guide](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html).

The pinned [optimized activation iterator](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/conv/threadblock/conv3d_fprop_activation_tile_access_iterator_optimized.h)
uses initial predicates and fast-divmod indexing. Its
[parameter object](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/conv/threadblock/conv3d_params.h)
precomputes pointer increments for the next S, R, T, and C position.
The [multistage mainloop](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/conv/threadblock/implicit_gemm_multistage.h)
pipelines asynchronous global-to-shared copies with Tensor Core work.
The [kernel composition](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/conv/kernel/default_conv3d_fprop.h)
exposes threadblock shape, warp shape, and pipeline depth independently.

## Explicit tactics

All tactics retain generic shapes and the same checked mathematical operation.
The caller chooses an ID; there is no runtime table claiming a fastest kernel.
The separate `c96` VAE policy selects only 96-to-96, 3x3x3, stride-one modules
with channels-last 3D inputs. Other modules/layouts stay native with coverage
and fallback reasons recorded. Generic kernels also support other aligned
channel counts and validated strides.

| ID | Threadblock M,N,K | Warp M,N,K | Stages | Warps/CTA | A+B staging estimate, KiB | Accumulator FP32 words/thread |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | 128,128,32 | 64,64,32 | 3 | 4 | 48 | 128 |
| 1 | 128,64,32 | 64,32,32 | 3 | 4 | 36 | 64 |
| 2 | 64,64,64 | 32,32,64 | 3 | 4 | 48 | 32 |
| 3 | 256,64,32 | 64,32,32 | 3 | 8 | 60 | 64 |
| 4 | 128,32,32 | 64,32,32 | 3 | 2 | 30 | 64 |
| 5 | 128,32,64 | 64,32,64 | 3 | 2 | 60 | 64 |
| 6 | 64,32,64 | 32,32,64 | 3 | 2 | 36 | 32 |
| 7 | 128,32,32 | 64,32,32 | 4 | 2 | 40 | 64 |

The staging estimate is `2*(TileM+TileN)*TileK*Stages` bytes. It excludes layout
padding and epilogue shared storage. Accumulator words exclude pointers,
predicates, operand fragments, and compiler temporaries; they are not measured
register counts. Inspect the compiled kernel's resource usage and actual shared
storage before deriving occupancy. More stages or larger tiles can lower CTA
residency, even when they improve reuse or hide load latency.

For the motivating `input_channels=output_channels=96`, a 32-wide N tile has no
output-channel tail; 64/128-wide tiles allocate 128 columns. Narrow N also creates
more CTAs and repeats activation loads, so removing the tail is not sufficient
evidence of acceleration. According to the pinned
[3D loop-count function](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/conv/conv3d_problem_size.h),
3x3x3 fprop uses `27*ceil(96/TileK)` iterations: 81 or 54 for K=32 or 64.
K=64 performs masked work for 32 of 128 allocated channels. Tactics 4/5 compare
that loop-overhead/padding tradeoff, 5/6 compare the M tile and accumulator size,
and 4/7 isolate pipeline depth. These are analytical tradeoffs, not measured
speedups.

A K=128 candidate was rejected at compilation: this pinned BF16
`DefaultMmaCore` specialization sets its shared-memory crosswise extent equal
to TileK. The [underlying layout](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/layout/tensor_op_multiplicand_sm75.h)
requires that extent to fit within a 128-byte line (64 BF16 values). Supporting
K=128 here requires a different shared-memory layout and iterator composition;
it cannot be enabled by changing this template argument alone.

## Measurement boundary

Use [bench_convolution.py](../../../benchmarks/inference/bench_convolution.py)
with an actual profiler trace and explicit `--tactics 0 1 2 3 4 5 6 7`.
It preserves recorded input strides, emits unsupported coverage, checks numerical
error, and records device time, synchronized eager wall time, and one-time weight
packing separately. Sustained warmup defaults to 500 ms per scope; raw samples
and unlocked GPU clock/power state are retained. Inspect temporal drift and
repeat comparisons before selecting a tactic for complete policy measurement.

Compare cuDNN default and `--cudnn-benchmark` in fresh processes, with the flag
set before any convolution. Changing it inside one process is not a reliable
plan-cache reset. Restart after changing the compiled tactic set; the loader
rejects an older extension already loaded in the process. Validate complete
policy output/latency after any selected kernel change; microbench speedups and
BF16 tolerances do not establish task success rates.
