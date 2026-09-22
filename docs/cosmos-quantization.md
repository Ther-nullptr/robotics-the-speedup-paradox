# Cosmos quantization and VAE optimization

The current Cosmos quantization implementation phase is complete: fixed W8A8
and W4A4 inference, progressive DiT coverage, owned kernels, and shared LIBERO /
RoboCasa VAE optimizations are integrated. This is an implementation and
performance milestone; quantized full-suite task quality is still unverified.
All optimization switches remain explicit and disabled by default.

## Supported boundary

| Item | Implemented and evaluated | Limit |
| --- | --- | --- |
| Models | Cosmos-Policy-LIBERO-Predict2-2B and Cosmos-Policy-RoboCasa-Predict2-2B | Each case requires its own checkpoint, statistics and text cache |
| Integer scope | 280 Linear sites in 28 DiT blocks: self/cross-attention Q/K/V/out and both MLP projections | VAE and sensitive noncandidate operations retain native precision |
| Formats | W8A8 and W4A4, symmetric signed integers; per-output-channel weight scales and dynamic per-row activation scales | No calibration, rotation, smoothing or fine-tuning is applied |
| GEMM | Integer Tensor Core accumulation in INT32; fused scale/bias conversion to BF16 | Activation preparation and layout costs remain part of complete-call timing |
| Hardware | RTX 6000 Ada, SM89; PyTorch 2.7.0+cu128 and CUDA 12.8 development toolkit | Ampere requires its own device validation; preserved Blackwell FP4/FP8 source is not an SM89 result |
| Execution | One serialized model instance, CPU observation to CPU action chunk | No concurrent requests, planning or future-image decoding in the optimized action boundary |

Integer source and the pinned CUTLASS headers are maintained inside this package.
There is no runtime import from a sibling VLM project or local quantization
prototype. See the [kernel guide](../src/robotics_kernels/README.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md) for build and provenance details.
Loading frameworks, the compatible Cosmos VAE/attention libraries, simulator
packages, weights and assets remain separate dependencies.

## Starting recipe for performance experiments

The recipe below retains native DiT normalization and attention. Quantization
itself is lossy. Fixed-input equality for a fusion is always assessed against the
same-precision reference, not used as a claim that INT4 equals BF16.

| Switch | Purpose |
| --- | --- |
| `modulation`, `gated_residual`, `cuda_graph` | Shared pointwise fusion and DiT graph replay |
| `shared_quant` | Reuse prepared activations across compatible projection consumers |
| `activation_quant_fusion` | Fuse the MLP activation with integer preparation |
| `integer_pack_reuse` | Reuse row data during INT4 quantization and nibble packing |
| `modulation_quant` | Fuse BF16 modulation with integer preparation, keeping native LayerNorm |
| `vae_norm_fusion`, `vae_silu_fusion` | Keep native VAE channel reduction and fuse subsequent pointwise work |
| `vae_condition_prefix` | Skip complete encoder chunks outside the required conditioning prefix |

Tactic `1` is the measured starting choice on this Ada device, not a universal
dispatch optimum. Select the target architecture explicitly when building;
`TORCH_CUDA_ARCH_LIST=8.9` describes the measured Ada machine only.

Activate the case's model environment and load its resource variables using the
[environment guide](environment_setup.md). Obtain a raw observation with
[capture_libero.py](../benchmarks/inference/capture_libero.py) or
[capture_robocasa.py](../benchmarks/inference/capture_robocasa.py). Use the task
instruction recorded in that capture's manifest; RoboCasa obtains it after reset.

```bash
python benchmarks/inference/run.py \
  --case cosmos_libero --gpu 0 \
  --input /path/to/observation.npz --task 'the matching captured instruction' \
  --steps 5 --seed 195 --warmup 5 --repeats 13 \
  --precision int4 --quant-scope dit --integer-tactic 1 \
  --enable modulation --enable gated_residual --enable cuda_graph \
  --enable shared_quant --enable activation_quant_fusion \
  --enable integer_pack_reuse --enable modulation_quant \
  --enable vae_norm_fusion --enable vae_silu_fusion --enable vae_condition_prefix \
  --output-dir runs/optimization/cosmos-libero/int4-001 --profile
```

Use `--precision int8` for W8A8. For RoboCasa, run a separate process with
`--case cosmos_robocasa`, its resource variables, its captured three-camera input
and a new output directory. The benchmark selects H16 or H32 from the case;
it does not reuse LIBERO observations for RoboCasa.

This command measures an original BF16 anchor, BF16 with the same shared
optimizations, and the requested integer candidate. To compare the VAE switches
off/on in one loaded-model cohort, use `--variants` with `--ablate-shared`; an
additional matched BF16 reference is inserted for each shared configuration.
The [benchmark guide](../benchmarks/inference/README.md) documents this interface,
validation inputs/seeds, and optional profile-visualizer rendering.

Static case launchers accept the same `--model-runtime owned`, `--precision`,
`--quant-scope dit`, `--integer-tactic` and repeated `--enable` arguments.
Keep the legacy protocol argument `--quant none`; actual model arithmetic is
selected by `--precision`. Full launch commands, simulator inputs and output
locations are in the [LIBERO](../benchmarks/static/cosmos_libero/README.md) and
[RoboCasa](../benchmarks/static/cosmos_robocasa/README.md) case guides.

## Shared VAE implementation

[cosmos_prefix.py](../src/robotics_bench/optimizations/cosmos_prefix.py) contains
one encoder gate and request lifecycle. A small case table validates these
different contracts:

| Case | Raw / latent frames | Conditioning frames | Encoder frames executed | Action horizon |
| --- | --- | --- | --- | --- |
| LIBERO | 33 / 9 | 4 | 17, with the native 16-frame window | 16 |
| RoboCasa | 41 / 11 | 5 | 17, with the native 16-frame window | 32 |

The adapter preserves the full raw input, final latent projection, normalization,
noise/DiT dimensions, native RNG behavior and per-request cache clearing. Every
denoise checks the conditioning mask. The shared
[VAE pointwise adapter](../src/robotics_bench/optimizations/cosmos_pointwise.py)
and [kernel](../src/robotics_kernels/common/vae.py) serve both cases as well.

The supported output is the engine's action chunk. The omitted auxiliary
`orig_clean_latent_frames` suffix is not preserved, so planning or future-image
consumers must not enable the prefix switch. Direct VAE calls outside an explicit
engine action request retain native encoding. Simulator rollout video recording
is independent of this private model output.

## Measured snapshot

The following existing measurements were collected on 2026-09-22, on the Ada
setup above, with 5 sampling steps, 5 warmups and 13 timed repetitions. Values are
complete policy-call medians in milliseconds; loading, weight packing,
compilation and warmup are excluded. Diagnostic GPU kernel sums are a separate
metric and are not substituted for these wall times.

| Case / cohort | Original BF16 | Shared optimized BF16 | W8A8 | W4A4 |
| --- | ---: | ---: | ---: | ---: |
| LIBERO / `round8-shared-vae` | 436.52 | 381.94 | 214.36 | 167.12 |
| RoboCasa / `round11-shared-vae` | 496.19 | 416.92 | Not measured in this cohort | 188.57 |

All optimized columns include the declared VAE recipe. Within the LIBERO cohort,
adding the VAE bundle to the preceding INT4 recipe reduced latency from 222.88
to 167.12 ms (1.334x); the reversed-order repeat was 222.38 vs 167.29 ms.
The direct original-BF16-to-INT4 ratio is 2.612x. LIBERO convolution-family GPU
work fell from 46.85 to 24.33 ms and other elementwise work from 34.84 to 8.78 ms.
The incremental result belongs to the VAE bundle, not to a separately isolated
prefix or pointwise switch.

RoboCasa's prefix-off/on INT4 comparison was 253.28 vs 188.57 ms, with VAE
pointwise fusion enabled on both sides. This revalidated the existing RoboCasa
optimization after sharing the adapter; it is not another gain on top of its
previously enabled prefix path. The fixed scenario was TurnOffMicrowave,
layout/style 1/1, with the reset instruction "press the stop button on the microwave".

The LIBERO snapshot used one saved observation and seeds 0/42/195; RoboCasa used
three saved observations and those three seeds. VAE changes produced exact
same-precision actions for all 3 and 9 checks respectively. The LIBERO timing
instruction was "put both the alphabet soup and the tomato sauce in the basket".
These are bounded observations, not full-suite success rates or portable latency
guarantees. Quantized task quality remains experimental.

Raw traces, action arrays, inputs, measurements, ledgers and generated figures
remain outside Git. Each benchmark output directory records its source and input
hashes, active switches, actual kernel coverage and numerical comparisons. The
cohort names above identify the existing evidence; reproducing a new cohort uses
the documented commands and its own captured inputs.

## Experimental alternatives and rollback

| Option | Current conclusion |
| --- | --- |
| `norm_modulation_quant`, `residual_norm_modulation_quant` | Small measured gains in some cohorts; altered LayerNorm reduction order changed actions. Explicit numerical candidates, excluded from the starting recipe |
| `cross_kv_cache` | Correct fixed-input results, but request refresh/check/copy overhead outweighed projection savings in the measured RoboCasa cohort |
| Alternative attention backends | No measured improvement over the native backend in the audited RoboCasa workload; some changed numerical results |
| `conv_cutlass_*`, `vae_spatial_padding` | Owned experiments retained; microkernel savings did not establish a complete-policy advantage over the selected native convolution path |
| Projection grouping / additional tactics | Optional shape-dependent candidates; not automatically enabled by the precision choice |

Remove a switch to disable that intervention. For an original owned BF16
reference, use `--precision bf16` and omit all `--enable` options. For the external
runtime reference, also set `--model-runtime native`; it rejects local
optimizations. Do not change precision or weights inside an active graph/cache
context; reinstall the optimization context and keep requests serialized.

The [progressive quantization protocol](protocols/progressive-quantization.md)
defines LIBERO tiers 0..10, their coverage and paired task pilot entry. The preset
pilot driver has its own explicit recipe; it does not automatically inherit the
VAE switches above. Control-step and paper-model speedups follow the separate
[speedup protocol](protocols/speedup-metrics.md), including failure-budget totals
and omission of success-conditioned speedups when a configuration has no successes.

This phase does not claim calibrated quantization, retraining, full-suite quality,
new hardware validation, or a global performance limit. Those remain separate
work items. PI0.5 quantization remains limited in priority to its text/LLM path;
its diffusion/action expert is deferred.
