# Inference optimization experiments

This entry measures one complete policy call from a prepared CPU observation to
a CPU action chunk. Model loading, weight packing, compilation, CUDA Graph
capture and warmup are separate. Simulator/control-period timing is not included.
All console output and generated report labels are English.

Model execution code is maintained in `src/robotics_bench/models/`. PI0.5 owns its
PaliGemma/Gemma/SigLIP forwards; Cosmos owns its policy, sampler and DiT forwards.
Common framework services, Cosmos VAE/attention libraries and simulator packages
remain optional external dependencies. Original source hashes and licenses are
recorded alongside the imported code.

Use the existing case environment and resource variables from
`benchmarks/static/pi05_libero/paths.env.example` or
`benchmarks/static/cosmos_libero/paths.env.example`. The benchmark never downloads
weights. Prepare missing resources with `tools/prepare_resources.py` first.

```bash
python benchmarks/inference/run.py \
  --case pi05_libero \
  --input /path/to/prepared-observation.npz \
  --task 'pick up the alphabet soup and place it in the basket' \
  --gpu 0 --output-dir runs/optimization/pi05/trial-001 \
  --enable flow_loop --enable mask_cache --enable rope \
  --enable gated_residual --enable gelu_mul --enable norm --enable cuda_graph \
  --repeats 10 --profile \
  --profile-skill /path/to/profiler-visualizer \
  --render-python /path/to/render-env/bin/python
```

The input NPZ uses prepared LeRobot CPU observation keys for PI0.5; for Cosmos it
contains raw `primary_image`, `wrist_image`, and `proprio` arrays as required by
its simulator boundary. Additional `--validation-input` files and
`--validation-seed` values broaden the numerical comparison. Input files must
belong to the declared task; no alternate task instruction is inferred.

The entry always measures an unoptimized BF16 anchor. For low precision it also
measures BF16 with the same shared optimizations, then the selected candidate:

```bash
export CUDA_HOME=/path/to/compatible-cuda
# Add to the PI0.5 command above:
# --precision int8 --quant-scope text --enable shared_quant --enable activation_quant_fusion
```

Integer formats use signed INT8 or packed signed INT4, dynamic per-row activation
scales, per-output-channel weight scales, S32 accumulation and a BF16 epilogue.
Integer source and its pinned CUTLASS headers are packaged inside robotics;
the integer build does not use `ROBOTICS_CUTLASS_ROOT` or another project's kernels.
Activation prepare/pack and output conversion are part of the timing. Tactics 0
through 7 select explicit CUTLASS tiles; compare them on real shapes with
`--integer-tactic`. INT4/INT8 are not FP4/FP8, and low precision is not lossless.
FP4/FP8 sources and build tooling are preserved under
`src/robotics_kernels/blackwell/` and `tools/build_blackwell.py`; they require a
compatible Blackwell device/toolchain and separate hardware validation.

For Cosmos use `--case cosmos_libero`, its resource variables, and appropriate
switches: `modulation`, `gated_residual`, `cuda_graph`, and integer-specific
`shared_quant`/`activation_quant_fusion`. Its quantization scope is `dit`.

`norm_modulation_quant` additionally fuses non-affine LayerNorm with AdaLN
modulation and activation packing. It requires `modulation`, `modulation_quant`,
INT4 or INT8, and the `dit` scope. LIBERO and RoboCasa use the same owned kernel
and DiT adapter; their observations, checkpoints and VAE settings stay case-specific.
Only boundaries whose consumers are all integer projections use the fused path;
other boundaries retain native normalization. The kernel preserves BF16 rounding
between normalization, multiplication and addition, but its FP32 reduction order
can differ from native LayerNorm. Treat it as a numerical candidate, not a
guaranteed lossless switch, and inspect the recorded same-precision action drift.
It is disabled by default. Remove only `--enable norm_modulation_quant` to return
to native LayerNorm plus the existing modulation/packing fusion.

`residual_norm_modulation_quant` further combines the preceding gated residual
with normalization and packing at the cross-attention and MLP input boundaries.
It requires `norm_modulation_quant` and `gated_residual`. The BF16 residual is
still written for the later skip connection, while normalization consumes it
directly inside the kernel. Single-format integer consumers use this path;
mixed-format or unquantized boundaries keep the separate implementation. The
final MLP residual stays separate. This switch shares the normalization
candidate's numerical caveat and is also disabled by default.

Add the following to an existing Cosmos INT4/INT8 command that already enables
`modulation` and `gated_residual` to evaluate the complete fusion:

```bash
--enable modulation_quant --enable norm_modulation_quant --enable residual_norm_modulation_quant
```

Each result directory contains a manifest, repeated measurements, numerical
checks, checkpoint audit and a skill-compatible `ledger.json`. `--profile` adds
separate diagnostic traces/tables. With `--profile-skill`, every completed round
also produces timestamped PNG/SVG/Markdown, an interactive HTML page and its
manifest under `figures/`. Install CairoSVG/Cairo only in the rendering environment.

Exact fixed-input action equality is reported separately from quantization drift.
It does not establish task-suite success rate. Quantized configurations remain
experimental until the corresponding closed-loop evaluation passes. Historical
ratios from different measurement windows are not multiplied.

For a rollout, the existing case launchers accept `--model-runtime owned`,
repeatable `--enable`, `--precision`, `--quant-scope` and `--integer-tactic`.
The optimization defaults are disabled. `cuda_graph` on PI0.5 requires
`flow_loop`; caches/graphs are per model instance and assume immutable weights
and serialized calls. Reinstall the optimization context after changing weights.
The `native` runtime remains an explicit reference and rejects local switches.

Capture several real initial observations using the matching case environment:

```bash
python benchmarks/inference/capture_libero.py \
  --case pi05_libero --suite libero_object --task-id 0 \
  --initial-states 0,1,2 --gpu 0 \
  --output-dir runs/optimization/inputs/pi05-object
```

Pass `observation-000.npz` as `--input` and the other files as repeatable
`--validation-input` arguments. The capture manifest records task text and actual
initial-state IDs. Capturing observations runs the simulator; timing replay uses
the saved observations and does not run or advance the simulator.

`empty_image_cache` is an additional PI0.5 option requiring `cuda_graph`. It
reuses only the encoder output for known missing-camera placeholders, whose
pixels are fixed by the owned policy. Real observation images are always
re-encoded. Changing camera-presence roles invalidates the graph; cached
embedding storage is retained for the lifetime of its graph.

## Integer fusion and tuning

The implementation is under `src/robotics_kernels/ampere_ada/`; portable fusion
and graph helpers are under `common/`, while Blackwell keeps its own sources.

Add independent switches to the complete policy benchmark or case launcher:

```bash
# INT8 or INT4 uses actual integer Tensor Core GEMM:
# --precision int4 --integer-tactic 1 --quant-scope text
# --enable shared_quant --enable activation_quant_fusion
# --enable integer_pack_reuse
# Optional ablations:
# --enable integer_qkv
# --enable integer_gate_up       # PI0.5 gated MLP only
# --enable integer_group_views   # requires projection grouping
```

`integer_grouped` enables both projection group kinds. Mixed integer formats are
grouped only within compatible subsets. `integer_group_views` avoids explicit
output-slice copies; model consumers still determine any further layout work.
Group output and input caches belong to one serialized model call.

For a cohort with one loaded model, `--variants FILE` accepts a list of variant
objects with `id`, `precision`, `switches`, `scopes` and `tactic`. Omitted fields
inherit CLI settings. IDs must be unique safe filenames. Shared fusion switches
must match the CLI configuration so every low-precision row has a matching BF16
reference. Quantization-specific switches may differ. The benchmark also saves
validation action arrays and comparisons against the first same-precision row.
Large action drift is reported and does not stop finite-output performance tests.

Measure individual shapes separately from complete policy latency:

```bash
python benchmarks/inference/bench_integer.py --case both --gpu 0 \
  --pack-reuse --output-dir runs/optimization/integer/round-001
python benchmarks/inference/render_integer.py \
  --input runs/optimization/integer/round-001 \
  --skill /path/to/profiler-visualizer \
  --render-python /path/to/render-env/bin/python
```

Shapes come from recorded PI0.5 text and Cosmos DiT calls, including explicit
projection concatenations. `prepared_gemm` excludes activation preparation;
`complete_linear` includes it. Both use CUDA events around repeated graph nodes,
not policy wall time. The JSON retains every tactic and repeated sample; CSV
lists the best observed complete-Linear tactic per shape/precision. It does not
silently change a model's dispatch policy. Validate the chosen tactic in the
complete policy path on the same GPU.

Diagnostic charts distinguish integer and floating matrix work, matrix reduction,
copy/cast, normalization, quantization and pointwise arithmetic. GPU duration
sums remain separate from synchronized policy-service medians. Breakdowns with
more than four profiles are paginated with their original anchor repeated.

`integer_biasless` is an experimental specialized epilogue for Linear modules
without bias; real bias uses the normal epilogue. It is independently switched
and is not automatically selected from an insignificant timing difference.

PI0.5 additionally supports `condition_projection_cache`, requiring
`condition_cache` and `flow_loop`. It reuses adaptive-normalization projections
of registered, immutable timestep embeddings. Observation-dependent hidden
states are still recomputed. Weights and timestep-embedding settings must remain
unchanged within the optimization context; reinstall the context after changing
them. Graphs retain their condition/projection buffers through their lifetime.

For experiments changing shared optimizations between variants, explicitly add
`--ablate-shared`. The benchmark inserts a matching BF16 measurement for each
distinct shared-switch set. Without it, mismatched shared switches are rejected.

The [progressive quantization protocol](../../docs/protocols/progressive-quantization.md)
defines Cosmos `--progressive-sweep`, single `--quant-tier` settings and the
matched task-pilot entry `evaluate_cosmos.py`. The current adaptation retains
BF16 as the full-precision reference and measures its own speedups.

Current quantization work prioritizes Cosmos. PI0.5 uses `--quant-scope text`;
its action expert/diffusion scope is deferred. Older scope-expansion measurements
are exploratory records, not recommended presets.

Cosmos has an experimental `modulation_quant` switch, used together with integer
precision and `modulation`. It keeps the native LayerNorm reduction and fuses
BF16 modulation with dynamic activation quantization, avoiding an intermediate
BF16 activation write/read. Mixed tiers prepare only the formats needed by each
consumer group. The packed carrier stays inside the owned model; the engine's
CPU action interface is unchanged. Same-format action equality and complete
policy timings must both be checked; faster producer microbenchmarks alone do
not establish end-to-end benefit.

To inspect the actual per-layer formats recorded by a completed tier sweep:

```bash
# Uses the optional Matplotlib CPU environment.
python benchmarks/inference/render_progressive.py \
  --input runs/optimization/cosmos/tier-sweep-001
```

## RoboCasa and remaining Cosmos hotspots

`--case cosmos_robocasa` uses the `ROBOTICS_COSMOS_ROBOCASA_*` model-resource
variables, three cameras and H=32. It never falls back to the LIBERO checkpoint
or its H=16 configuration. Use the separate RoboCasa Python environment.

Capture a real reset observation and read its actual instruction from the
generated manifest before starting replay:

```bash
python benchmarks/inference/capture_robocasa.py \
  --task TurnOffMicrowave --layout-id 1 --style-id 1 --env-seeds 0 1 2 \
  --gpu 0 --output-dir runs/optimization/inputs/robocasa-001

python benchmarks/inference/run.py --case cosmos_robocasa \
  --input runs/optimization/inputs/robocasa-001/observation-000.npz \
  --task 'press the stop button on the microwave' --gpu 0 --steps 5 \
  --precision int4 --quant-scope dit --integer-tactic 1 \
  --enable modulation --enable gated_residual --enable cuda_graph \
  --enable shared_quant --enable activation_quant_fusion --enable integer_pack_reuse \
  --output-dir runs/optimization/robocasa/int4-001 --profile
```

Capture uses a physical numeric GPU index for RoboCasa EGL. It saves the actual
post-settling three-camera input, language, scene state and fingerprints. A
different scene may produce a different instruction; do not infer it from the
environment class name. Simulator execution is excluded from replay timings.

The following shared optimizations have separate switches and remain disabled
by default. Use matched BF16 controls when evaluating their quantized variants.

| Switch | Scope and boundary |
| --- | --- |
| `vae_norm_fusion` | Wan VAE native channel-norm reduction, followed by fused division, scale, affine and bias. BF16 rounding boundaries are retained. |
| `vae_silu_fusion` | Requires `vae_norm_fusion`; also fuses adjacent SiLU using a table generated by the installed Torch implementation. |
| `vae_spatial_padding` | Encoder-only experiment inspired by worldmodel/Wan: retain explicit causal time padding, move symmetric spatial padding into the convolution descriptor, and skip zero-padding copies. No cache survives between requests. |
| `vae_condition_prefix` | Shared LIBERO/RoboCasa action-only path: execute only complete causal encoder chunks needed by each case's conditioning prefix. Retain the full raw input, final latent projection, normalization and diffusion/noise shapes. Auxiliary clean-latent suffixes are not preserved. |
| `cross_kv_cache` | Cache native text K/V projections and norms within each policy request. Refresh fixed-address buffers outside the DiT graph; retain their storage in the graph and invalidate on layout changes. Query-dependent attention still runs. |
| `attention_sdpa_flash` / `attention_sdpa_efficient` / `attention_sdpa_math` / `attention_flash_attn` | Select exactly one dense DiT attention backend. QKV, RoPE and output projections remain unchanged; unsupported kernels raise rather than falling back. VAE attention keeps its native backend. |
| `conv_cutlass_c96_0` ... `conv_cutlass_c96_7` | Explicit VAE 96-to-96, 3x3x3, stride-one family ablation on channels-last 3D inputs. Other modules/layouts remain native with recorded reasons. `conv_cutlass_c96` is an alias for tactic 0. |
| `conv_cutlass_0` ... `conv_cutlass_7` | Experimental broader dispatch for supported BF16 Conv2d/3d modules. These are explicit tactics, not automatic fastest-kernel selection. |

CUTLASS convolution preserves the native causal padding and feature-cache
wrapper. Its complete-call timing includes required activation/output layout
conversions; weight packing is initialization work. Different reduction orders
in convolution or attention may change outputs. Record the actual action
comparison and full policy timing even when a microbenchmark looks faster.

To compare against cuDNN algorithm search, start a fresh benchmark process with
`--cudnn-benchmark`. This is a whole-cohort environment setting applied before
model loading, not a per-variant switch: toggling it after a convolution has run
does not invalidate PyTorch's cached execution plan. The manifest records
benchmark/benchmark_limit, deterministic and TF32 settings; never treat two
different cuDNN modes as one measurement cohort.

The [owned convolution notes](../../src/robotics_kernels/ampere_ada/CONVOLUTION.md)
describe the pinned CUTLASS iterator, layout, tile/pipeline choices and their
resource tradeoffs. CUDA code and build headers are maintained inside robotics.

```bash
python benchmarks/inference/bench_convolution.py \
  --trace runs/optimization/robocasa/int4-001/candidate-trace.json \
  --gpu 0 --limit 12 --warmup-ms 500 \
  --output-dir runs/optimization/robocasa/conv-tactics-001

# CPU-only selected breakdown; use names actually present in the cohort.
python benchmarks/inference/render_breakdown.py \
  --input runs/optimization/robocasa/int4-001 \
  --profiles original optimized-bf16 candidate \
  --skill /path/to/profiler-visualizer \
  --render-python /path/to/render-env/bin/python
```

The PNG adapter converts the skill's CSS HSL colors to equivalent RGB for
CairoSVG compatibility. The original SVG and raw durations are retained.

Integer GEMM already applies activation/weight scales, bias and BF16 conversion
inside its epilogue, without a separate INT32-result dequantization pass.
Activation quantization and packing still write GPU buffers before GEMM.
Shared QKV input preparation does not imply a grouped QKV GEMM: grouping must be
enabled explicitly. GEMM-to-residual and whole-row-quantization boundaries remain
separate optimization opportunities; CUDA Graph does not fuse those kernels.

The spatial-padding adapter is independently maintained in
`src/robotics_bench/optimizations/cosmos_vae_memory.py`. Its method reference is
worldmodel revision `51529dc48047c61c17bd25fc088c8569c005ddd1`,
`wan/modules/vae2_1.py`; that project is not a runtime dependency. The context
enters before any CUTLASS convolution plan reads padding and restores both the
forward method and padding on exit. Asymmetric downsample padding is untouched.
Changing the convolution descriptor may select a different kernel and change
floating-point rounding; fewer padding calls alone do not establish a speedup.

For the current action-only Cosmos path, the relevant VAE work is encoding;
future-image decoding is disabled. Worldmodel's steady decoder batching,
upsample folding, FP32 encoder kernels and SM110-specific C96 epilogues need
separate applicability checks. Preserve Cosmos's configured temporal window,
tail handling and per-request cache clearing when adapting further techniques.

`vae_condition_prefix` uses one adapter and encoder gate for the supported cases:

| Case | Raw / latent frames | Fixed conditioning frames | Action horizon | Camera keys |
| --- | --- | --- | --- | --- |
| LIBERO | 33 / 9 | 4 | 16 | `primary_image`, `wrist_image` |
| RoboCasa | 41 / 11 | 5 | 32 | `primary_image`, `secondary_image`, `wrist_image` |

With the configured 16-frame encoder window, both execute the first 1+16 raw
frames. LIBERO skips the final 16-frame chunk; RoboCasa skips the final 16+8
frames. Finite feature tensors replace omitted chunks before the unchanged full
final projection (9 or 11 latent frames). The engine request must match the
case selected by the model's geometry. The VAE normalization/SiLU switches also
share the same implementation across both cases. Enable them independently with
`--enable vae_norm_fusion --enable vae_silu_fusion --enable vae_condition_prefix`.
The policy-specific encode method still executes, including
its RNG seeding and cache clearing. Every denoise call checks that the omitted
suffix remains unconditioned. A changed/invalid mask raises before actions are
delivered. This is an **engine action-output** optimization: callers consuming
the private `orig_clean_latent_frames` suffix must not enable it. Direct VAE
encoding outside the engine's explicit action request uses the native path.

`cross_kv_cache` does not cache queries or attention outputs, and recomputes K/V
for every request, including repeated prompts. Same-request context changes are
rejected; inference tensors without version counters are compared to a snapshot.
Those checks, snapshot copies and buffer refreshes are included in complete-policy
timing. Nonidentity text projections, image cross-attention, training and context
parallelism are unsupported. Weights and precision must remain immutable while
the optimization context is active. The engine and each cache require serialized
calls; use distinct model instances for concurrent requests.
