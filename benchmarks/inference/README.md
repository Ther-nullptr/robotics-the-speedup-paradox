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
