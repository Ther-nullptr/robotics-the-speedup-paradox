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
export ROBOTICS_CUTLASS_ROOT=/path/to/compatible-cutlass
export CUDA_HOME=/path/to/compatible-cuda
# Add to the PI0.5 command above:
# --precision int8 --quant-scope text --enable shared_quant --enable activation_quant_fusion
```

Integer formats use signed INT8 or packed signed INT4, dynamic per-row activation
scales, per-output-channel weight scales, S32 accumulation and a BF16 epilogue.
Activation prepare/pack and output conversion are part of the timing. Tactics 0
and 1 select explicit CUTLASS tiles; compare them on real shapes with
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
