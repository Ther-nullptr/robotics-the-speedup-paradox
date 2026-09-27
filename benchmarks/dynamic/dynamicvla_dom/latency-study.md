# Streaming latency versus task success

This optional experiment holds DynamicVLA weights, precision, refinement steps,
action horizon and native DOM physics fixed while increasing worker service time.
It studies **synthetic extra latency**, not the speed of different hardware or
quantization quality. The native streaming queue and0.04s physics grid remain
unchanged; there is no paper_sync protocol.

## Measurements and intervention

The three study switches are off by default:

- `--measure-inference`: synchronize CUDA around a real model generation and
  record CPU-batch-to-CPU-action service time separately from native pacing.
- `--extra-delay-ms D`: wait D wall milliseconds after actions become available
  on CPU, then publish the result. The worker also waits before starting another
  inference, so this models slower service rather than only a delayed network.
- `--episode-seed-mode`: reseed model sampling with `seed + episode_id` at reset,
  matching simulator trial seeds. Default native runs retain continuous RNG.

All treatment cells and the zero-delay baseline use the same measured/paired
protocol. Instrumentation can change scheduling, so the zero-delay study baseline
is separate from an uninstrumented native smoke. Same-input validation checks that
instrumentation does not alter action values. Subsequent closed-loop inputs and
random-number consumption can diverge as a consequence of changed scheduling.

The generation event records these boundaries:

| Field | Definition |
| --- | --- |
| `model_compute_ms` | Synchronized generation call with GPU-resident inputs; excludes original pacing and artificial delay |
| `worker_compute_ms` | CPU prepared batch through input transfer, generation, delta conversion and CPU action return; excludes actual pacing |
| `native_pacing_ms` | Actual native dt_scale sleep; retained as a separate component |
| `extra_delay_requested_ms`, `extra_delay_actual_ms` | Requested/measured additional wall delay |
| `compute_ready_wall_s`, `delay_started_wall_s`, `released_wall_s` | CPU output ready, extra-delay section start, and release gate opened before queue transport |
| `post_compute_overhead_ms` | Measured bookkeeping or host scheduling between compute completion and the delay section |
| `chunk_observation_*` | Observation that generated this chunk, retained through queue replacement and control holds |
| `action_age_sim_ms`, `action_age_wall_ms` | Age of that generating observation when a control command is applied |

Client raw-image preprocessing is outside `worker_compute_ms`. The independent
benchmark below includes it. Do not add artificial delay to a reported physical
GPU compute latency. Effective service is measured from worker start to release gate, including explicit host overhead. Native dt_scale compensation remains based on its original
host span; it is not exact simulator-time latency emulation. Use observed action
age and physics/control cadence to interpret results.

A horizon is20*40ms=800ms for the local DOM checkpoint. If every returned chunk is
already expired, zero applied actions is a valid task outcome only when the
completed cell has episode-bound generation evidence after the terminal ACK. A result may finish after the scene terminates; the client drains it before acknowledging. The study records
`policy_starved`; missing evidence, disconnection or worker exceptions remain
infrastructure errors and stop the cell rather than reducing success rate.

## Run a paired study

Load the same explicit offline resources as the [native case](README.md), then:

```bash
source .local/dynamicvla-dom.env
python benchmarks/dynamic/dynamicvla_dom/study.py \
  --task-json /path/to/DOM/tests/scene-a.json \
  --task-json /path/to/DOM/tests/scene-b.json \
  --task-json /path/to/DOM/tests/scene-c.json \
  --delays-ms 0,100,200,400,800 --episodes 10 --seed 42 \
  --num-steps 10 --model-gpu 0 --sim-gpu 1 \
  --output-dir runs/dynamic/dynamicvla_dom/streaming-latency-001
```

Add `--dry-run` to validate all cells without launching GPU workers or creating
outputs. The matrix defaults to native streaming, measured inference and paired
episode seeds. Conditions run in a fixed randomized order on the same GPU pair;
no foreign jobs are terminated. Cells wait for clean idle GPUs.

`study.json` binds complete task/condition coverage and source/config identity;
`state.json` tracks cell progress. The driver resumes completed audited cells and
uses new attempt directories for retries. A file lock prevents concurrent supervisors, and the resume identity includes model weight contents. Changing a resource, seed, condition or
runtime source requires a new study directory. Infrastructure failure stops the
matrix for investigation. Model load/startup/warmup are excluded from generation
samples; all episode videos remain recorded unless `--no-record-video` is given.

For long studies, run the command under tmux and redirect output to a log. Stop the
study supervisor with SIGTERM for worker cleanup; check its state/log before
resuming. After every completed cell, `analysis/` is refreshed automatically.

## Independent complete-policy timing

Use a saved raw observation and its instruction metadata from a native run:

```bash
"$ROBOTICS_DYNAMICVLA_PYTHON" benchmarks/inference/benchmark_dynamicvla.py \
  --checkpoint "$ROBOTICS_DYNAMICVLA_CHECKPOINT" \
  --input /path/to/run/observations/000000.npz \
  --metadata /path/to/run/observations/000000.json \
  --steps 10 --gpu 0 --warmup 5 --repeats 21 \
  --output-dir runs/inference/dynamicvla/fixed-input-001
```

This benchmark synchronizes before/after the complete raw-CPU-observation to
CPU-action-chunk call. It excludes loading, warmup, simulator, video and both
pacing/artificial sleeps. It is representative-input timing, not a substitute
for all rollout requests. Different refinement-step experiments change model
quality as well as latency and must be reported separately from pure delay.

## Analysis

```bash
PYTHONPATH=src python -m robotics_bench.dynamicvla_dom.analysis \
  --study runs/dynamic/dynamicvla_dom/streaming-latency-001/study.json \
  --output-dir runs/dynamic/dynamicvla_dom/streaming-latency-001/analysis
```

The CPU-only analyzer recomputes completed ledgers and checks generation/applied
chunk mapping, source/checkpoint/task/device identities, control budgets and timing
boundaries. It outputs JSON, per-cell/per-task/pooled CSV and an English report:

- Success rate with descriptive Wilson95% intervals, and paired seed gains/losses
  versus zero added delay. These fixed scenes do not establish full DOM accuracy.
- Successful-episode generated/applied mean chunks (null at zero successes).
- Failure-budget mean control steps, held-control fraction and policy starvation.
- Real compute, native pacing, added delay and effective service median/P95.
- Actual observation age and expired actions with explicit denominators.

Incomplete conditions are visible but have no aggregate accuracy. Pooling requires
all declared tasks at a condition. Latency distributions weight actual generation
samples; action age weights applied control steps. Successful-only chunk averages
can change because the subset of successful episodes changes, not just because
execution becomes faster. No task speedup is inferred from these counts alone.

Render a standalone figure after audited results exist:

```bash
"$ROBOTICS_DYNAMICVLA_PYTHON" benchmarks/dynamic/dynamicvla_dom/plot_study.py \
  --results runs/dynamic/dynamicvla_dom/streaming-latency-001/analysis/results.json \
  --output-dir runs/dynamic/dynamicvla_dom/streaming-latency-001/analysis
```

Matplotlib is an optional plotting dependency. The PNG/PDF distinguish effective
service latency, actual worker compute and generating-observation age; no latency
or accuracy point is inferred for an incomplete cell.
