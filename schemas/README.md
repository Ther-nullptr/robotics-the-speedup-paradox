# Protocol contracts, version 1.0.0

These schemas validate serialized run settings and trace records. They do not run a policy, simulator, scheduler, or hardware profiler. A passing trace does not demonstrate control performance or paper reproduction.

From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python tools/validate_contracts.py --examples
python -m pytest -q
```

To check your own complete run:

```bash
python tools/validate_contracts.py --manifest run.manifest.json --trace run.trace.jsonl
```

`--examples` and `--manifest/--trace` are mutually exclusive. Invalid artifacts return exit status 1 with a diagnostic; command usage errors return 2. The validator only reads artifacts and resolves its bundled schemas locally.

## Manifest

`run-manifest.schema.json` fixes `schema_version` to `1.0.0`. It requires the run identifier, an explicit `synthetic` flag, protocol settings, clock-domain declarations, episode IDs/seeds, and a task budget. This is a minimal protocol contract, not a full provenance manifest for trained models or benchmark results.

The schedule axis is independent of the clock/delay axis:

| Schedule | Meaning |
| --- | --- |
| `sync` | Sequential observation, request, output and action operation |
| `history_observation` | A synchronous control flow supplies historical observations; requires `history_steps` |
| `async` | The producer declares overlap scheduling |

| Clock | Allowed delay modes |
| --- | --- |
| `virtual` | `zero`, `profile_replay` |
| `realtime` | `measured`, `additive`, `target_total` |

All three schedules can be combined with all five valid clock/delay pairs. `profile_replay` requires `profile_id`. Chunk prediction length and execution length are separate, with execution length no larger than prediction length. The validator does not infer historical-observation offsets or prove that an `async` producer actually overlaps work.

Each `clock_domains` key identifies an independent origin, with a `kind`, `process_id`, and `monotonic_origin`. Reuse of an ID means reuse of the same origin for the whole run; an episode reset must not reset that domain's timestamp. Cross-machine/process clocks need separate IDs unless an external alignment protocol establishes a shared origin. No duration is computed across domains.

`budget.clock_domain` selects the simulation or host domain, and `start_event` is explicitly `first_observation_sampled`. A producer must provide an observation timestamp in that domain to calculate elapsed budget time. The optional `host_watchdog` is separate and requires a host-monotonic domain. This validator checks declarations, not budget enforcement or clock synchronization.

Optional `metrics` describe a summary with `trials`, `successes`, `successful_task_time_mean_ns`, and `successful_chunk_count_mean`. Successes cannot exceed trials; when there are no successes both means must be null. Summary values are not recomputed from traces in this version. Passing their schema checks does not establish accurate experimental metrics.

## Trace

`trace-event.schema.json` describes one JSON object per JSONL line. Blank lines are ignored. JSON numbers must be finite; nanosecond timestamps are nonnegative integers. Every event includes:

```text
schema_version, run_id, episode_id, event_seq,
event_type, clock_domain, timestamp_ns
```

`event_seq` increases strictly in serialized run order; gaps are allowed. Within each declared clock domain, timestamps do not go backwards. Timestamps in distinct domains can have unrelated values. Producers must serialize their records accordingly; this is not an unordered distributed-log merger.

The validator checks these associations and lifecycle conditions:

- An observation exists before its request is submitted, and both belong to the same episode.
- A request starts before completion. Successful completion assigns a unique chunk. Error/cancelled completion has no chunk.
- A chunk completes successfully before release and is released before action execution. Release refers to its request's chunk and uses the manifest's delay mode/profile ID.
- Each action refers to the chunk's original observation and an index below the declared execution length. It starts once and finishes only after a start.
- Dropped output cannot execute. A pending, started request may complete after being dropped so that a late callback can be recorded.
- Episodes run sequentially. A new episode requires the previous episode's terminal boundary. Episode IDs and observation/request/chunk IDs are unique within a run.
- Each declared episode has exactly one `episode_finished`. New observations/requests, inference starts, releases and action events are rejected after terminal. Completion/drop diagnostics and underrun diagnostics can still be recorded to describe cleanup.

Supported event types are `observation_sampled`, `inference_submitted`, `inference_started`, `inference_completed`, `result_released`, `action_started`, `action_finished`, `queue_underrun`, `result_dropped`, and `episode_finished`. Event-specific payloads are defined by the schema. Unknown properties are rejected to catch misspellings and silent semantic changes.

`result_released.delay_ns` is a declared delay sample: total modeled delay for `profile_replay`, extra delay for `additive`, or target total for `target_total`. It is not an observed age or an automatically recomputed wait. `overshoot_ns` records the producer's overshoot and `profile_id` is null when no profile is configured. The validator checks their types and associations, not whether runtime timing implements those numbers. A no-injection mode should record a zero delay sample; mode-specific timing equations are deferred.

Actual `observation_age` uses observation and `action_started` timestamps in the same domain. Actual boundary wait uses action start minus the preceding chunk boundary, including release gating. `inference_completed` does not substitute for `result_released` when injection postpones availability. These quantities are intentionally not filled from the paper's residual-delay formula.

This version does not validate physical dynamics, numerical model quality, clock alignment, injection arithmetic, real-time deadlines, queue capacity, single-flight scheduling, complete resource drain, or metric estimates. Those require future producer implementations and additional behavior tests. Contract extensions should include a version decision, example data and acceptance/rejection tests.
