"""Run a fixed RoboCasa task and scene with Cosmos Policy, one environment at a time."""

from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load_file(
    Path(__file__).parent.parent / "cosmos_libero/run.py", "_cosmos_run_helpers"
)


def reference_mismatches(reference, index, actual):
    expected = json.loads(
        (
            Path(reference) / "initializations" / f"{index:06d}" / "episode.json"
        ).read_text()
    )
    keys = (
        "task",
        "init_state_id",
        "env_seed",
        "layout_id",
        "style_id",
        "description",
        "initial_state_sha256",
        "initial_xml_sha256",
        "initial_observation_sha256",
    )
    return [key for key in keys if expected.get(key) != actual.get(key)]


def check_reference(reference, index, actual):
    mismatch = reference_mismatches(reference, index, actual)
    if mismatch:
        raise RuntimeError(f"Episode initialization differs from reference: {mismatch}")


class ReferenceReset:
    """Bounded observation-only retries before any policy request or control step.

    Every accepted reset must exactly match the reference. State, XML, language
    and identity mismatches are fatal immediately; no tolerance is introduced.
    """

    def __init__(self, simulator, reference, output, retries, snapshots=None):
        self.simulator = simulator
        self.reference = reference
        self.output = Path(output)
        if type(retries) is not int or not 0 <= retries <= 16:
            raise ValueError("Initialization retries must be in 0..16")
        self.retries = retries
        self.snapshots = snapshots
        self._metadata = None
        self._initial_frame = None
        self._rendered_observation = None
        self._consumed_observation = None

    def __getattr__(self, name):
        return getattr(self.simulator, name)

    @property
    def episode_metadata(self):
        return (
            self._metadata
            if self._metadata is not None
            else self.simulator.episode_metadata
        )

    def render(self):
        return (
            self._initial_frame.copy()
            if self._initial_frame is not None
            else self.simulator.render()
        )

    def step(self, action):
        self._initial_frame = None
        return self.simulator.step(action)

    def save_initialization(self, directory):
        self.simulator.save_initialization(directory)
        if self._consumed_observation is not None:
            import numpy as np

            metadata_path = Path(directory) / "episode.json"
            metadata = json.loads(metadata_path.read_text())
            metadata.update(
                {
                    key: self._metadata[key]
                    for key in (
                        "initial_observation_sha256",
                        "initial_rendered_observation_sha256",
                        "initial_observation_replay",
                    )
                }
            )
            shared.write_json(metadata_path, metadata)
            np.savez(
                Path(directory) / "rendered-observation.npz",
                **self._rendered_observation,
            )
            np.savez(
                Path(directory) / "consumed-observation.npz",
                **self._consumed_observation,
            )

    def reset(self, init_state_id, seed=None):
        self._metadata = None
        self._initial_frame = None
        self._rendered_observation = self._consumed_observation = None
        for attempt in range(self.retries + 1):
            observation = self.simulator.reset(init_state_id, seed=seed)
            actual = self.simulator.episode_metadata
            mismatch = reference_mismatches(self.reference, init_state_id, actual)
            replay_audit = None
            if mismatch == ["initial_observation_sha256"] and self.snapshots:
                from robotics_bench.simulators.reference_observation import (
                    restore_reference_observation,
                )

                expected = json.loads(
                    (
                        Path(self.reference)
                        / "initializations"
                        / f"{init_state_id:06d}"
                        / "episode.json"
                    ).read_text()
                )
                try:
                    replay = restore_reference_observation(
                        self.snapshots, init_state_id, expected, actual, observation
                    )
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    with (self.output / "initialization-attempts.jsonl").open(
                        "a"
                    ) as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "init_state_id": init_state_id,
                                    "env_seed": seed,
                                    "attempt": attempt + 1,
                                    "accepted": False,
                                    "mismatches": mismatch,
                                    "initial_state_sha256": actual[
                                        "initial_state_sha256"
                                    ],
                                    "initial_xml_sha256": actual["initial_xml_sha256"],
                                    "initial_observation_sha256": actual[
                                        "initial_observation_sha256"
                                    ],
                                    "observation_replay_error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            + "\n"
                        )
                    raise
                if replay is not None:
                    import numpy as np

                    self._rendered_observation = observation
                    observation, replay_audit = replay
                    self._consumed_observation = observation
                    self._metadata = {
                        **actual,
                        "initial_observation_sha256": replay_audit[
                            "consumed_observation_sha256"
                        ],
                        "initial_rendered_observation_sha256": replay_audit[
                            "rendered_observation_sha256"
                        ],
                        "initial_observation_replay": replay_audit,
                    }
                    actual = self._metadata
                    self._initial_frame = np.concatenate(
                        [
                            observation[key][::-1]
                            for key in (
                                "primary_image",
                                "secondary_image",
                                "wrist_image",
                            )
                        ],
                        axis=1,
                    )
                    mismatch = reference_mismatches(
                        self.reference, init_state_id, actual
                    )
            record = {
                "init_state_id": init_state_id,
                "env_seed": seed,
                "attempt": attempt + 1,
                "accepted": not mismatch,
                "mismatches": mismatch,
                "observation_replay": replay_audit,
                **{
                    key: actual[key]
                    for key in (
                        "initial_state_sha256",
                        "initial_xml_sha256",
                        "initial_observation_sha256",
                    )
                },
            }
            with (self.output / "initialization-attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if not mismatch:
                return observation
            if mismatch != ["initial_observation_sha256"] or attempt == self.retries:
                raise RuntimeError(
                    f"Episode initialization differs from reference: {mismatch}"
                )
            print(
                f"Retry initialization {init_state_id}: observation differs, attempt {attempt + 1}/{self.retries + 1}",
                flush=True,
            )
        raise AssertionError("Unreachable reset state")


def execute(plan):
    sys.path.insert(0, str(ROOT / "src"))
    from robotics_bench.engines.cosmos import CosmosEngine
    from robotics_bench.simulators.robocasa import NativeRoboCasaSimulator
    from robotics_bench.protocols.static_runner import run_episode

    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    opts, paths = plan["options"], plan["resources"]
    manifest = {
        **plan,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    planned = [
        {
            "task": opts["task"],
            "init_state_id": i,
            "env_seed": opts["env_seed"] + i,
            "layout_id": opts["layout_id"],
            "style_id": opts["style_id"],
        }
        for i in range(opts["episodes"])
    ]
    manifest["planned_episodes"] = planned
    manifest["environment_protocol"] = "robocasa_seeded_scene_v1"
    shared.write_json(output / "case-manifest.json", manifest)
    shared.write_json(output / "paper-async.json", plan["paper_async"])
    summaries = load_file(ROOT / "tools/summarize_experiment.py", "_robocasa_summary")
    video_tools = load_file(
        ROOT / "benchmarks/static/pi05_libero/video_recorder.py", "_robocasa_video"
    )
    environment = {
        "CUDA_VISIBLE_DEVICES": plan["gpu"],
        "MUJOCO_GL": "egl",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "DETERMINISTIC": "true",
    }
    old_env = {
        key: os.environ.get(key) for key in [*environment, "MUJOCO_EGL_DEVICE_ID"]
    }
    engine = simulator = None
    rows = []
    with shared.console_log(output / "run.log"):
        try:
            environment["MUJOCO_EGL_DEVICE_ID"] = shared.egl_index(plan["gpu"])
            os.environ.update(environment)
            shared.write_json(
                output / "coverage.json",
                shared.coverage_report(planned, rows, opts["max_steps"], "running"),
            )
            engine = CosmosEngine(
                paths["cosmos_source"],
                paths["checkpoint"],
                paths["dataset_stats"],
                paths["text_embeddings"],
                paths["vae_checkpoint"],
                suite="robocasa",
                action_horizon=32,
                num_inference_steps=opts["num_inference_steps"],
            )
            engine.load(None, output / "checkpoint-load.json")
            from robotics_bench.optimizations.entry import from_dict

            engine.configure_optimizations(
                from_dict(opts["optimizations"]), runtime=opts["model_runtime"]
            )
            manifest["engine"] = engine.metadata
            manifest["runtime_python"] = sys.executable
            manifest["runtime_packages"] = {}
            for package in (
                "torch",
                "numpy",
                "mujoco",
                "robosuite",
                "robocasa",
                "transformer-engine",
                "natten",
            ):
                try:
                    manifest["runtime_packages"][package] = importlib.metadata.version(
                        package
                    )
                except importlib.metadata.PackageNotFoundError:
                    manifest["runtime_packages"][package] = None
            shared.write_json(output / "case-manifest.json", manifest)
            if (
                json.loads((output / "checkpoint-load.json").read_text())["status"]
                != "passed"
            ):
                raise RuntimeError("Checkpoint load audit did not pass")
            simulator = NativeRoboCasaSimulator(
                paths["robocasa_source"],
                paths["controller_config"],
                opts["task"],
                layout_id=opts["layout_id"],
                style_id=opts["style_id"],
            )
            if opts.get("initialization_retries", 0) or plan.get(
                "reference_observations"
            ):
                simulator = ReferenceReset(
                    simulator,
                    plan["reference_run"],
                    output,
                    opts["initialization_retries"],
                    snapshots=plan.get("reference_observations"),
                )
            video = None
            if opts["record_video"]:
                import imageio.v2 as imageio

                video = video_tools.EpisodeVideoRecorder(
                    output,
                    opts["episodes"],
                    opts["video_fps"],
                    lambda path, frames, fps: imageio.mimsave(path, frames, fps=fps),
                )
            started = time.perf_counter()
            with (
                (output / "episodes.jsonl").open("x") as ledger,
                (output / "requests.jsonl").open("x") as requests,
            ):
                for index, item in enumerate(planned):
                    video_callback = None

                    def on_frame(frame):
                        nonlocal video_callback
                        if video_callback is None:
                            video_callback = video.start(
                                [(item["task"], simulator.description, [index])], 0
                            )

                        class FrameBatch:
                            def render(self):
                                return [frame]

                        video_callback(FrameBatch())

                    def on_request(request):
                        if request["inference_index"] == 1:
                            simulator.save_initialization(
                                output / "initializations" / f"{index:06d}"
                            )
                            if plan["reference_run"]:
                                check_reference(
                                    plan["reference_run"],
                                    index,
                                    simulator.episode_metadata,
                                )
                        requests.write(json.dumps({**item, **request}) + "\n")
                        requests.flush()

                    row = run_episode(
                        engine,
                        simulator,
                        task=item["task"],
                        init_state_id=index,
                        env_seed=item["env_seed"],
                        sampling_seed=opts["seed"],
                        max_steps=opts["max_steps"],
                        n_action_steps=opts["n_action_steps"],
                        schedule=opts["schedule"],
                        overlap_actions=opts["overlap_actions"],
                        on_frame=on_frame if video else None,
                        on_request=on_request,
                    )
                    row.update(
                        layout_id=opts["layout_id"],
                        style_id=opts["style_id"],
                        description=simulator.description,
                        control_dt_seconds=simulator.control_dt,
                        initial_state_sha256=simulator.episode_metadata[
                            "initial_state_sha256"
                        ],
                        initial_observation_sha256=simulator.episode_metadata[
                            "initial_observation_sha256"
                        ],
                        initial_rendered_observation_sha256=simulator.episode_metadata.get(
                            "initial_rendered_observation_sha256",
                            simulator.episode_metadata["initial_observation_sha256"],
                        ),
                        initial_observation_source="reference_snapshot"
                        if simulator.episode_metadata.get("initial_observation_replay")
                        else "native",
                        initial_observation_replay=simulator.episode_metadata.get(
                            "initial_observation_replay"
                        ),
                        reference_initialization_passed=True
                        if plan["reference_run"]
                        else None,
                    )
                    ledger.write(json.dumps(row, allow_nan=False) + "\n")
                    ledger.flush()
                    rows.append(row)
                    if video:
                        video.finish([row])
                    shared.write_json(
                        output / "coverage.json",
                        shared.coverage_report(
                            planned, rows, opts["max_steps"], "running"
                        ),
                    )
                    print(
                        f"Episode {index + 1}/{len(planned)}: {item['task']} success={row['success']} steps={row['primitive_steps']}",
                        flush=True,
                    )
            result = shared.evaluation_results(rows)
            result["eval_s"] = time.perf_counter() - started
            if video:
                result["video_paths"], result["video_metadata"] = (
                    video.paths,
                    video.records,
                )
            shared.write_json(output / "eval_results.json", result)
            shared.write_json(
                output / "coverage.json",
                shared.coverage_report(planned, rows, opts["max_steps"], "passed"),
            )
            report = summaries.write_run_summary(
                output, rows, max_steps=opts["max_steps"]
            )
            print(summaries.markdown_report(report))
            manifest["summary_files"] = {
                "markdown": str(output / "episode-summary.md"),
                "json": str(output / "episode-summary.json"),
            }
            manifest["status"] = "completed"
        except BaseException as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            shared.write_json(
                output / "failure.json",
                {"error": manifest["error"], "traceback": traceback.format_exc()},
            )
            shared.write_json(
                output / "coverage.json",
                shared.coverage_report(
                    planned, rows, opts["max_steps"], "failed", manifest["error"]
                ),
            )
            raise
        finally:
            for resource in (simulator, engine):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        manifest.setdefault("cleanup_errors", []).append(str(exc))
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            shared.write_json(output / "case-manifest.json", manifest)


def main(argv=None):
    config = load_file(Path(__file__).with_name("config.py"), "_cosmos_robocasa_config")
    parser = config.build_parser()
    args = parser.parse_args(argv)
    try:
        plan = config.build_plan(args)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    execute(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
