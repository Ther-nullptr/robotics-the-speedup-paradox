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


def check_reference(reference, index, actual):
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
    mismatch = [key for key in keys if expected.get(key) != actual.get(key)]
    if mismatch:
        raise RuntimeError(f"Episode initialization differs from reference: {mismatch}")


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
