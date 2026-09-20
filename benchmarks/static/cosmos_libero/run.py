"""Run one native Cosmos Policy / LIBERO environment at a time."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        return self.original.write(text)

    def flush(self):
        self.log.flush()
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def console_log(path):
    with path.open("x", encoding="utf-8", buffering=1) as log:
        with (
            redirect_stdout(_Tee(sys.stdout, log)),
            redirect_stderr(_Tee(sys.stderr, log)),
        ):
            yield


def egl_index(selector):
    if selector.isdigit():
        return selector
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    for line in listing.splitlines():
        index, uuid = (part.strip() for part in line.split(",", 1))
        if uuid == selector:
            return index
    raise ValueError("GPU UUID is not present in the NVIDIA device list")


def allocate_episodes(tasks, count, selected_ids=None):
    """Distribute the total budget across selected tasks without repeating states."""
    available = {task["id"]: task for task in tasks}
    ids = list(available) if selected_ids is None else list(selected_ids)
    if not ids or len(set(ids)) != len(ids) or any(key not in available for key in ids):
        raise ValueError("Task selection contains missing or duplicate task IDs")
    if type(count) is not int or count < 1:
        raise ValueError("episodes must be a positive integer")
    capacities = {key: available[key]["initial_states"] for key in ids}
    if any(type(value) is not int or value < 1 for value in capacities.values()):
        raise ValueError("Every selected task must have initial states")
    if count > sum(capacities.values()):
        raise ValueError("Requested episodes exceed available selected initial states")
    allocation = dict.fromkeys(ids, 0)
    remaining = count
    while remaining:
        for key in ids:
            if remaining and allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
    return [
        {
            "task_id": key,
            "task": available[key]["name"],
            "description": available[key]["description"],
            "init_state_id": initial,
        }
        for key in ids
        for initial in range(allocation[key])
    ]


def coverage_report(planned, rows, max_steps, status, error=None):
    expected = {}
    for item in planned:
        expected.setdefault(item["task"], []).append(item["init_state_id"])
    tasks = {}
    for task, ids in expected.items():
        finished = [row for row in rows if row["task"] == task]
        tasks[task] = {
            "episodes": len(finished),
            "successes": sum(row["success"] for row in finished),
            "init_state_ids": sorted(row["init_state_id"] for row in finished),
            "expected_init_state_ids": ids,
        }
    value = {
        "status": status,
        "expected_episodes": len(planned),
        "completed_episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "max_primitive_steps": max_steps,
        "tasks": tasks,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        value["error"] = error
    return value


def evaluation_results(rows):
    results = {}
    for task in ["overall", *sorted({row["task"] for row in rows})]:
        selected = (
            rows if task == "overall" else [row for row in rows if row["task"] == task]
        )
        results[task] = {
            "episodes": len(selected),
            "successes": sum(row["success"] for row in selected),
            "pc_successes": 100
            * sum(row["success"] for row in selected)
            / len(selected),
            "avg_episode_length": sum(row["primitive_steps"] for row in selected)
            / len(selected),
        }
    return results


def execute(plan, *, engine_type=None, suite_type=None):
    """Execute the case; injectable adapters allow CPU lifecycle verification."""
    sys.path.insert(0, str(ROOT / "src"))
    from robotics_bench.protocols.static_runner import run_episode

    if engine_type is None:
        from robotics_bench.engines.cosmos import CosmosEngine

        engine_type = CosmosEngine
    if suite_type is None:
        from robotics_bench.simulators.libero import LiberoSuite

        suite_type = LiberoSuite
    summaries = load_file(ROOT / "tools/summarize_experiment.py", "_cosmos_summary")
    videos = load_file(
        ROOT / "benchmarks/static/pi05_libero/video_recorder.py", "_cosmos_video"
    )
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    options, assets = plan["options"], plan["resources"]
    manifest = {
        **plan,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "case-manifest.json", manifest)
    write_json(output / "paper-async.json", plan["paper_async"])
    environment = {
        "CUDA_VISIBLE_DEVICES": plan["gpu"],
        "MUJOCO_GL": "egl",
        "LIBERO_CONFIG_PATH": assets["libero_config_dir"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "DETERMINISTIC": "true",
    }
    saved_environment = {
        key: os.environ.get(key) for key in [*environment, "MUJOCO_EGL_DEVICE_ID"]
    }
    engine, simulator, active_task = None, None, None
    planned, rows = [], []
    with console_log(output / "run.log"):
        try:
            environment["MUJOCO_EGL_DEVICE_ID"] = egl_index(plan["gpu"])
            os.environ.update(environment)
            suite = suite_type(options["suite"])
            tasks = suite.tasks()
            planned = allocate_episodes(tasks, options["episodes"], options["task_ids"])
            manifest["planned_episodes"] = planned
            write_json(output / "case-manifest.json", manifest)
            write_json(
                output / "coverage.json",
                coverage_report(planned, rows, options["max_steps"], "running"),
            )
            engine = engine_type(
                assets["cosmos_source"],
                assets["checkpoint"],
                assets["dataset_stats"],
                assets["text_embeddings"],
                assets["vae_checkpoint"],
                config_name=options["model_config"],
                num_inference_steps=options["num_inference_steps"],
                action_horizon=16,
            )
            engine.load(
                sorted({item["description"] for item in planned}),
                output / "checkpoint-load.json",
            )
            audit = json.loads((output / "checkpoint-load.json").read_text())
            if audit.get("status") != "passed":
                raise RuntimeError("Checkpoint audit did not pass")
            manifest["engine"] = engine.metadata
            manifest["runtime_packages"] = {}
            for package in (
                "torch",
                "numpy",
                "mujoco",
                "robosuite",
                "libero",
                "transformer-engine",
                "natten",
            ):
                try:
                    manifest["runtime_packages"][package] = importlib.metadata.version(
                        package
                    )
                except importlib.metadata.PackageNotFoundError:
                    manifest["runtime_packages"][package] = None
            write_json(output / "case-manifest.json", manifest)
            limit = options["video_episodes_per_task"]
            video = None
            if limit != 0:
                import imageio.v2 as imageio

                video = videos.EpisodeVideoRecorder(
                    output,
                    len(planned) if limit == "all" else limit,
                    options["video_fps"],
                    imageio.mimsave,
                )
            started = time.perf_counter()
            with (
                (output / "episodes.jsonl").open("x") as episode_file,
                (output / "requests.jsonl").open("x") as requests,
            ):
                for index, item in enumerate(planned):
                    if active_task != item["task_id"]:
                        if simulator is not None:
                            simulator.close()
                        simulator = suite.make_simulator(
                            item["task_id"],
                            env_seed=options["env_seed"],
                            settle_steps=10,
                        )
                        active_task = item["task_id"]
                    if simulator.task_name != item["task"]:
                        raise RuntimeError(
                            "Simulator task identity differs from the planned episode"
                        )
                    video_callback = (
                        video.start(
                            [
                                (
                                    item["task"],
                                    item["description"],
                                    [item["init_state_id"]],
                                )
                            ],
                            0,
                        )
                        if video
                        else None
                    )

                    def on_frame(frame):
                        class FrameBatch:
                            def render(self):
                                return [frame]

                        video_callback(FrameBatch())

                    def on_request(request):
                        requests.write(
                            json.dumps(
                                {
                                    "task": item["task"],
                                    "init_state_id": item["init_state_id"],
                                    "env_seed": options["env_seed"],
                                    **request,
                                }
                            )
                            + "\n"
                        )
                        requests.flush()

                    row = run_episode(
                        engine,
                        simulator,
                        task=item["task"],
                        init_state_id=item["init_state_id"],
                        env_seed=options["env_seed"],
                        sampling_seed=options["seed"],
                        max_steps=options["max_steps"],
                        n_action_steps=options["n_action_steps"],
                        schedule=options["schedule"],
                        overlap_actions=options["overlap_actions"],
                        on_frame=on_frame if video_callback else None,
                        on_request=on_request,
                    )
                    if (row["task"], row["init_state_id"], row["env_seed"]) != (
                        item["task"],
                        item["init_state_id"],
                        options["env_seed"],
                    ):
                        raise RuntimeError(
                            "Returned episode identity differs from the plan"
                        )
                    row["task_id"] = item["task_id"]
                    row["control_dt_seconds"] = simulator.control_dt
                    episode_file.write(json.dumps(row, allow_nan=False) + "\n")
                    episode_file.flush()
                    rows.append(row)
                    if video:
                        video.finish([row])
                    write_json(
                        output / "coverage.json",
                        coverage_report(planned, rows, options["max_steps"], "running"),
                    )
                    print(
                        f"Episode {index + 1}/{len(planned)}: {row['task']} init={row['init_state_id']} "
                        f"success={row['success']} steps={row['primitive_steps']}",
                        flush=True,
                    )
            if len(rows) != len(planned):
                raise RuntimeError("Incomplete episode coverage")
            result = evaluation_results(rows)
            result["eval_s"] = time.perf_counter() - started
            if video:
                result["video_paths"], result["video_metadata"] = (
                    video.paths,
                    video.records,
                )
            write_json(output / "eval_results.json", result)
            write_json(
                output / "coverage.json",
                coverage_report(planned, rows, options["max_steps"], "passed"),
            )
            report = summaries.write_run_summary(
                output, rows, max_steps=options["max_steps"]
            )
            print(summaries.markdown_report(report))
            manifest["summary_files"] = {
                "markdown": str(output / "episode-summary.md"),
                "json": str(output / "episode-summary.json"),
            }
            manifest["status"] = "completed"
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            write_json(
                output / "failure.json",
                {"error": manifest["error"], "traceback": traceback.format_exc()},
            )
            write_json(
                output / "coverage.json",
                coverage_report(
                    planned, rows, options["max_steps"], "failed", manifest["error"]
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
            for key, value in saved_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(output / "case-manifest.json", manifest)


def main(argv=None):
    config = load_file(Path(__file__).with_name("config.py"), "_cosmos_case_config")
    parser = config.build_parser()
    args = parser.parse_args(argv)
    try:
        plan = config.build_plan(args)
    except (ValueError, OSError, TypeError, SyntaxError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    execute(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
