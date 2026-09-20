"""Build a Cosmos LIBERO run plan without importing model or simulator code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess


SUITE_STEP_LIMITS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
RESOURCE_ENV = {
    "cosmos_source": "ROBOTICS_COSMOS_SOURCE",
    "checkpoint": "ROBOTICS_COSMOS_CHECKPOINT",
    "dataset_stats": "ROBOTICS_COSMOS_DATASET_STATS",
    "text_embeddings": "ROBOTICS_COSMOS_TEXT_EMBEDDINGS",
    "vae_checkpoint": "ROBOTICS_COSMOS_VAE_CHECKPOINT",
    "libero_config_dir": "ROBOTICS_COSMOS_LIBERO_CONFIG_DIR",
}
DEFAULT_MODEL_CONFIG = "cosmos_predict2_2b_480p_libero__inference_only"
REPOSITORY = Path(__file__).resolve().parents[3]


def build_parser():
    """Return the public CLI; environment paths are defaults, never overrides."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name, environment in RESOURCE_ENV.items():
        value = os.environ.get(environment, "").strip()
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=Path,
            default=Path(value) if value else None,
            required=not value,
            help=f"Explicit resource path; defaults to ${environment}",
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", help="One physical GPU index or full GPU UUID")
    parser.add_argument("--suite", choices=SUITE_STEP_LIMITS, default="libero_object")
    parser.add_argument("--task-ids", default="all", help="Comma-separated IDs or all")
    for name, default in {
        "episodes": 1,
        "seed": 195,
        "env-seed": 0,
        "n-action-steps": 16,
        "num-inference-steps": 5,
        "overlap-actions": 0,
    }.items():
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--schedule", choices=("sync", "paper_async"))
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--quant", choices=("none",), default="none")
    parser.add_argument("--max-steps", type=int)
    video = parser.add_mutually_exclusive_group()
    video.add_argument(
        "--record-video",
        action="store_const",
        const="all",
        dest="video_episodes_per_task",
    )
    video.add_argument(
        "--no-record-video",
        action="store_const",
        const=0,
        dest="video_episodes_per_task",
    )
    video.add_argument("--video-episodes-per-task", type=int)
    parser.set_defaults(video_episodes_per_task=0)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--paper-action-time-ms", type=float, default=50.0)
    parser.add_argument("--paper-inference-time-ms", type=float)
    parser.add_argument("--paper-inference-time-source")
    parser.add_argument("--run-kind", choices=("smoke", "evaluation"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _task_ids(value):
    if value == "all":
        return None
    if not re.fullmatch(r"[0-9]+(?:\s*,\s*[0-9]+)*", value):
        raise ValueError("task_ids must be comma-separated IDs or all")
    ids = [int(item) for item in value.split(",")]
    if len(set(ids)) != len(ids) or any(task >= 10 for task in ids):
        raise ValueError("task_ids must be unique IDs in 0..9")
    return ids


def _options(args):
    names = (
        "suite",
        "episodes",
        "seed",
        "env_seed",
        "n_action_steps",
        "num_inference_steps",
        "overlap_actions",
        "model_config",
        "quant",
        "video_episodes_per_task",
        "video_fps",
        "paper_action_time_ms",
        "paper_inference_time_ms",
        "paper_inference_time_source",
    )
    options = {name: getattr(args, name) for name in names}
    options.update(
        task_ids=_task_ids(args.task_ids),
        schedule=args.schedule
        or ("paper_async" if args.overlap_actions > 0 else "sync"),
        max_steps=args.max_steps
        if args.max_steps is not None
        else SUITE_STEP_LIMITS[args.suite],
        settle_steps=10,
        action_horizon=16,
        delay_state_with_observation=True,
    )
    for name in ("episodes", "n_action_steps", "num_inference_steps", "max_steps"):
        _integer(options[name], name, 1)
    for name in ("seed", "env_seed", "overlap_actions"):
        _integer(options[name], name)
    if options["n_action_steps"] > options["action_horizon"]:
        raise ValueError("n_action_steps must be <= 16")
    if options["video_episodes_per_task"] != "all":
        _integer(options["video_episodes_per_task"], "video_episodes_per_task")
    if not math.isfinite(options["video_fps"]) or options["video_fps"] <= 0:
        raise ValueError("video_fps must be finite and positive")
    if not options["model_config"].strip():
        raise ValueError("model_config must be nonempty")
    return options


def _paper_contract(options):
    source = REPOSITORY / "benchmarks/static/pi05_libero/paper_async.py"
    spec = importlib.util.spec_from_file_location("_cosmos_paper_contract", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.paper_async_contract(options)


def _required_file(path):
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    return path


def _check_source(root):
    source = _required_file(root / "cosmos_policy/experiments/robot/cosmos_utils.py")
    _required_file(root / "cosmos_policy/_src/predict2/utils/model_loader.py")
    _required_file(root / "cosmos_policy/config/config.py")
    try:
        tree = ast.parse(source.read_text())
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse Cosmos source: {source}") from exc
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_action"
        ),
        None,
    )
    needed = {
        "cfg",
        "model",
        "dataset_stats",
        "obs",
        "task_label_or_embedding",
        "seed",
        "randomize_seed",
        "num_denoising_steps_action",
        "generate_future_state_and_value_in_parallel",
        "worker_id",
        "batch_size",
    }
    parameters = (
        set()
        if function is None
        else {arg.arg for arg in function.args.args + function.args.kwonlyargs}
    )
    if not needed <= parameters:
        raise ValueError(
            f"Incompatible get_action signature; missing parameters: {sorted(needed - parameters)}"
        )
    positional = [arg.arg for arg in function.args.args]
    if positional[:5] != [
        "cfg",
        "model",
        "dataset_stats",
        "obs",
        "task_label_or_embedding",
    ]:
        raise ValueError("Incompatible get_action positional parameter order")
    required = set(positional[: len(positional) - len(function.args.defaults)])
    required.update(
        arg.arg
        for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
        if default is None
    )
    unsupported = required - (needed - {"worker_id"})
    if unsupported:
        raise ValueError(
            f"Incompatible get_action required parameters: {sorted(unsupported)}"
        )


def _check_statistics(path):
    try:
        statistics = json.loads(path.read_text())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset statistics JSON: {path}") from exc
    if not isinstance(statistics, dict):
        raise ValueError("Dataset statistics must be a JSON object")
    for prefix, size in (("actions", 7), ("proprio", 9)):
        for suffix in ("min", "max"):
            name = f"{prefix}_{suffix}"
            values = statistics.get(name)
            if (
                not isinstance(values, list)
                or len(values) != size
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in values
                )
            ):
                raise ValueError(f"{name} must contain {size} finite numbers")


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _source_identity(root):
    files = {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in sorted((root / "cosmos_policy").rglob("*.py"))
        if path.is_file()
    }
    identity = {
        "python_files_sha256": files,
        "python_tree_sha256": _json_hash(files),
        "git_available": (root / ".git").exists(),
    }
    if identity["git_available"]:
        try:

            def git(*arguments):
                return subprocess.check_output(
                    ["git", "-C", str(root), *arguments],
                    stderr=subprocess.PIPE,
                    timeout=10,
                )

            identity["git_head"] = git("rev-parse", "HEAD").decode().strip()
            identity["git_dirty"] = bool(
                git("status", "--porcelain", "--untracked-files=all")
            )
            identity["git_diff_sha256"] = hashlib.sha256(
                git("diff", "HEAD")
            ).hexdigest()
        except (OSError, subprocess.SubprocessError) as exc:
            identity["git_identity_error"] = type(exc).__name__
    return identity


def _entry_identity():
    files = set(Path(__file__).parent.glob("*.py"))
    files.update((REPOSITORY / "src/robotics_bench").rglob("*.py"))
    files.update(
        REPOSITORY / name
        for name in (
            "benchmarks/static/pi05_libero/paper_async.py",
            "benchmarks/static/pi05_libero/video_recorder.py",
            "tools/compare_speedups.py",
            "tools/episode_statistics.py",
            "tools/summarize_experiment.py",
        )
    )
    return {
        path.relative_to(REPOSITORY).as_posix(): _file_hash(path)
        for path in sorted(files)
        if path.is_file()
    }


def build_plan(args):
    """Validate local artifacts and return a JSON-serializable, content-bound plan."""
    if os.environ.get("COSMOS_SMOKE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        raise ValueError(
            "COSMOS_SMOKE must be disabled; it can skip checkpoint loading"
        )
    options = _options(args)
    paper = _paper_contract(options)
    output = args.output_dir.expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError(
            f"Output directory already exists; choose a new directory: {output}"
        )
    output = output.resolve()
    gpu_pattern = r"[0-9]+|GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    if args.gpu is not None and not re.fullmatch(gpu_pattern, args.gpu):
        raise ValueError("--gpu must select one GPU index or full GPU UUID")
    if not args.dry_run and args.gpu is None:
        raise ValueError("Actual execution requires an explicit --gpu")
    resources = {
        name: getattr(args, name).expanduser().resolve() for name in RESOURCE_ENV
    }
    _check_source(resources["cosmos_source"])
    files = {
        name: _required_file(path)
        for name, path in resources.items()
        if name not in {"cosmos_source", "libero_config_dir"}
    }
    if resources["checkpoint"].suffix != ".pt":
        raise ValueError("checkpoint must be a .pt file")
    files["libero_config"] = _required_file(
        resources["libero_config_dir"] / "config.yaml"
    )
    _check_statistics(resources["dataset_stats"])
    identity = {
        "case": options,
        "resources_sha256": {name: _file_hash(path) for name, path in files.items()},
        "source": _source_identity(resources["cosmos_source"]),
        "entry_code_sha256": _entry_identity(),
    }
    return {
        "format": "cosmos-libero-run-v1",
        "case_id": "cosmos_libero",
        "run_kind": args.run_kind
        or ("smoke" if options["episodes"] == 1 else "evaluation"),
        "output_dir": str(output),
        "gpu": args.gpu,
        "resources": {name: str(path) for name, path in resources.items()},
        "options": options,
        "paper_async": paper,
        "identity": identity,
        "case_fingerprint": _json_hash(identity),
    }
