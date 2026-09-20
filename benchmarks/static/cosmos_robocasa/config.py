"""Build an explicit, CPU-only Cosmos RoboCasa single-environment run plan."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from robotics_bench.simulators.robocasa import TASK_MAX_STEPS  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "_cosmos_shared_config", Path(__file__).parent.parent / "cosmos_libero/config.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

RESOURCE_ENV = {
    "cosmos_source": "ROBOTICS_COSMOS_SOURCE",
    "robocasa_source": "ROBOTICS_ROBOCASA_SOURCE",
    "checkpoint": "ROBOTICS_COSMOS_ROBOCASA_CHECKPOINT",
    "dataset_stats": "ROBOTICS_COSMOS_ROBOCASA_DATASET_STATS",
    "text_embeddings": "ROBOTICS_COSMOS_ROBOCASA_TEXT_EMBEDDINGS",
    "vae_checkpoint": "ROBOTICS_COSMOS_ROBOCASA_VAE_CHECKPOINT",
    "controller_config": "ROBOTICS_ROBOCASA_CONTROLLER_CONFIG",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name, variable in RESOURCE_ENV.items():
        value = os.environ.get(variable)
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=Path,
            default=Path(value) if value else None,
            required=not value,
            help=f"Explicit path; defaults to ${variable}",
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=TASK_MAX_STEPS, default="TurnOffMicrowave")
    for name, default in {
        "episodes": 1,
        "layout-id": 1,
        "style-id": 1,
        "env-seed": 0,
        "seed": 195,
        "n-action-steps": 16,
        "num-inference-steps": 5,
        "overlap-actions": 0,
    }.items():
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--schedule", choices=("sync", "paper_async"))
    parser.add_argument("--quant", choices=("none",), default="none")
    parser.add_argument("--gpu")
    parser.add_argument(
        "--reference-run",
        type=Path,
        help="Require initialization to match the corresponding completed run",
    )
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--record-video", dest="record_video", action="store_true")
    video.add_argument("--no-record-video", dest="record_video", action="store_false")
    parser.set_defaults(record_video=False)
    parser.add_argument("--video-fps", type=float, default=30)
    parser.add_argument("--paper-action-time-ms", type=float, default=50)
    parser.add_argument("--paper-inference-time-ms", type=float)
    parser.add_argument("--paper-inference-time-source")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_plan(args):
    if os.environ.get("COSMOS_SMOKE", "").lower() in {"1", "true", "yes", "y", "on"}:
        raise ValueError("COSMOS_SMOKE must be disabled")
    output = args.output_dir.expanduser()
    if output.exists() or output.is_symlink():
        raise ValueError("Output directory already exists; choose a new directory")
    for name in ("episodes", "n_action_steps", "num_inference_steps"):
        common._integer(getattr(args, name), name, 1)
    for name in ("layout_id", "style_id", "env_seed", "seed", "overlap_actions"):
        common._integer(getattr(args, name), name)
    if args.n_action_steps > 32:
        raise ValueError("n_action_steps must be <= 32")
    if not math.isfinite(args.video_fps) or args.video_fps <= 0:
        raise ValueError("video_fps must be finite and positive")
    max_steps = (
        args.max_steps if args.max_steps is not None else TASK_MAX_STEPS[args.task]
    )
    common._integer(max_steps, "max_steps", 1)
    if args.gpu is not None and not re.fullmatch(
        r"[0-9]+|GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu
    ):
        raise ValueError("--gpu must select one GPU index or full GPU UUID")
    if not args.dry_run and args.gpu is None:
        raise ValueError("Actual execution requires an explicit --gpu")
    options = {
        name: getattr(args, name)
        for name in (
            "task",
            "episodes",
            "layout_id",
            "style_id",
            "env_seed",
            "seed",
            "n_action_steps",
            "num_inference_steps",
            "overlap_actions",
            "quant",
            "record_video",
            "video_fps",
            "paper_action_time_ms",
            "paper_inference_time_ms",
            "paper_inference_time_source",
        )
    }
    options.update(
        max_steps=max_steps,
        action_horizon=32,
        settle_steps=10,
        delay_state_with_observation=True,
        schedule=args.schedule or ("paper_async" if args.overlap_actions else "sync"),
    )
    paper = common._paper_contract(options)
    resources = {key: getattr(args, key).expanduser().resolve() for key in RESOURCE_ENV}
    common._check_source(resources["cosmos_source"])
    files = {
        key: common._required_file(value)
        for key, value in resources.items()
        if key not in {"cosmos_source", "robocasa_source"}
    }
    if resources["checkpoint"].suffix != ".pt":
        raise ValueError("checkpoint must be a .pt file")
    common._check_statistics(resources["dataset_stats"])
    robocasa = resources["robocasa_source"] / "robocasa"
    common._required_file(
        robocasa / "environments/kitchen/single_stage/kitchen_microwave.py"
    )
    asset_root = robocasa / "models/assets"
    for directory in ("fixtures", "textures", "objects/objaverse", "scenes"):
        if not (asset_root / directory).is_dir():
            raise ValueError(
                f"Missing RoboCasa assets: {asset_root / directory}; run the compatible fork's download_kitchen_assets.py"
            )
    reference = (
        args.reference_run.expanduser().resolve() if args.reference_run else None
    )
    if reference:
        manifest = json.loads(
            common._required_file(reference / "case-manifest.json").read_text()
        )
        if (
            manifest.get("case_id") != "cosmos_robocasa"
            or manifest.get("status") != "completed"
        ):
            raise ValueError("reference-run must be a completed Cosmos RoboCasa run")
        for key in ("task", "episodes", "layout_id", "style_id", "env_seed"):
            if manifest["options"][key] != options[key]:
                raise ValueError(f"reference-run differs in {key}")
    robocasa_identity = common._source_identity(resources["robocasa_source"])
    robocasa_identity["python_files_sha256"] = {
        p.relative_to(robocasa).as_posix(): common._file_hash(p)
        for p in sorted(robocasa.rglob("*.py"))
    }
    robocasa_identity["python_tree_sha256"] = common._json_hash(
        robocasa_identity["python_files_sha256"]
    )
    identity = {
        "case": options,
        "resources_sha256": {key: common._file_hash(p) for key, p in files.items()},
        "cosmos_source": common._source_identity(resources["cosmos_source"]),
        "robocasa_source": robocasa_identity,
        "entry_code_sha256": {
            **common._entry_identity(),
            **{
                p.relative_to(ROOT).as_posix(): common._file_hash(p)
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        },
    }
    return {
        "format": "cosmos-robocasa-run-v1",
        "case_id": "cosmos_robocasa",
        "run_kind": "smoke" if args.episodes == 1 else "evaluation",
        "options": options,
        "resources": {k: str(v) for k, v in resources.items()},
        "output_dir": str(output.resolve()),
        "gpu": args.gpu,
        "reference_run": str(reference) if reference else None,
        "paper_async": paper,
        "identity": identity,
        "case_fingerprint": common._json_hash(identity),
    }
