"""Run a bounded LeRobot PI0.5 trial through a compatible external LIBERO evaluator.

Only the standard library is imported until execution. Source/checkpoint paths
are explicit; the external evaluator and model dependencies are not vendored.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback


DEFAULT_CONFIG = Path(__file__).with_name("case.json")
SUITES = {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}
LADDERS = {"none", "fp16_to_w8a8", "fp16_to_w4a4", "w8a8_to_w4a4"}


def read_object(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_file(path):
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    return path


def source_identity(root):
    files = {}
    for directory in ("vlash", "benchmarks", "mini_qserve_gemm", "eval"):
        for path in sorted((root / directory).rglob("*.py")):
            files[path.relative_to(root).as_posix()] = file_hash(path)
    identity = {"python_files_sha256": files}
    if (root / ".git").exists():
        try:
            identity["git_head"] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=10
            ).strip()
            diff = subprocess.check_output(
                ["git", "-C", str(root), "diff", "HEAD"], timeout=10
            )
            identity["git_dirty"] = bool(diff)
            identity["git_diff_sha256"] = hashlib.sha256(diff).hexdigest()
        except (OSError, subprocess.SubprocessError) as exc:
            identity["git_identity_error"] = type(exc).__name__
    return identity


def load_options(path, cli_options=None):
    defaults = read_object(DEFAULT_CONFIG)
    overrides = read_object(path) if path is not None else {}
    unknown = set(overrides) - (set(defaults) | {"async_delay"})
    if unknown:
        raise ValueError(f"Unknown case options: {sorted(unknown)}")
    # Canonical n-prime; retain the old upstream spelling only as an input alias.
    if "async_delay" in overrides:
        delay = overrides["async_delay"]
        if type(delay) is not int or delay < 0:
            raise ValueError("async_delay must be a nonnegative integer")
        if "overlap_actions" in overrides and (
            type(overrides["overlap_actions"]) is not int
            or overrides["overlap_actions"] != delay
        ):
            raise ValueError("async_delay and overlap_actions disagree")
        overrides["overlap_actions"] = delay
        del overrides["async_delay"]
    overrides.update(cli_options or {})
    options = {**defaults, **overrides}
    if "schedule" not in overrides and options["overlap_actions"] != 0:
        options["schedule"] = "paper_async"
    for name in ("episodes", "batch_size", "n_action_steps", "num_inference_steps"):
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("seed", "overlap_actions"):
        if type(options[name]) is not int or options[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    video_limit = options["video_episodes_per_task"]
    if video_limit != "all" and (type(video_limit) is not int or video_limit < 0):
        raise ValueError(
            "video_episodes_per_task must be 'all' or a nonnegative integer"
        )
    if (
        type(options["video_fps"]) not in (int, float)
        or not math.isfinite(options["video_fps"])
        or options["video_fps"] <= 0
    ):
        raise ValueError("video_fps must be a finite positive number")
    for name, default in defaults.items():
        if isinstance(default, bool) and type(options[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if options["suite"] not in SUITES:
        raise ValueError(f"Unknown LIBERO suite: {options['suite']}")
    if options["batch_size"] > options["episodes"]:
        raise ValueError("batch_size must not exceed episodes")
    if type(options["action_quant"]) is not int or options["action_quant"] != 1:
        raise ValueError("This case requires action_quant=1")
    if options["delay_state_with_observation"] is not True:
        raise ValueError(
            "This case requires images and state from the same observation"
        )
    if options["quant_ladder"] not in LADDERS:
        raise ValueError("Unsupported quant_ladder")
    if options["quant_compute_dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("Unsupported quant_compute_dtype")
    if options["quant_selected_profile"] is not None and not isinstance(
        options["quant_selected_profile"], str
    ):
        raise ValueError("quant_selected_profile must be a string or null")
    if options["quant_ladder"] != "none" and options["compile_model"]:
        raise ValueError("Quantized smoke runs require compile_model=false")
    paper_contract(options)
    return options


def paper_contract(options):
    # Resolve our own helper by file, including under runpy/spawn imports.
    spec = importlib.util.spec_from_file_location(
        "pi05_paper_async_contract", Path(__file__).with_name("paper_async.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.paper_async_contract(options)


def write_episode_summary(output, ledger):
    source = Path(__file__).resolve().parents[3] / "tools/summarize_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "_robotics_experiment_summary", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.write_run_summary(output, ledger.rows, max_steps=ledger.max_steps)
    print(module.markdown_report(report))
    return {
        "markdown": str(output / "episode-summary.md"),
        "json": str(output / "episode-summary.json"),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog="Experiment flags override --config, then case defaults. "
        "Source paths may also be provided via ROBOTICS_* environment variables.",
    )
    paths = parser.add_argument_group("source and resource paths")
    for name in (
        "sim-source",
        "checkpoint",
        "tokenizer",
        "libero-config-dir",
        "quant-source",
        "kernel-source",
    ):
        env_name = "ROBOTICS_" + name.upper().replace("-", "_")
        value = os.environ.get(env_name)
        default = Path(value) if value and value.strip() else None
        paths.add_argument(
            f"--{name}",
            type=Path,
            default=default,
            required=default is None and name not in {"quant-source", "kernel-source"},
            help=f"Path; defaults to ${env_name} when set",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New result directory (must not already exist)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON preset; ordinary experiments can use flags only",
    )
    parser.add_argument("--gpu", help="One physical GPU index or full GPU UUID")
    parser.add_argument("--quant-profile", type=Path)
    experiment = parser.add_argument_group("experiment options (override JSON)")
    experiment.add_argument(
        "--suite", choices=sorted(SUITES), default=argparse.SUPPRESS
    )
    experiment.add_argument(
        "--schedule", choices=("sync", "paper_async"), default=argparse.SUPPRESS
    )
    for name in (
        "episodes",
        "batch-size",
        "seed",
        "n-action-steps",
        "num-inference-steps",
    ):
        experiment.add_argument(f"--{name}", type=int, default=argparse.SUPPRESS)
    experiment.add_argument(
        "--overlap-actions",
        type=int,
        default=argparse.SUPPRESS,
        help="Paper n-prime: primitive history steps, 0..n-action-steps",
    )
    experiment.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    video = parser.add_argument_group("optional episode videos")
    video_mode = video.add_mutually_exclusive_group()
    video_mode.add_argument(
        "--record-video",
        action="store_const",
        const="all",
        dest="video_episodes_per_task",
        default=argparse.SUPPRESS,
        help="Record every evaluated episode from initial through terminal frame",
    )
    video_mode.add_argument(
        "--no-record-video",
        action="store_const",
        const=0,
        dest="video_episodes_per_task",
        default=argparse.SUPPRESS,
        help="Disable video (the default)",
    )
    video_mode.add_argument(
        "--video-episodes-per-task",
        type=int,
        default=argparse.SUPPRESS,
        help="Record the first N evaluated episodes per task; default 0 disables video",
    )
    video.add_argument(
        "--video-fps",
        type=float,
        default=argparse.SUPPRESS,
        help="Playback FPS (default 30); does not change simulator control timing",
    )
    quant = parser.add_argument_group("quantization")
    selection = quant.add_mutually_exclusive_group()
    selection.add_argument(
        "--quant",
        choices=("none",),
        help="Disable quantization; no built-in quantized model preset is available",
    )
    selection.add_argument(
        "--quant-ladder",
        choices=sorted(LADDERS),
        default=argparse.SUPPRESS,
        help="Advanced: use with an explicit profile",
    )
    quant.add_argument("--quant-selected-profile", default=argparse.SUPPRESS)
    quant.add_argument(
        "--quant-compute-dtype",
        choices=("float16", "bfloat16", "float32"),
        default=argparse.SUPPRESS,
    )
    for name in (
        "quantize-vlm-text",
        "quantize-vision-tower",
        "quantize-diffusion-head",
        "quantize-suffix-head",
        "quantize-mm-projector",
    ):
        quant.add_argument(
            f"--{name}",
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
        )
    timing = parser.add_argument_group("optional paper timing estimate")
    for name in ("paper-action-time-ms", "paper-inference-time-ms"):
        timing.add_argument(f"--{name}", type=float, default=argparse.SUPPRESS)
    timing.add_argument("--paper-inference-time-source", default=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check files and print the plan without importing model or simulator code",
    )
    return parser


def build_plan(args):
    case_fields = read_object(DEFAULT_CONFIG)
    cli_options = {
        name: value for name, value in vars(args).items() if name in case_fields
    }
    quant_profile = args.quant_profile
    if args.quant is not None:
        if "quant_selected_profile" in cli_options or quant_profile is not None:
            raise ValueError(
                "--quant none cannot be combined with custom profile flags; "
                "use --quant-ladder for a custom profile"
            )
        cli_options.update(quant_ladder="none", quant_selected_profile=None)
    options = load_options(args.config, cli_options)
    record_all = options["video_episodes_per_task"] == "all"
    video_limit = (
        options["episodes"] if record_all else options["video_episodes_per_task"]
    )
    source = args.sim_source.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    libero_config = args.libero_config_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError(
            f"Output directory already exists; choose a new directory: {output}"
        )
    if args.gpu is not None and not re.fullmatch(r"\d+|GPU-[A-Za-z0-9-]+", args.gpu):
        raise ValueError("--gpu must select one GPU index or UUID")
    if not args.dry_run and args.gpu is None:
        raise ValueError("Actual execution requires an explicit --gpu")
    evaluator_path = required_file(source / "vlash/eval_libero.py")
    tree = ast.parse(evaluator_path.read_text())
    fields = {
        node.target.id
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "EvalConfig"
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    needed = {
        "runtime_stack",
        "async_delay",
        "delay_state_with_observation",
        "action_quant",
        "quant_ladder",
    }
    if not needed <= fields:
        raise ValueError(
            f"Incompatible sim evaluator; missing EvalConfig fields: {sorted(needed - fields)}"
        )
    checkpoint_config = read_object(required_file(checkpoint / "config.json"))
    if checkpoint_config.get("type") != "pi05":
        raise ValueError("This entry requires a LeRobot pi05 checkpoint")
    declared_source = " ".join(
        str(checkpoint_config.get(key, "")) for key in ("repo_id", "pretrained_path")
    )
    if "vlash" in declared_source.lower():
        raise ValueError(
            "This case does not use VLASH-finetuned checkpoint declarations"
        )
    horizon = checkpoint_config.get(
        "chunk_size", checkpoint_config.get("action_horizon")
    )
    if type(horizon) is not int or options["n_action_steps"] > horizon:
        raise ValueError("n_action_steps exceeds the declared checkpoint horizon")
    checkpoint_files = [
        required_file(checkpoint / "model.safetensors"),
        checkpoint / "config.json",
    ]
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = required_file(checkpoint / name)
        checkpoint_files.append(path)
        for step in read_object(path).get("steps", []):
            state_file = step.get("state_file")
            if state_file:
                relative = Path(state_file)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"Processor state_file must stay inside checkpoint: {state_file}"
                    )
                checkpoint_files.append(required_file(checkpoint / relative))
    if not tokenizer.is_dir() or not any(
        (tokenizer / name).is_file() for name in ("tokenizer.model", "tokenizer.json")
    ):
        raise ValueError(f"Tokenizer assets are missing: {tokenizer}")
    config_file = required_file(libero_config / "config.yaml")
    sources = {"sim": {"path": str(source), "identity": source_identity(source)}}
    arguments = {
        "policy.path": str(checkpoint),
        "output_dir": str(output),
        "env.type": "libero",
        "env.task": options["suite"],
        "eval.n_episodes": options["episodes"],
        "eval.batch_size": options["batch_size"],
        "eval.runtime_stack": "lerobot",
        "eval.quant_ladder": options["quant_ladder"],
        "eval.async_delay": options["overlap_actions"],
        "eval.action_quant": 1,
        "eval.delay_state_with_observation": True,
        "policy.device": "cuda",
        "policy.use_amp": False,
        "policy.compile_model": options["compile_model"],
        "policy.n_action_steps": options["n_action_steps"],
        "policy.num_inference_steps": options["num_inference_steps"],
        "seed": options["seed"],
        "num_gpus": 1,
    }
    profile_identity = None
    if options["quant_ladder"] != "none":
        if not all(
            (
                args.quant_source,
                args.kernel_source,
                quant_profile,
                options["quant_selected_profile"],
            )
        ):
            raise ValueError(
                "Quantization requires --quant-source, --kernel-source, --quant-profile and quant_selected_profile"
            )
        quant_source = args.quant_source.expanduser().resolve()
        kernel_source = args.kernel_source.expanduser().resolve()
        profile = required_file(quant_profile.expanduser().resolve())
        required_file(quant_source / "vlash/quantization/runtime_apply.py")
        required_file(kernel_source / "eval/quant_linear.py")
        profiles = read_object(profile).get("profiles", [])
        if not isinstance(profiles, list) or not all(
            isinstance(p, dict) for p in profiles
        ):
            raise ValueError("quant profile must contain a profiles list of objects")
        selected = [
            p for p in profiles if p.get("name") == options["quant_selected_profile"]
        ]
        if len(selected) != 1:
            raise ValueError("Selected quant profile name must occur exactly once")
        names_key = (
            "w8_short_names"
            if options["quant_ladder"] == "fp16_to_w8a8"
            else "w4_short_names"
        )
        names = selected[0].get(names_key)
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name.strip() for name in names)
        ):
            raise ValueError(
                f"Selected quant profile requires a nonempty list[str] in {names_key}"
            )
        if len(set(names)) != len(names):
            raise ValueError("Selected quant profile contains duplicate target names")
        sources["quant"] = {
            "path": str(quant_source),
            "identity": source_identity(quant_source),
        }
        sources["kernel"] = {
            "path": str(kernel_source),
            "identity": source_identity(kernel_source),
        }
        profile_identity = file_hash(profile)
        arguments.update(
            {
                "eval.quant_source_root": str(quant_source),
                "eval.quant_profiles_file": str(profile),
                "eval.quant_selected_profile": options["quant_selected_profile"],
                "eval.quant_skip_calibration": True,
                "eval.quant_compute_dtype": options["quant_compute_dtype"],
            }
        )
        arguments.update(
            {
                f"eval.{key}": options[key]
                for key in options
                if key.startswith("quantize_")
            }
        )
    identity = {
        "entrypoint_sha256": {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "run.py",
                "load_guard.py",
                "paper_async.py",
                "case.json",
                "libero_adapter.py",
                "evaluation_audit.py",
                "video_recorder.py",
            )
        },
        "timing_tool_sha256": file_hash(
            Path(__file__).resolve().parents[3] / "tools/compare_speedups.py"
        ),
        "summary_tools_sha256": {
            name: file_hash(Path(__file__).resolve().parents[3] / "tools" / name)
            for name in ("summarize_experiment.py", "episode_statistics.py")
        },
        "case": options,
        "sources": {key: value["identity"] for key, value in sources.items()},
        "checkpoint": {
            p.relative_to(checkpoint).as_posix(): file_hash(p) for p in checkpoint_files
        },
        "tokenizer": {
            p.name: file_hash(p) for p in sorted(tokenizer.iterdir()) if p.is_file()
        },
        "libero_config": file_hash(config_file),
        "quant_profile": profile_identity,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    return {
        "format": "pi05-libero-smoke-v1",
        "run_kind": "functional_smoke" if options["episodes"] == 1 else "evaluation",
        "environment_protocol": "explicit_initial_state_v1",
        "case_fingerprint": fingerprint,
        "identity": identity,
        "sources": sources,
        "checkpoint": str(checkpoint),
        "tokenizer": str(tokenizer),
        "libero_config_dir": str(libero_config),
        "output_dir": str(output),
        "gpu": args.gpu,
        "schedule": options["schedule"],
        "paper_async": paper_contract(options),
        "video": {
            "enabled": video_limit > 0,
            "record_all_episodes": record_all,
            "max_episodes_per_task": video_limit,
            "fps": options["video_fps"],
            "camera": "native_primary",
            "directory": str(output / "videos"),
        },
        "upstream_arguments": [
            f"--{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in arguments.items()
        ],
    }


def egl_device_index(selector):
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
    raise ValueError("Cannot resolve GPU UUID to an EGL device index")


def audit_quantized_modules(models, ladder):
    """Record installed wrapper markers; this is not CUDA-kernel verification."""
    if len(models) != 1:
        raise RuntimeError("Trial must load exactly one audited policy")
    modules = {
        name: getattr(module, "_vlash_qserve_scheme")
        for name, module in models[0].named_modules()
        if getattr(module, "_vlash_qserve_scheme", None) in {"w8a8", "w4a4"}
    }
    if ladder == "none":
        if modules:
            raise RuntimeError("Unquantized baseline contains quantized module markers")
        return modules
    expected = "w8a8" if ladder == "fp16_to_w8a8" else "w4a4"
    if expected not in modules.values():
        raise RuntimeError(
            f"Requested {expected}, but no corresponding quantized wrappers were installed"
        )
    return modules


def execute(plan):
    # Load our own standard-library guard before exposing external imports.
    from load_guard import checkpoint_load_guard
    from libero_adapter import make_lerobot_libero_env
    from evaluation_audit import evaluation_audit
    from video_recorder import EpisodeVideoRecorder

    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "case-manifest.json"
    manifest = {
        **plan,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "paper-async.json").write_text(
        json.dumps(plan["paper_async"], indent=2) + "\n"
    )
    original_argv = sys.argv[:]
    original_path = sys.path[:]
    original_cwd = Path.cwd()
    try:
        os.environ.update(
            {
                "CUDA_VISIBLE_DEVICES": plan["gpu"],
                "MUJOCO_GL": "egl",
                "MUJOCO_EGL_DEVICE_ID": egl_device_index(plan["gpu"]),
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "LIBERO_CONFIG_PATH": plan["libero_config_dir"],
                "VLASH_PALIGEMMA_TOKENIZER": plan["tokenizer"],
                # Capture through our audited recorder, avoiding the external
                # writer thread's swallowed errors and unnecessary frame collection.
                "VLASH_MAX_EPISODES_RENDERED": "0",
            }
        )
        if "kernel" in plan["sources"]:
            os.environ["VLASH_QSERVE_ROOT"] = plan["sources"]["kernel"]["path"]
        sys.path.insert(0, plan["sources"]["sim"]["path"])
        os.chdir(output)
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        import torch

        evaluator = importlib.import_module("vlash.eval_libero")
        expected = Path(plan["sources"]["sim"]["path"]) / "vlash/eval_libero.py"
        if Path(evaluator.__file__).resolve() != expected.resolve():
            raise RuntimeError("Imported a different VLASH evaluator than requested")
        manifest["runtime"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("lerobot", "transformers", "numpy", "mujoco", "robosuite")
            },
        }
        sys.argv = [str(expected), *plan["upstream_arguments"]]
        recorder = (
            EpisodeVideoRecorder(
                output,
                plan["video"]["max_episodes_per_task"],
                plan["video"]["fps"],
                evaluator.write_video,
            )
            if plan["video"]["enabled"]
            else None
        )
        with checkpoint_load_guard(
            PI05Policy, output / "checkpoint-load.json"
        ) as loaded_models:
            with evaluation_audit(
                evaluator,
                make_lerobot_libero_env,
                output,
                plan["identity"]["case"]["episodes"],
                video_recorder=recorder,
            ) as episode_ledger:
                evaluator.main()
            if recorder is not None:
                manifest["video"]["files"] = recorder.paths
            manifest["quantized_modules"] = audit_quantized_modules(
                loaded_models, plan["identity"]["case"]["quant_ladder"]
            )
        audit = read_object(required_file(output / "checkpoint-load.json"))
        if audit.get("status") != "passed":
            raise RuntimeError("Trial ended without a successful checkpoint audit")
        read_object(required_file(output / "eval_results.json"))
        manifest["max_primitive_steps"] = episode_ledger.max_steps
        manifest["summary_files"] = write_episode_summary(output, episode_ledger)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (output / "failure.json").write_text(
            json.dumps(
                {"error": manifest["error"], "traceback": traceback.format_exc()},
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path
        os.chdir(original_cwd)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
    except (ValueError, OSError, SyntaxError, TypeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    execute(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
