#!/usr/bin/env python3
"""Measure owned policy inference, validate actions, and render measured history.

Timing starts from a prepared CPU observation and ends with a CPU action chunk.
Model load, packing, compilation, graph capture and warmup are excluded and
reported separately. The simulator is not part of this inference timing domain.
"""

from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotics_bench.optimizations.config import (  # noqa: E402
    OptimizationConfig,
    SWITCHES,
    TACTIC_IDS,
    measurement_configurations,
)
from robotics_bench.profiling.history import build_history  # noqa:E402


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--case", choices=("pi05_libero", "cosmos_libero"), required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--enable", action="append", choices=SWITCHES, default=[])
    p.add_argument(
        "--precision", choices=("bf16", "int8", "int4", "fp8", "fp4"), default="bf16"
    )
    p.add_argument(
        "--quant-scope",
        action="append",
        choices=("text", "expert", "vision", "projector", "dit"),
    )
    p.add_argument("--integer-tactic", type=int, choices=TACTIC_IDS, default=0)
    p.add_argument("--quant-tier", type=int, choices=range(11))
    p.add_argument(
        "--variants",
        type=Path,
        help="Measure explicit variant configurations in one loaded-model cohort",
    )
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-seed", action="append", type=int)
    p.add_argument("--validation-input", action="append", type=Path, default=[])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument(
        "--profile",
        action="store_true",
        help="Collect a separate diagnostic CUDA trace after clean timings",
    )
    p.add_argument("--profile-skill", type=Path)
    p.add_argument("--render-python", default=sys.executable)
    for name in (
        "checkpoint",
        "tokenizer",
        "cosmos-source",
        "dataset-stats",
        "text-embeddings",
        "vae-checkpoint",
    ):
        p.add_argument("--" + name, type=Path)
    return p


def resource(args, name, variable):
    value = getattr(args, name) or os.environ.get(variable)
    if not value:
        raise ValueError(f"Provide --{name.replace('_', '-')} or ${variable}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner(args, output):
    import torch

    if args.case == "pi05_libero":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from robotics_bench.models.pi05.modeling_pi05 import PI05Policy
        from robotics_bench.optimizations.pi05 import optimize_pi05

        sys.path.insert(0, str(ROOT / "benchmarks/static/pi05_libero"))
        from load_guard import checkpoint_load_guard

        checkpoint = resource(args, "checkpoint", "ROBOTICS_CHECKPOINT")
        tokenizer = resource(args, "tokenizer", "ROBOTICS_TOKENIZER")
        cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
        cfg.compile_model = False
        cfg.device = "cuda"
        cfg.n_action_steps = 5
        cfg.num_inference_steps = args.steps or 10
        with checkpoint_load_guard(PI05Policy, output / "checkpoint-load.json"):
            policy = (
                PI05Policy.from_pretrained(str(checkpoint), config=cfg)
                .eval()
                .to("cuda")
            )
        pre, post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {"rename_map": {}},
                "tokenizer_processor": {"tokenizer_name": str(tokenizer)},
            },
        )

        def call(observation, task, seed):
            batch = {k: torch.from_numpy(v.copy()) for k, v in observation.items()}
            batch["task"] = [task]
            return post(policy.predict_action_chunk(pre(batch))).cpu().numpy()

        return (
            call,
            lambda c: optimize_pi05(policy.model, c),
            lambda: None,
            {
                "checkpoint": str(checkpoint),
                "checkpoint_config_sha256": digest(checkpoint / "config.json"),
                "steps": cfg.num_inference_steps,
                "horizon": cfg.chunk_size,
                "runtime": "owned",
            },
        )
    from robotics_bench.engines.cosmos import CosmosEngine
    from robotics_bench.models.cosmos.runtime import bind_owned_runtime
    from robotics_bench.optimizations.cosmos import optimize_cosmos

    source = resource(args, "cosmos_source", "ROBOTICS_COSMOS_SOURCE")
    checkpoint = resource(args, "checkpoint", "ROBOTICS_COSMOS_CHECKPOINT")
    engine = CosmosEngine(
        source,
        checkpoint,
        resource(args, "dataset_stats", "ROBOTICS_COSMOS_DATASET_STATS"),
        resource(args, "text_embeddings", "ROBOTICS_COSMOS_TEXT_EMBEDDINGS"),
        resource(args, "vae_checkpoint", "ROBOTICS_COSMOS_VAE_CHECKPOINT"),
        num_inference_steps=args.steps or 5,
    )
    engine.load([args.task], output / "checkpoint-load.json")
    bindings = bind_owned_runtime(engine._model)
    from robotics_bench.models.cosmos import cosmos_utils

    engine._backend.utils = cosmos_utils
    engine.reset(("inference_benchmark", 0))
    return (
        engine.infer_chunk,
        lambda c: optimize_cosmos(engine._model, c),
        engine.close,
        {
            "checkpoint": str(checkpoint),
            "steps": args.steps or 5,
            "horizon": 16,
            "runtime": "owned",
            "owned_bindings": bindings,
        },
    )


def main(argv=None):
    args = parser().parse_args(argv)
    if (
        args.repeats < 3
        or args.warmup < 1
        or args.seed < 0
        or (args.steps is not None and args.steps < 1)
    ):
        raise ValueError(
            "Require repeats>=3, warmup>=1, nonnegative seed and positive steps"
        )
    config = OptimizationConfig(
        tuple(args.enable),
        args.precision,
        tuple(
            args.quant_scope or (("text",) if args.case == "pi05_libero" else ("dit",))
        ),
        args.integer_tactic,
        args.quant_tier,
    )
    variants = json.loads(args.variants.read_text()) if args.variants else None
    configurations = measurement_configurations(config, variants)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        WANDB_MODE="disabled",
    )
    import torch
    import numpy as np

    inputs = []
    for path in [args.input, *args.validation_input]:
        path = path.expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            inputs.append({k: data[k].copy() for k in data.files})
    source_hashes = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / "src").rglob("*"))
        if p.suffix in (".py", ".cu", ".cuh")
    }
    git_head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    metadata = {
        "case": args.case,
        "config": config.to_dict(),
        "configurations": [{"id": name, **c.to_dict()} for name, c in configurations],
        "variants_sha256": digest(args.variants) if args.variants else None,
        "task": args.task,
        "input_sha256": [
            digest(p.expanduser().resolve())
            for p in [args.input, *args.validation_input]
        ],
        "source_sha256": source_hashes,
        "git_head": git_head,
        "git_status": subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_selector": args.gpu,
        "cpu_threads": torch.get_num_threads(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "validation_seeds": list(
            dict.fromkeys(args.validation_seed or [args.seed, 0, 195])
        ),
        "status": "running",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    call = apply = close = None
    records = []
    profiles = []
    precision_references = {}
    try:
        start = time.perf_counter()
        call, apply, close, model_info = load_runner(args, output)
        metadata.update(model=model_info, load_seconds=time.perf_counter() - start)
        reference = []
        for name, current in configurations:
            start = time.perf_counter()
            with apply(current) as coverage:
                with torch.inference_mode():
                    for _ in range(args.warmup):
                        torch.manual_seed(args.seed)
                        call(inputs[0], args.task, args.seed)
                    torch.cuda.synchronize()
                    preparation = time.perf_counter() - start
                    samples = []
                    for _ in range(args.repeats):
                        torch.manual_seed(args.seed)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        call(inputs[0], args.task, args.seed)
                        torch.cuda.synchronize()
                        samples.append((time.perf_counter() - start) * 1000)
                    exact = True
                    error = 0.0
                    squared = 0.0
                    count = 0
                    checks = []
                    validation_actions = []
                    for index, observation in enumerate(inputs):
                        for seed in metadata["validation_seeds"]:
                            torch.manual_seed(seed)
                            actual = call(observation, args.task, seed)
                            validation_actions.append(actual.copy())
                            if name == "original":
                                reference.append(actual.copy())
                            expected = reference[len(checks)]
                            if (
                                actual.shape != expected.shape
                                or not np.isfinite(actual).all()
                            ):
                                raise ValueError("Invalid action output")
                            difference = actual.astype(np.float64) - expected.astype(
                                np.float64
                            )
                            local_exact = bool(np.array_equal(actual, expected))
                            local_error = float(np.max(np.abs(difference)))
                            exact &= local_exact
                            error = max(error, local_error)
                            squared += float(np.sum(difference * difference))
                            count += actual.size
                            checks.append(
                                {
                                    "input": index,
                                    "seed": seed,
                                    "exact": local_exact,
                                    "max_abs": local_error,
                                }
                            )
                    record = {
                        "id": name,
                        "precision": current.precision_label,
                        "switches": list(current.enabled),
                        "scopes": list(current.scopes),
                        "tactic": current.tactic,
                        "quant_tier": current.quant_tier,
                        "samples_ms": samples,
                        "median_ms": float(np.median(samples)),
                        "exact": exact,
                        "max_abs": error,
                        "rmse": float(np.sqrt(squared / count)),
                        "checks": checks,
                        "preparation_seconds": preparation,
                        "coverage": deepcopy(coverage),
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    }
                    signature = (
                        current.precision,
                        current.scopes if current.precision != "bf16" else (),
                        current.quant_tier,
                    )
                    numerical_id, numerical_actions = precision_references.setdefault(
                        signature, (name, validation_actions)
                    )
                    record["same_precision_reference"] = {
                        "id": numerical_id,
                        "exact": all(
                            np.array_equal(a, b)
                            for a, b in zip(validation_actions, numerical_actions)
                        ),
                        "max_abs": max(
                            float(
                                np.max(
                                    np.abs(a.astype(np.float64) - b.astype(np.float64))
                                )
                            )
                            for a, b in zip(validation_actions, numerical_actions)
                        ),
                    }
                    np.savez_compressed(
                        output / (name + "-validation-actions.npz"),
                        **{f"check_{i}": a for i, a in enumerate(validation_actions)},
                    )
                    record["speedup_vs_baseline"] = (
                        records[0]["median_ms"] / record["median_ms"]
                        if records
                        else 1.0
                    )
                    if args.profile:
                        # Validation may have changed the captured signature.
                        # Rewarm the measured input before diagnostic collection.
                        torch.manual_seed(args.seed)
                        call(inputs[0], args.task, args.seed)
                        torch.cuda.synchronize()
                        torch.manual_seed(args.seed)
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                        ) as prof:
                            call(inputs[0], args.task, args.seed)
                            torch.cuda.synchronize()
                        prof.export_chrome_trace(str(output / (name + "-trace.json")))
                        from robotics_bench.profiling.breakdown import summarize

                        profiles.append(
                            summarize(
                                json.loads(
                                    (output / (name + "-trace.json")).read_text()
                                ),
                                name,
                            )
                        )
                        (output / (name + "-profile.txt")).write_text(
                            prof.key_averages().table(
                                sort_by="self_cuda_time_total", row_limit=40
                            )
                        )
                    records.append(record)
                    (output / "measurements.json").write_text(
                        json.dumps(records, indent=2) + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in record.items()
                                if k not in ("coverage", "checks")
                            }
                        ),
                        flush=True,
                    )
        protocol = {
            "id": args.case,
            "label": args.case + " fixed-input policy service",
            "workload": {
                "input_sha256": metadata["input_sha256"],
                "task": args.task,
                "seed": args.seed,
                **model_info,
                "boundary": "Prepared CPU observation to CPU action chunk; simulator/load/warmup excluded",
                "warmup": args.warmup,
            },
            "environment": {
                "gpu": metadata["gpu"],
                "selector": args.gpu,
                "torch": torch.__version__,
                "cpu_threads": metadata["cpu_threads"],
            },
            "metric": {
                "name": "Complete policy service",
                "unit": "ms",
                "statistic": "median",
            },
        }
        current = next(
            (
                r["id"]
                for r in reversed(records)
                if r["precision"] == "bf16" and r["exact"]
            ),
            "original",
        )
        history = build_history(
            records,
            protocol=protocol,
            cohort=output.name,
            source="measurements.json",
            revision=git_head
            + (" + working changes" if metadata["git_status"] else ""),
            current=current,
        )
        (output / "ledger.json").write_text(json.dumps(history, indent=2) + "\n")
        if profiles:
            from robotics_bench.profiling.breakdown import document

            breakdown = document(
                profiles,
                [r["id"] + "-trace.json" for r in records],
                args.case + " measured GPU work",
            )
            (output / "profile-breakdown.json").write_text(
                json.dumps(breakdown, indent=2) + "\n"
            )
        if args.profile_skill:
            renderer = (
                args.profile_skill.expanduser().resolve()
                / "scripts/render_optimization_history.py"
            )
            subprocess.run(
                [
                    args.render_python,
                    str(renderer),
                    str(output / "ledger.json"),
                    "--output-dir",
                    str(output / "figures"),
                    "--format",
                    "both",
                    "--update-index",
                ],
                check=True,
            )
        if args.profile_skill and profiles:
            from robotics_bench.profiling.breakdown import pages

            renderer = (
                args.profile_skill.expanduser().resolve()
                / "scripts/render_profile_breakdown.py"
            )
            for index, page in enumerate(pages(breakdown), 1):
                page_path = output / f"profile-breakdown-p{index:02}.json"
                page_path.write_text(json.dumps(page, indent=2) + "\n")
                subprocess.run(
                    [
                        args.render_python,
                        str(renderer),
                        str(page_path),
                        "--output-dir",
                        str(output / "figures"),
                    ],
                    check=True,
                )
        metadata["status"] = "completed"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if close is not None:
            close()
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
