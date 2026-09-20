#!/usr/bin/env python3
"""Capture real LIBERO initial observations for repeatable inference validation."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("pi05_libero", "cosmos_libero"), required=True
    )
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-states", default="0,1,2")
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--libero-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    variable = (
        "ROBOTICS_LIBERO_CONFIG_DIR"
        if args.case == "pi05_libero"
        else "ROBOTICS_COSMOS_LIBERO_CONFIG_DIR"
    )
    config = args.libero_config or os.environ.get(variable)
    if not config:
        parser.error(f"Provide --libero-config or ${variable}")
    ids = [int(v) for v in args.initial_states.split(",")]
    if not ids or min(ids) < 0 or args.task_id < 0 or args.env_seed < 0:
        parser.error("IDs and seed must be nonnegative")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID="0",
        LIBERO_CONFIG_PATH=str(Path(config).expanduser().resolve()),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    records = []
    for initial in ids:
        if args.case == "pi05_libero":
            sys.path.insert(0, str(root / "benchmarks/static/pi05_libero"))
            from libero_adapter import make_lerobot_libero_env
            from lerobot.envs.utils import preprocess_observation

            env = make_lerobot_libero_env(
                args.suite,
                task.name,
                initial,
                {"observation_width": 256, "observation_height": 256},
            )
            try:
                raw, _ = env.reset(seed=args.env_seed)
            finally:
                env.close()

            def vectorize(value):
                if isinstance(value, dict):
                    return {k: vectorize(v) for k, v in value.items()}
                return np.expand_dims(value, 0).copy()

            observation = {
                k: v.numpy() for k, v in preprocess_observation(vectorize(raw)).items()
            }
        else:
            from robotics_bench.simulators.libero import LiberoSuite

            env = LiberoSuite(args.suite).make_simulator(
                args.task_id, env_seed=args.env_seed
            )
            try:
                observation = env.reset(initial)
            finally:
                env.close()
        target = output / f"observation-{initial:03d}.npz"
        np.savez(target, **observation)
        record = {
            "case": args.case,
            "suite": args.suite,
            "task_id": args.task_id,
            "task": task.language,
            "initial_state_id": initial,
            "env_seed": args.env_seed,
            "path": target.name,
        }
        target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
    (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
