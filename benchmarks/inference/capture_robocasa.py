#!/usr/bin/env python3
"""Capture actual three-camera RoboCasa observations for offline policy replay."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robotics_bench.simulators.robocasa import (  # noqa: E402
    NativeRoboCasaSimulator,
    TASK_MAX_STEPS,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--task", choices=tuple(TASK_MAX_STEPS), default="TurnOffMicrowave"
    )
    parser.add_argument("--layout-id", type=int, default=1)
    parser.add_argument("--style-id", type=int, default=1)
    parser.add_argument("--env-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robocasa-source", type=Path)
    parser.add_argument("--controller-config", type=Path)
    args = parser.parse_args(argv)
    if not args.gpu.isdigit():
        parser.error("Use a physical numeric GPU index for RoboCasa EGL capture")
    if min([args.layout_id, args.style_id, *args.env_seeds]) < 0 or len(
        set(args.env_seeds)
    ) != len(args.env_seeds):
        parser.error("Layout, style and unique seeds must be nonnegative")
    resources = {}
    for name, variable in (
        ("robocasa_source", "ROBOTICS_ROBOCASA_SOURCE"),
        ("controller_config", "ROBOTICS_ROBOCASA_CONTROLLER_CONFIG"),
    ):
        value = getattr(args, name) or os.environ.get(variable)
        if not value:
            parser.error(f"Provide --{name.replace('_', '-')} or ${variable}")
        resources[name] = Path(value).expanduser().resolve()
        if not resources[name].exists():
            raise FileNotFoundError(resources[name])
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID=args.gpu,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    import numpy as np

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    simulator = NativeRoboCasaSimulator(
        resources["robocasa_source"],
        resources["controller_config"],
        args.task,
        layout_id=args.layout_id,
        style_id=args.style_id,
    )
    records = []
    try:
        for index, seed in enumerate(args.env_seeds):
            observation = simulator.reset(index, seed=seed)
            target = output / f"observation-{seed:03d}.npz"
            np.savez(target, **observation)
            simulator.save_initialization(output / f"initialization-{seed:03d}")
            record = {
                "case": "cosmos_robocasa",
                "task": simulator.description,
                "task_name": args.task,
                "env_seed": seed,
                "layout_id": args.layout_id,
                "style_id": args.style_id,
                "settle_steps": simulator.settle_steps,
                "path": target.name,
                "observation_sha256": simulator.episode_metadata[
                    "initial_observation_sha256"
                ],
                "shapes": {
                    key: list(value.shape) for key, value in observation.items()
                },
            }
            target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
        (output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records, indent=2))
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
