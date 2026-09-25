"""Measure native DynamicVLA from raw CPU observations to a CPU action chunk.

Runs offline on an idle GPU. No simulator, artificial delay, dt_scale pacing,
model loading or warmup is included in the timed region.
"""

from pathlib import Path
import argparse
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from robotics_bench.dynamicvla_dom.runner import digest, require_idle_gpus, write  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or args.repeats < 2:
        parser.error("steps/warmup must be positive and repeats at least two")
    devices = require_idle_gpus([args.gpu])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        CUDA_VISIBLE_DEVICES=args.gpu,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
    )
    import numpy as np
    import torch
    from robotics_bench.engines.dynamicvla import DynamicVLAEngine
    from robotics_bench.models.dynamicvla.preprocessing import transform_observations
    from robotics_bench.models.dynamicvla.rotations import get_quaternion

    metadata = json.loads(args.metadata.read_text())
    observation = {
        "index": 0,
        "dt_scale": 1.0,
        "task": metadata["task"],
        "observation.state": {"end_effector": {}},
    }
    with np.load(args.input, allow_pickle=False) as data:
        for key in data.files:
            if key.startswith("end_effector."):
                observation["observation.state"]["end_effector"][
                    key.split(".", 1)[1]
                ] = data[key].copy()
            else:
                observation[key] = data[key].copy()
    engine = DynamicVLAEngine(
        args.checkpoint, num_steps=args.steps, rotation="euler", seed=42
    )
    manifest = dict(
        status="running",
        case="dynamicvla_dom",
        scope="raw CPU observations -> CPU absolute action chunk, synchronized",
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_config_sha256=digest(args.checkpoint / "config.json"),
        input_sha256=digest(args.input),
        input_metadata=metadata,
        gpu=args.gpu,
        gpu_preflight=devices,
        steps=args.steps,
        warmup=args.warmup,
        repeats=args.repeats,
        timing_exclusions=[
            "load",
            "compile",
            "warmup",
            "simulator",
            "recording",
            "native_pacing",
            "artificial_delay",
        ],
    )
    source = Path(__file__).resolve().parents[2] / "src/robotics_bench"
    manifest["source_sha256"] = {
        str(p.relative_to(source)): digest(p)
        for p in (source / "models/dynamicvla").rglob("*.py")
    }
    manifest["source_sha256"]["engines/dynamicvla.py"] = digest(
        source / "engines/dynamicvla.py"
    )
    manifest["benchmark_sha256"] = digest(__file__)
    write(args.output_dir / "manifest.json", manifest)
    try:
        engine.load()
        policy = engine.policy
        policy.config.disable_dt_scale_sleep = True

        def call(seed):
            torch.manual_seed(seed)
            policy.reset()
            torch.cuda.synchronize()
            start = time.perf_counter()
            batch = transform_observations(
                [observation] * policy.config.n_obs_steps,
                "euler",
                policy.config.input_features,
                "cuda:0",
            )
            state = batch["observation.state"][:, -1:, :].clone()
            actions = policy.predict_action_chunk(batch)
            if policy.config.use_delta_action:
                actions[..., :-1] += state[..., : actions.shape[-1] - 1]
            actions = actions.cpu().numpy()
            quaternion = get_quaternion(
                actions[..., 3:-1].reshape(-1, 3), "euler"
            ).reshape(*actions.shape[:-1], 4)
            absolute = np.concatenate(
                [actions[..., :3], quaternion, actions[..., -1:]], axis=-1
            )
            torch.cuda.synchronize()
            return absolute, (time.perf_counter() - start) * 1000

        checks = []
        for seed in (42, 195, 0):
            policy.config.measure_inference = False
            native, _ = call(seed)
            policy.config.measure_inference = True
            measured, _ = call(seed)
            checks.append(
                {
                    "seed": seed,
                    "exact": bool(np.array_equal(native, measured)),
                    "max_abs": float(np.max(np.abs(native - measured))),
                }
            )
        if not all(item["exact"] for item in checks):
            raise RuntimeError(
                "Measurement instrumentation changes fixed-input actions"
            )
        policy.config.measure_inference = False
        for i in range(args.warmup):
            call(42 + i)
        samples = [call(42 + i)[1] for i in range(args.repeats)]
        result = dict(
            samples_ms=samples,
            median_ms=float(np.median(samples)),
            p95_ms=float(np.percentile(samples, 95)),
            instrumentation_checks=checks,
        )
        write(args.output_dir / "measurements.json", result)
        manifest.update(
            status="completed",
            engine=engine.describe(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
        )
        print(json.dumps(result, indent=2))
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        engine.close()
        write(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
