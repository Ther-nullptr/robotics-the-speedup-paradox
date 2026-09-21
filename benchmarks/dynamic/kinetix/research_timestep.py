"""Research-only native-grid refinement study; production case defaults are unchanged."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot(state, mask):
    import jax
    import numpy as np

    polygon, circle = jax.device_get((state.polygon, state.circle))
    return {
        key: np.concatenate([getattr(polygon, key), getattr(circle, key)])[mask]
        for key in ("position", "velocity", "rotation", "angular_velocity")
    }


def compare(reference, candidate, tolerance=1e-6):
    import numpy as np

    count = min(len(reference["position"]), len(candidate["position"]))
    result = {"common_control_steps": count - 1, "position_tolerance": tolerance}
    for key in reference:
        delta = candidate[key][:count] - reference[key][:count]
        if key == "rotation":
            delta = np.arctan2(np.sin(delta), np.cos(delta))
        result[f"{key}_rmse_at_common_end"] = float(np.sqrt(np.mean(delta[-1] ** 2)))
        if key == "position":
            per_step = np.max(np.abs(delta).reshape(count, -1), axis=1)
            indices = np.flatnonzero(per_step > tolerance)
            result["first_position_divergence_control"] = (
                int(indices[0]) if len(indices) else None
            )
            result["position_rmse_control_5"] = (
                float(np.sqrt(np.mean(delta[5] ** 2))) if count > 5 else None
            )
    return result


def run(args):
    os.environ.update(
        JAX_PLATFORMS="cpu" if args.cpu else "cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONDONTWRITEBYTECODE="1",
        SDL_VIDEODRIVER="dummy",
    )
    sys.dont_write_bytecode = True
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from robotics_bench.kinetix.native.kinetix.environment import env as native
    from robotics_bench.kinetix.native.kinetix.util.saving import load_from_json_file
    from robotics_bench.kinetix.native.jax2d.engine import create_empty_sim

    output = args.output_dir.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    factors = sorted(set(int(x) for x in args.factors.split(",")))
    if not factors or factors[0] != 1 or any(f < 1 for f in factors):
        raise ValueError("Factors must include1 and be positive integers")
    tasks = args.levels.split(",")
    patterns = args.patterns.split(",")
    if any(p not in {"zero", "fixed-motor"} for p in patterns):
        raise ValueError("Unknown open-loop action pattern")
    if not 1 <= args.controls <= 256:
        raise ValueError("controls must be in1..256")
    manifest = {
        "format": "kinetix-timestep-study-v1",
        "status": "running",
        "factors": factors,
        "levels": tasks,
        "patterns": patterns,
        "observed_control_prefix": args.controls,
        "action_noise_std": 0,
        "injected_latency_ms": 0,
        "policy_used": False,
        "termination_cadence": "native_control_boundary",
        "precision": "float32",
        "devices": [
            {"kind": d.device_kind, "platform": d.platform} for d in jax.devices()
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tasks": {},
        "records": [],
    }
    write_json(output / "manifest.json", manifest)
    env_cache, rows = {}, []
    seed = 0
    if args.only_freefall:
        _, static, base = load_from_json_file(
            str(ROOT / "src/robotics_bench/kinetix/levels" / f"{tasks[0]}.json")
        )
        env_cache["freefall"] = native.make_kinetix_env_from_name(
            "Kinetix-Symbolic-Continuous-v1", static_env_params=static
        )
    with (output / "trials.jsonl").open("x") as ledger:
        for task in [] if args.only_freefall else tasks:
            path = ROOT / "src/robotics_bench/kinetix/levels" / f"{task}.json"
            level, static, base = load_from_json_file(str(path))
            manifest["tasks"][task] = {
                "level_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "native_env_params": asdict(base),
                "native_static_params": asdict(static),
            }
            mask = np.concatenate(
                [
                    np.asarray(level.polygon.active & (level.polygon.inverse_mass > 0)),
                    np.asarray(level.circle.active & (level.circle.inverse_mass > 0)),
                ]
            )
            control_dt = float(base.dt) * int(static.frame_skip)
            for pattern in patterns:
                baseline = None
                for factor in factors:
                    variants = (
                        ["fixed"] if factor == 1 else ["fixed", "scaled_collision"]
                    )
                    variant_static = static.replace(
                        frame_skip=int(static.frame_skip) * factor
                    )
                    cache_key = json.dumps(asdict(variant_static), sort_keys=True)
                    if cache_key not in env_cache:
                        env_cache[cache_key] = native.make_kinetix_env_from_name(
                            "Kinetix-Symbolic-Continuous-v1",
                            static_env_params=variant_static,
                        )
                    env = env_cache[cache_key]
                    for beta_rule in variants:
                        beta = float(base.baumgarte_coefficient_collision) / (
                            factor if beta_rule == "scaled_collision" else 1
                        )
                        params = base.replace(
                            dt=float(base.dt) / factor,
                            baumgarte_coefficient_collision=beta,
                        )
                        assert math.isclose(
                            float(params.dt) * variant_static.frame_skip,
                            control_dt,
                            abs_tol=1e-15,
                        )
                        assert params.max_timesteps == base.max_timesteps
                        _, state = env.reset_to_level(
                            jax.random.key(seed), level, params
                        )
                        action = jnp.zeros(
                            env.action_space(params).shape, dtype=jnp.float32
                        )
                        if pattern == "fixed-motor":
                            values = jnp.where(
                                jnp.arange(static.num_motor_bindings) % 2 == 0,
                                0.5,
                                -0.5,
                            )
                            action = action.at[: static.num_motor_bindings].set(values)
                        frames = [snapshot(state, mask)]
                        rewards = []
                        terminal_control, terminal_success = None, None
                        started = time.perf_counter()
                        print(
                            f"task={task} pattern={pattern} factor={factor} beta={beta_rule}",
                            flush=True,
                        )
                        for index in range(args.controls):
                            _, state, reward, done, info = env.step_env(
                                jax.random.fold_in(jax.random.key(seed), index),
                                state,
                                action,
                                params,
                            )
                            reward, done, goal = jax.device_get(
                                (reward, done, info["GoalR"])
                            )
                            frame = snapshot(state, mask)
                            if not all(
                                np.isfinite(value).all() for value in frame.values()
                            ):
                                raise RuntimeError(
                                    f"Nonfinite state in {task}/{pattern}/{factor}/{beta_rule}"
                                )
                            frames.append(frame)
                            rewards.append(float(reward))
                            if bool(done):
                                terminal_control, terminal_success = (
                                    index + 1,
                                    bool(goal),
                                )
                                break
                        trajectory = {
                            key: np.stack([f[key] for f in frames]) for key in frames[0]
                        }
                        if factor == 1:
                            baseline = trajectory
                        metrics = compare(baseline, trajectory)
                        name = f"{task}_{pattern}_r{factor}_{beta_rule}"
                        np.savez_compressed(output / f"{name}.npz", **trajectory)
                        row = {
                            "task": task,
                            "pattern": pattern,
                            "factor": factor,
                            "collision_rule": beta_rule,
                            "physics_dt_seconds": float(params.dt),
                            "frame_skip": variant_static.frame_skip,
                            "control_dt_seconds": control_dt,
                            "native_episode_budget": params.max_timesteps,
                            "collision_beta": beta,
                            "collision_beta_over_dt": beta / float(params.dt),
                            "solver_iterations_per_physics_step": static.num_solver_iterations,
                            "solver_iterations_per_simulated_second": static.num_solver_iterations
                            / float(params.dt),
                            "observed_controls": len(frames) - 1,
                            "first_terminal_control": terminal_control,
                            "first_terminal_seconds": terminal_control * control_dt
                            if terminal_control is not None
                            else None,
                            "terminal_success": terminal_success,
                            "state_finite": True,
                            "return_observed": sum(rewards),
                            "host_seconds_including_compilation": time.perf_counter()
                            - started,
                            "trajectory": f"{name}.npz",
                            **metrics,
                        }
                        rows.append(row)
                        ledger.write(json.dumps(row, allow_nan=False) + "\n")
                        ledger.flush()
                        manifest["records"] = rows
                        write_json(output / "manifest.json", manifest)
                        print(
                            f"  common_end_position_rmse={metrics['position_rmse_at_common_end']:.6g} terminal_control={terminal_control}",
                            flush=True,
                        )
    freefall = []
    if args.freefall:
        # Isolated single active body: no contacts, joints, motors or policy.
        engine = next(iter(env_cache.values())).physics_engine
        static = next(iter(env_cache.values())).static_env_params
        state = create_empty_sim(static, add_floor=False, add_walls_and_ceiling=False)
        state = state.replace(
            gravity=jnp.array([0.0, -9.81], dtype=jnp.float32),
            circle=state.circle.replace(
                active=state.circle.active.at[0].set(True),
                position=state.circle.position.at[0].set(jnp.array([0.0, 5.0])),
                inverse_mass=state.circle.inverse_mass.at[0].set(1.0),
                inverse_inertia=state.circle.inverse_inertia.at[0].set(1.0),
                radius=state.circle.radius.at[0].set(0.1),
            ),
        )
        assert int(state.polygon.active.sum()) == 0
        assert int(state.circle.active.sum()) == 1
        assert int(state.joint.active.sum()) == int(state.thruster.active.sum()) == 0
        manifest["freefall_setup"] = {
            "active_circles": 1,
            "active_polygons": 0,
            "joints": 0,
            "thrusters": 0,
            "gravity_y": -9.81,
        }
        command = jnp.zeros(static.num_joints + static.num_thrusters)

        @jax.jit
        def integrate(initial, params, steps):
            return jax.lax.fori_loop(
                0,
                steps,
                lambda _, current: engine.step(current, params, command)[0],
                initial,
            )

        duration = 8 / 30
        exact_y = 5 - 0.5 * 9.81 * duration**2
        for factor in factors:
            params = base.replace(dt=(1 / 60) / factor)
            final = integrate(state, params, 16 * factor)
            y = float(jax.device_get(final.circle.position[0, 1]))
            row = {
                "factor": factor,
                "physics_dt_seconds": float(params.dt),
                "duration_seconds": duration,
                "analytic_y": exact_y,
                "simulated_y": y,
                "absolute_error": abs(y - exact_y),
            }
            freefall.append(row)
            print("Freefall:", row, flush=True)
    manifest.update(status="completed", freefall=freefall)
    write_json(output / "manifest.json", manifest)
    if rows:
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(output / "freefall.json", freefall)
    print(f"Study completed: {len(rows)} trajectory comparisons", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="mjc_walker,mjc_half_cheetah,mjc_swimmer")
    parser.add_argument("--factors", default="1,2,4,40")
    parser.add_argument("--patterns", default="zero,fixed-motor")
    parser.add_argument("--controls", type=int, default=32)
    parser.add_argument(
        "--freefall", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--only-freefall", action="store_true")
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--gpu", type=int)
    device.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.only_freefall and not args.freefall:
        parser.error("--only-freefall cannot be combined with --no-freefall")
    run(args)


if __name__ == "__main__":
    main()
