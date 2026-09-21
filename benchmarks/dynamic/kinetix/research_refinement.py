"""Compare actual smaller physics timesteps using one recorded command tape."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from refinement import make_refined_env  # noqa: E402
from research_timestep import compare, snapshot, write_json  # noqa: E402
from robotics_bench.kinetix.command_clock import CommandClock  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="mjc_walker,car_launch")
    parser.add_argument(
        "--factors",
        default="1,2",
        help="Physics refinement factors; default compares native and 2x refinement",
    )
    parser.add_argument(
        "--variants",
        default="impulse_motor_collision",
    )
    parser.add_argument(
        "--controls",
        type=int,
        default=256,
        help="Maximum native 30 Hz ticks, including held-command ticks",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=10,
        help="Command update rate; default 10 holds each command for 3 native ticks; 30 restores the original cadence",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--policy-dir", type=Path)
    source.add_argument("--action-tape-dir", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--flow-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    factors = sorted(set(int(x) for x in args.factors.split(",")))
    if not factors or factors[0] != 1 or not 1 <= args.controls <= 256:
        parser.error("Factors must include 1; controls must be within 1..256")
    if args.flow_steps < 1:
        parser.error("flow-steps must be positive")
    try:
        CommandClock(
            control_hz=args.control_hz, native_tick_seconds=1 / 30, fixed_action=0
        )
    except ValueError as exc:
        parser.error(str(exc))
    with ExitStack() as resources:
        run(args, factors, resources)


def run(args, factors, resources):
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONDONTWRITEBYTECODE="1",
        SDL_VIDEODRIVER="dummy",
        IMAGEIO_FFMPEG_NO_PREVENT_SIGINT="1",
    )
    sys.dont_write_bytecode = True
    import jax
    import jax.numpy as jnp
    import numpy as np

    from robotics_bench.kinetix.native.kinetix.util.saving import load_from_json_file
    from robotics_bench.kinetix.policy import KinetixFlowPolicy

    finite = jax.jit(
        lambda state: jax.tree.reduce(
            jnp.logical_and, jax.tree.map(lambda x: jnp.isfinite(x).all(), state), True
        )
    )
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "seed": args.seed,
        "action_noise_std": 0,
        "action_source": "supplied_tape"
        if args.action_tape_dir
        else ("native_policy_tape" if args.policy_dir else "fixed_motor"),
        "flow_steps": args.flow_steps if args.policy_dir else None,
        "observation_clock_seconds": 1 / 30,
        "terminal_cadence": "control_boundary",
        "reward_sampling": "native_physics_starts",
        "native_physics_seconds": 1 / 60,
        "command_hz": args.control_hz,
        "execute_horizon_commands": 4,
        "nominal_policy_hz": args.control_hz / 4 if args.policy_dir else None,
        "step_unit": "native_control_tick",
        "tape_semantics": "one effective command per native tick",
        "reference_state_projection": False,
        "devices": [str(x) for x in jax.devices()],
        "rows": [],
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "benchmarks/dynamic/kinetix/refinement.py",
                "benchmarks/dynamic/kinetix/research_refinement.py",
                "benchmarks/dynamic/kinetix/research_timestep.py",
                "src/robotics_bench/kinetix/native/jax2d/engine.py",
                "src/robotics_bench/kinetix/command_clock.py",
            )
        },
    }
    write_json(output / "manifest.json", manifest)
    for task in args.levels.split(","):
        level_path = ROOT / "src/robotics_bench/kinetix/levels" / f"{task}.json"
        level, static, base = load_from_json_file(str(level_path))
        movable = np.concatenate(
            [
                np.asarray(level.polygon.active & (level.polygon.inverse_mass > 0)),
                np.asarray(level.circle.active & (level.circle.inverse_mass > 0)),
            ]
        )
        actions, baseline = [], None
        tape = (
            np.load(args.action_tape_dir / f"{task}_actions.npy", allow_pickle=False)
            if args.action_tape_dir
            else None
        )
        if tape is not None and (
            tape.ndim != 2
            or not len(tape)
            or tape.shape[1] != static.num_motor_bindings + static.num_thruster_bindings
            or not np.isfinite(tape).all()
        ):
            raise ValueError(
                "Action tape must be a nonempty finite [controls, action_dim] array"
            )
        for factor in factors:
            for variant in ["direct"] if factor == 1 else args.variants.split(","):
                print(f"START {task} r={factor} variant={variant}", flush=True)
                env, params, vstatic = make_refined_env(static, base, factor, variant)
                obs, state = env.reset_to_level(
                    jax.random.key(args.seed), level, params
                )
                policy = None
                if factor == 1 and args.policy_dir and tape is None:
                    checkpoint = args.policy_dir / f"worlds_l_{task}.pkl"
                    policy = KinetixFlowPolicy(
                        checkpoint,
                        observation_dim=int(obs.shape[-1]),
                        action_dim=env.action_space(params).shape[0],
                    )
                    policy.reset(args.seed)
                if factor == 1:
                    fixed_action = None
                    if policy is None and tape is None:
                        fixed_action = np.zeros(
                            env.action_space(params).shape, np.float32
                        )
                        fixed_action[: static.num_motor_bindings] = np.where(
                            np.arange(static.num_motor_bindings) % 2 == 0, 0.5, -0.5
                        )
                    command_clock = CommandClock(
                        control_hz=args.control_hz,
                        native_tick_seconds=float(base.dt) * static.frame_skip,
                        infer=partial(policy.infer, flow_steps=args.flow_steps)
                        if policy
                        else None,
                        tape=tape,
                        fixed_action=fixed_action,
                    )
                frames, rewards, goals = [snapshot(state, movable)], [], []
                terminal_control = None
                count = (
                    (
                        min(args.controls, len(tape))
                        if tape is not None
                        else args.controls
                    )
                    if factor == 1
                    else len(actions)
                )
                name = f"{task}_r{factor}_{variant}"
                writer = None
                video_frames = 0
                if args.record_video:
                    import imageio.v2 as imageio
                    from robotics_bench.kinetix.native.kinetix.render.renderer_pixels import (
                        make_render_pixels,
                    )

                    render = jax.jit(
                        make_render_pixels(params, vstatic.replace(downscale=1))
                    )
                    writer = imageio.get_writer(
                        output / f"{name}.mp4",
                        fps=1 / (base.dt * static.frame_skip),
                        macro_block_size=1,
                    )
                    resources.callback(writer.close)
                    writer.append_data(
                        np.asarray(jax.device_get(render(state)), dtype=np.uint8)
                    )
                    video_frames = 1
                for index in range(count):
                    if factor == 1:
                        action = command_clock.sample(index, obs)
                        actions.append(np.asarray(action))
                    obs, state, reward, done, info = env.step_env(
                        jax.random.fold_in(jax.random.key(args.seed), index),
                        state,
                        jnp.asarray(actions[index]),
                        params,
                    )
                    reward, done, goal, is_finite = jax.device_get(
                        (reward, done, info["GoalR"], finite(state))
                    )
                    frame = snapshot(state, movable)
                    if not bool(is_finite):
                        raise RuntimeError("Nonfinite physical state")
                    frames.append(frame)
                    if writer:
                        writer.append_data(
                            np.asarray(jax.device_get(render(state)), dtype=np.uint8)
                        )
                        video_frames += 1
                    rewards.append(float(reward))
                    goals.append(bool(goal))
                    if bool(done):
                        terminal_control = index + 1
                        break
                if writer:
                    writer.close()
                trajectory = {
                    key: np.stack([x[key] for x in frames]) for key in frames[0]
                }
                if factor == 1:
                    baseline = trajectory
                    np.save(output / f"{task}_actions.npy", np.stack(actions))
                np.savez_compressed(output / f"{name}.npz", **trajectory)
                row = {
                    "name": name,
                    "task": task,
                    "factor": factor,
                    "variant": variant,
                    "command_hz": args.control_hz,
                    "command_period_seconds": 1 / args.control_hz,
                    "native_ticks_per_command": command_clock.hold_ticks,
                    "command_updates": (len(rewards) + command_clock.hold_ticks - 1)
                    // command_clock.hold_ticks,
                    "policy_inference_calls": command_clock.inference_calls
                    if factor == 1
                    else 0,
                    "command_update_native_ticks": list(
                        range(0, len(rewards), command_clock.hold_ticks)
                    ),
                    "step_unit": "native_control_tick",
                    "motor_feedback_hz": 1 / base.dt
                    if "motor" in variant
                    else 1 / params.dt,
                    "joint_position_correction_hz": 1 / base.dt
                    if variant == "impulse_motor_collision_joint_clock"
                    else 1 / params.dt,
                    "joint_position_correction_coefficients_per_native_window": [
                        params.baumgarte_coefficient_joints_p
                    ]
                    + [0.0] * (factor - 1)
                    if variant == "impulse_motor_collision_joint_clock"
                    else [params.baumgarte_coefficient_joints_p] * factor,
                    "reward_sampling": "native_physics_starts",
                    "level_sha256": hashlib.sha256(level_path.read_bytes()).hexdigest(),
                    "params": asdict(params),
                    "static": asdict(vstatic),
                    "solver_iterations_per_physics_step": vstatic.num_solver_iterations,
                    "solver_iterations_per_native_window": vstatic.num_solver_iterations
                    * factor,
                    "solver_iterations_per_native_tick": vstatic.num_solver_iterations
                    * vstatic.frame_skip,
                    "physics_dt_seconds": params.dt,
                    "control_dt_seconds": params.dt * vstatic.frame_skip,
                    "observed_controls": len(rewards),
                    "executed_physics_steps": len(rewards) * vstatic.frame_skip,
                    "terminal_control": terminal_control,
                    "censored": terminal_control is None,
                    "termination_reason": (
                        "success"
                        if terminal_control and goals[-1]
                        else "budget"
                        if terminal_control and len(rewards) >= params.max_timesteps
                        else "task_terminal"
                        if terminal_control
                        else "observation_limit"
                        if len(rewards) >= args.controls
                        else "action_tape_exhausted"
                    ),
                    "terminal_seconds": terminal_control
                    * params.dt
                    * vstatic.frame_skip
                    if terminal_control
                    else None,
                    "success": bool(goals[-1]) if terminal_control else None,
                    "video": f"{name}.mp4" if writer else None,
                    "video_frames": video_frames,
                    "action_tape_sha256": hashlib.sha256(
                        np.stack(actions).tobytes()
                    ).hexdigest(),
                    "rewards": rewards,
                    **compare(baseline, trajectory),
                }
                manifest["rows"].append(row)
                write_json(output / "manifest.json", manifest)
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "name",
                                "observed_controls",
                                "terminal_control",
                                "success",
                                "position_rmse_at_common_end",
                                "velocity_rmse_at_common_end",
                            )
                        }
                    ),
                    flush=True,
                )
                del env, policy
        jax.clear_caches()
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
