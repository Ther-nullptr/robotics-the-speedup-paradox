"""Single-environment, native hold-last-control DOM server.

Isaac imports are deferred until SimulationApp is running. Upstream scene, IK,
camera and termination helpers live in ``native`` with their original license.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time


NATIVE_DT = 0.04
INITIAL_ACTION = [0.465906, 0.0, 0.382970, 0.008583, 0.921765, 0.020404, 0.387116, 1.0]


def select_action(
    messages, episode_id, current_index, previous_index=-1, *, with_metadata=False
):
    """Select latest valid same-episode target, rejecting stale cross-reset traffic."""
    import numpy as np

    selected = None
    metadata = {}
    rejected = 0
    for message in messages:
        if not isinstance(message, dict) or "action" not in message:
            rejected += 1
            continue
        index = message.get("observation_index")
        if (
            type(message.get("episode_id")) is not int
            or message["episode_id"] != episode_id
            or type(index) is not int
            or not previous_index <= index <= current_index
        ):
            rejected += 1
            continue
        # Multiple streamed actions may legitimately share an observation index.
        try:
            action = np.asarray(message["action"], dtype=np.float32)
        except (ValueError, TypeError):
            rejected += 1
            continue
        if action.shape != (1, 8) or not np.isfinite(action).all():
            rejected += 1
            continue
        if np.linalg.norm(action[0, 3:7]) < 1e-8:
            rejected += 1
            continue
        selected = action
        metadata = {
            key: message[key]
            for key in (
                "chunk_id",
                "action_index",
                "action_index_in_chunk",
                "chunk_observation_index",
                "chunk_observation_sim_time_s",
                "chunk_observation_wall_s",
            )
            if key in message
        }
        previous_index = index
    result = (selected, previous_index, rejected)
    return (*result, metadata) if with_metadata else result


def valid_model_progress(message, episode_id):
    return (
        isinstance(message, dict)
        and message.get("kind") == "model_progress"
        and type(message.get("episode_id")) is int
        and message["episode_id"] == episode_id
        and type(message.get("chunk_id")) is int
        and message["chunk_id"] >= 0
    )


def action_age(metadata, sim_time_s, wall_time_s):
    """Age of the generating observation, not the latest action-selection image."""
    result = {}
    for domain, current in (("sim", sim_time_s), ("wall", wall_time_s)):
        source = (
            metadata.get(f"chunk_observation_{domain}_time_s")
            if domain == "sim"
            else metadata.get("chunk_observation_wall_s")
        )
        if source is None:
            result[f"action_age_{domain}_ms"] = None
        elif not math.isfinite(source) or source > current + 1e-6:
            raise ValueError("Invalid chunk observation time")
        else:
            result[f"action_age_{domain}_ms"] = (current - source) * 1000
    return result


def dependency_versions():
    """Record installed package metadata without triggering simulator imports."""
    import importlib.metadata
    import platform

    versions = {"python": platform.python_version()}
    for label, packages in {
        "isaaclab": ("isaaclab",),
        "isaacsim": ("isaacsim", "isaacsim-app"),
        "torch": ("torch",),
    }.items():
        versions[label] = None
        for package in packages:
            try:
                versions[label] = importlib.metadata.version(package)
                break
            except importlib.metadata.PackageNotFoundError:
                pass
    return versions


def termination_reason(success, active_terms):
    """Separate the actual time budget from native truncated failure terms.

    DOM marks dropped objects as Isaac time_out=True too; the manager's aggregate
    time_outs therefore does not identify an exhausted episode budget.
    """
    if success:
        return "success"
    return "timeout" if "time_out" in active_terms else "native_failure"


def terminal_ack(message, episode_id):
    return (
        isinstance(message, dict)
        and message.get("ack") is True
        and type(message.get("episode_id")) is int
        and message["episode_id"] == episode_id
    )


def _write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def _drain(socket):
    import zmq

    messages = []
    while True:
        try:
            messages.append(socket.recv_pyobj(flags=zmq.NOBLOCK))
        except zmq.Again:
            return messages


def wait_for_ack(obs_socket, act_socket, terminal, timeout, app):
    """Republish terminal until its episode-specific ACK; never reset on timeout."""
    deadline = time.monotonic() + timeout
    while app.is_running() and time.monotonic() < deadline:
        obs_socket.send_pyobj(terminal)
        if any(terminal_ack(m, terminal["episode_id"]) for m in _drain(act_socket)):
            return
        time.sleep(0.05)
    raise TimeoutError(f"Terminal ACK missing for episode {terminal['episode_id']}")


def run_episode(
    env,
    obs_socket,
    act_socket,
    app,
    episode_id,
    instruction,
    seed,
    record,
    allow_policy_starvation=False,
):
    import torch
    from robotics_bench.dynamicvla_dom.native import simulation
    from robotics_bench.dynamicvla_dom.native.configs.termination_cfg import (
        get_done_term,
    )

    native = env.unwrapped
    term_manager = native.termination_manager
    done_term = get_done_term(term_manager.active_terms)
    if done_term is None:
        raise ValueError("DOM configuration has no supported success termination")
    step_dt = float(native.step_dt)
    physics_dt = float(native.physics_dt)
    if not math.isclose(physics_dt, NATIVE_DT) or native.cfg.decimation != 1:
        raise ValueError(
            "Initial DOM adapter requires native 0.04 s physics and decimation=1"
        )
    last_action = torch.tensor([INITIAL_ACTION], device=native.device)
    frames, path = [], []
    steps = accepted = held = rejected = 0
    previous_index = -1
    last_metadata = {}
    applied_chunk_ids = []
    step_time = None
    started = time.perf_counter()
    trace = []
    progress_chunks = set()
    while app.is_running():
        tick = time.perf_counter()
        cameras = simulation.get_camera_views(native.scene.sensors, ["rgb"])
        state = simulation.get_curr_state(
            ee_state=native.scene["ee_frame"].data,
            object_state=native.scene["object"].data,
            env_origins=native.scene["robot"].data.root_pos_w,
            robot_quat=native.scene["robot"].data.root_quat_w,
            device=native.device,
        )
        if record:
            frames.append(cameras)
        path.append(state["end_effector"]["pos"].cpu().numpy().tolist())
        observation = {
            "task": instruction,
            "episode_id": episode_id,
            "trial_seed": seed,
            "index": steps,
            "step_dt_s": step_dt,
            "dt_scale": 1.0 if step_time is None else max(1.0, step_time / step_dt),
            "sim_time_s": steps * step_dt,
            "wall_time_s": time.perf_counter(),
            "observation.state": {
                "end_effector": {
                    k: v.cpu().numpy() for k, v in state["end_effector"].items()
                }
            },
            **{f"observation.images.{k}": v["rgb"] for k, v in cameras.items()},
        }
        obs_socket.send_pyobj(observation)
        messages = _drain(act_socket)
        for message in messages:
            if valid_model_progress(message, episode_id):
                progress_chunks.add(message["chunk_id"])
        action_messages = [
            message
            for message in messages
            if not valid_model_progress(message, episode_id)
        ]
        action, previous_index, invalid, metadata = select_action(
            action_messages, episode_id, steps, previous_index, with_metadata=True
        )
        rejected += invalid
        if action is None:
            held += 1
        else:
            last_action = torch.from_numpy(action).to(native.device)
            last_metadata = metadata
            accepted += 1
        control_wall_s = time.perf_counter()
        age = action_age(last_metadata, steps * step_dt, control_wall_s)
        trace.append(
            {
                "control_index": steps,
                "sim_time_s": steps * step_dt,
                "wall_time_s": control_wall_s,
                "held": action is None,
                "observation_index": previous_index if previous_index >= 0 else None,
                "action": last_action.cpu().numpy().tolist(),
                **last_metadata,
                **age,
            }
        )
        env.step(last_action)
        # Count only the selected target actually handed to a successful env.step;
        # superseded queue messages and repeated hold steps add no new chunk.
        chunk_id = last_metadata.get("chunk_id")
        if chunk_id is not None and chunk_id not in applied_chunk_ids:
            applied_chunk_ids.append(chunk_id)
        steps += 1
        step_time = time.perf_counter() - tick
        if step_time < step_dt:
            time.sleep(step_dt - step_time)
        success = bool(term_manager.get_term(done_term).all())
        if success or bool(term_manager.dones.all()):
            # Study mode defers no-action validity to the terminal ACK and the
            # supervisor's completed-generation ledger check. A scene may end
            # before a slow result is released; that is not a disconnected model.
            if accepted == 0 and not allow_policy_starvation:
                raise RuntimeError(
                    "Episode terminated without receiving a model action; not a policy failure"
                )
            termination_terms = [
                name
                for name in term_manager.active_terms
                if bool(term_manager.get_term(name).all())
            ]
            reason = termination_reason(success, termination_terms)
            return (
                {
                    "episode_id": episode_id,
                    "init_state_id": episode_id,
                    "env_seed": seed,
                    "success": success,
                    "primitive_steps": steps,
                    "max_primitive_steps": int(native.max_episode_length),
                    "termination_reason": reason,
                    "termination_terms": termination_terms,
                    "actions_accepted": accepted,
                    "policy_starved": accepted == 0,
                    "generation_progress_chunks": len(progress_chunks),
                    "applied_chunk_ids": applied_chunk_ids,
                    "applied_chunks": len(applied_chunk_ids),
                    "held_control_steps": held,
                    "actions_rejected": rejected,
                    "sim_duration_s": steps * step_dt,
                    "wall_duration_s": time.perf_counter() - started,
                    "physics_dt_s": physics_dt,
                    "control_dt_s": step_dt,
                    "instruction": instruction,
                    "ee_path": path,
                },
                frames,
                trace,
            )
    raise RuntimeError("Isaac application closed before episode termination")


def run(args, app):
    import numpy as np
    import torch
    import zmq
    from robotics_bench.dynamicvla_dom.native.environment import get_test_env
    from robotics_bench.dynamicvla_dom.native.instruction_generator import (
        InstructionGenerator,
    )
    from robotics_bench.dynamicvla_dom.native import simulation

    output = Path(args.output_dir)
    from robotics_bench.dynamicvla_dom.object_motion import scale_object_speed

    config, object_motion = scale_object_speed(
        json.loads(Path(args.env_cfg).read_text()), args.object_speed_scale
    )
    if not config["scene"]["robot"]["spawn"]["usd_path"].endswith(
        "panda_instanceable.usd"
    ):
        raise ValueError("Initial DOM adapter supports only Franka scenes")
    config["seed"] = args.seed
    env = None
    context = zmq.Context()
    obs_socket = context.socket(zmq.PUB)
    act_socket = context.socket(zmq.PULL)
    obs_socket.setsockopt(zmq.LINGER, 0)
    act_socket.setsockopt(zmq.LINGER, 0)
    completed = 0
    try:
        obs_socket.bind(f"tcp://{args.host}:{args.img_port}")
        act_socket.bind(f"tcp://{args.host}:{args.act_port}")
        _write(
            output / "ready.json",
            {"pid": os.getpid(), "img_port": args.img_port, "act_port": args.act_port},
        )
        deadline = time.monotonic() + args.connection_timeout
        hello = None
        while app.is_running() and time.monotonic() < deadline:
            hello = next(
                (m for m in _drain(act_socket) if isinstance(m, dict) and "vla" in m),
                None,
            )
            if hello:
                break
            time.sleep(0.05)
        if not hello:
            raise TimeoutError("No model client handshake before connection timeout")
        env = get_test_env(
            config,
            1,
            args.scene_dir,
            args.object_dir,
            args.physics_time_step,
            args.tolerance,
            args.device,
            args.disable_fabric,
            args.path_tracing,
        )
        _write(
            output / "effective-config.json",
            {
                "physics_dt_s": float(env.unwrapped.physics_dt),
                "control_dt_s": float(env.unwrapped.step_dt),
                "decimation": env.unwrapped.cfg.decimation,
                "episode_length_s": config["episode_length_s"],
                "camera_update_periods_s": {
                    k: float(s.cfg.update_period)
                    for k, s in env.unwrapped.scene.sensors.items()
                    if type(s).__name__ == "Camera"
                },
                "franka_usd": args.franka_usd,
                "control_protocol": "native_hold_last",
                "video_fps": 24,
                "input_config": config,
                "object_speed_scale": args.object_speed_scale,
                "object_motion": object_motion,
                "dependency_versions": dependency_versions(),
            },
        )
        successes = 0
        for episode_id in range(args.episodes):
            seed = args.seed + episode_id
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            instruction = InstructionGenerator.generate_instruction(
                config["instruction"]
            )
            env.reset(seed=seed)
            target = env.unwrapped.scene["object"]
            reset_motion = {
                **object_motion,
                "default_lin_vel_mps": target.data.default_root_state[0, 7:10]
                .detach()
                .cpu()
                .tolist(),
                "reset_lin_vel_mps": target.data.root_lin_vel_w[0]
                .detach()
                .cpu()
                .tolist(),
                "reset_ang_vel_rad_s": target.data.root_ang_vel_w[0]
                .detach()
                .cpu()
                .tolist(),
            }
            result, frames, trace = run_episode(
                env,
                obs_socket,
                act_socket,
                app,
                episode_id,
                instruction,
                seed,
                args.record_video,
                args.allow_policy_starvation,
            )
            result["task"] = Path(args.env_cfg).stem
            result["object_speed_scale"] = args.object_speed_scale
            result["object_motion"] = reset_motion
            result["video_path"] = None
            name = f"episode-{episode_id:06d}"
            if frames:
                views = {}
                for frame in frames:
                    for camera, values in frame.items():
                        for sensor, image in values.items():
                            views.setdefault(f"{camera}_{sensor}", []).append(
                                image.squeeze(0)
                            )
                video = output / f"{name}.mp4"
                simulation.dump_video(
                    simulation.get_frames(views, state_keys=[]), video
                )
                result["video_path"] = str(video)
                result["video_frames"] = len(frames)
            with (output / f"{name}-controls.jsonl").open("w") as stream:
                for row in trace:
                    stream.write(json.dumps(row) + "\n")
            with (output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
                stream.flush()
            completed += 1
            successes += int(result["success"])
            terminal = {**result, "env_name": result["task"], "eps_name": name}
            wait_for_ack(obs_socket, act_socket, terminal, args.ack_timeout, app)
            # Preserve upstream manager reinitialization before the next explicit reset.
            manager = env.unwrapped.termination_manager
            manager.__init__(manager.cfg, env.unwrapped)
        obs_socket.send_pyobj(
            {
                "vla": hello["vla"],
                "completed": True,
                "success_rates": {Path(args.env_cfg).stem: successes / args.episodes},
            }
        )
        _write(
            output / "server-status.json",
            {"status": "completed", "completed_episodes": completed},
        )
    except BaseException as error:
        _write(
            output / "server-status.json",
            {
                "status": "failed",
                "completed_episodes": completed,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
        obs_socket.close()
        act_socket.close()
        context.term()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("scene-dir", "object-dir", "env-cfg", "output-dir"):
        result.add_argument(f"--{name}", required=True)
    result.add_argument(
        "--franka-usd", default=os.getenv("ROBOTICS_DYNAMICVLA_FRANKA_USD")
    )
    result.add_argument("--object-speed-scale", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--episodes", type=int, default=1)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--img-port", type=int, default=3186)
    result.add_argument("--act-port", type=int, default=3188)
    result.add_argument(
        "--physics-time-step", type=float, choices=[NATIVE_DT], default=NATIVE_DT
    )
    result.add_argument("--tolerance", type=float, default=0.07)
    result.add_argument("--disable-fabric", action="store_true")
    result.add_argument("--path-tracing", action="store_true")
    result.add_argument(
        "--allow-policy-starvation",
        action="store_true",
        help="Study mode: defer no-action validity to completed generation evidence at episode ACK",
    )
    result.add_argument("--record-video", action="store_true")
    result.add_argument("--connection-timeout", type=float, default=600)
    result.add_argument("--ack-timeout", type=float, default=120)
    return result


def main(argv=None):
    argument_parser = parser()
    # Standard help remains available on CPU without importing Isaac.
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        argument_parser.add_argument("--device", default="cuda:0")
        argument_parser.add_argument("--headless", action="store_true")
        argument_parser.add_argument("--enable_cameras", action="store_true")
        argument_parser.parse_args(argv)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(argument_parser)
    args = argument_parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "episodes.jsonl").exists():
        argument_parser.error("Output directory already contains episode records")
    if args.episodes < 1 or args.connection_timeout <= 0 or args.ack_timeout <= 0:
        argument_parser.error("Episode count and timeouts must be positive")
    if not args.franka_usd or not Path(args.franka_usd).is_file():
        argument_parser.error(
            "Provide an existing offline Franka USD via --franka-usd or ROBOTICS_DYNAMICVLA_FRANKA_USD"
        )
    os.environ["ROBOTICS_DYNAMICVLA_FRANKA_USD"] = str(Path(args.franka_usd).resolve())
    if not args.enable_cameras:
        argument_parser.error("DOM policy observations require --enable_cameras")
    launcher = None
    try:
        launcher = AppLauncher(args)
        run(args, launcher.app)
    except BaseException as error:
        if not (output / "server-status.json").exists():
            _write(
                output / "server-status.json",
                {
                    "status": "failed",
                    "completed_episodes": 0,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise
    finally:
        if launcher is not None:
            launcher.app.close()


if __name__ == "__main__":
    main()
