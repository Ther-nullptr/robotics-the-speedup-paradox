"""Native DynamicVLA client with episode-safe transport and generation evidence."""

import argparse
import json
from pathlib import Path
import time
import traceback

from robotics_bench.dynamicvla_dom.runner import write


def append(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")
    stream.flush()


def run(args):
    import numpy as np
    import torch
    import zmq

    from robotics_bench.engines.dynamicvla import DynamicVLAEngine

    output = args.output_dir
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    engine = DynamicVLAEngine(
        args.checkpoint,
        streaming=args.streaming,
        device=args.device,
        num_steps=args.num_steps,
        rotation=args.rotation,
        seed=args.seed,
        measure_inference=args.measure_inference,
        extra_delay_ms=args.extra_delay_ms,
        episode_seed_mode=args.episode_seed_mode,
    )
    context = zmq.Context()
    obs = context.socket(zmq.SUB)
    obs.setsockopt_string(zmq.SUBSCRIBE, "")
    obs.setsockopt(zmq.RCVHWM, 32)
    obs.connect(f"tcp://{args.host}:{args.img_port}")
    actions = context.socket(zmq.PUSH)
    actions.setsockopt(zmq.LINGER, 1000)
    actions.setsockopt(zmq.SNDTIMEO, int(args.timeout * 1000))
    actions.connect(f"tcp://{args.host}:{args.act_port}")
    active = None
    finished = set()
    started = time.monotonic()
    last_received = started
    try:
        engine.load()
        write(output / "engine.json", engine.describe())
        actions.send_pyobj({"vla": "dynamicvla-native", "epoch": 0})
        last_received = time.monotonic()
        with (
            (output / "requests.jsonl").open("x") as requests,
            (output / "client-actions.jsonl").open("x") as delivered,
        ):
            while len(finished) < args.episodes:
                if not obs.poll(100):
                    if time.monotonic() - last_received > args.timeout:
                        raise TimeoutError(
                            "No simulator observation/terminal within timeout"
                        )
                    continue
                # Retain control messages. Only intermediate ordinary observations may be superseded.
                message = obs.recv_pyobj()
                if "success" not in message and not message.get("completed"):
                    while obs.poll(0):
                        newer = obs.recv_pyobj()
                        message = newer
                        if "success" in newer or newer.get("completed"):
                            break
                last_received = time.monotonic()
                if message.get("completed"):
                    raise RuntimeError(
                        "Simulator completed before all terminal episodes were acknowledged"
                    )
                episode = message.get("episode_id")
                if type(episode) is not int or episode not in range(args.episodes):
                    raise ValueError("Invalid simulator episode identity")
                if episode in finished:
                    if "success" in message:
                        actions.send_pyobj({"ack": True, "episode_id": episode})
                    continue
                if "success" in message:
                    if active != episode:
                        raise RuntimeError(
                            "Episode terminated without a model observation"
                        )
                    for event in engine.finish_episode():
                        append(requests, event)
                    append(
                        delivered,
                        {
                            "kind": "terminal",
                            "episode_id": episode,
                            "success": message["success"],
                            "received_wall_s": time.perf_counter(),
                        },
                    )
                    finished.add(episode)
                    active = None
                    actions.send_pyobj({"ack": True, "episode_id": episode})
                    continue
                if episode != active:
                    if active is not None or episode != len(finished):
                        raise RuntimeError(
                            "Simulator advanced episode without terminal/reset handshake"
                        )
                    engine.reset(episode_id=episode)
                    active = episode
                    snapshots = output / "observations"
                    snapshots.mkdir(exist_ok=True)
                    arrays = {
                        key: value
                        for key, value in message.items()
                        if key.startswith("observation.images.")
                    }
                    arrays.update(
                        {
                            f"end_effector.{key}": value
                            for key, value in message["observation.state"][
                                "end_effector"
                            ].items()
                        }
                    )
                    np.savez(snapshots / f"{episode:06d}.npz", **arrays)
                    write(
                        snapshots / f"{episode:06d}.json",
                        {
                            key: message[key]
                            for key in (
                                "episode_id",
                                "index",
                                "task",
                                "sim_time_s",
                                "step_dt_s",
                            )
                        },
                    )
                begin = time.perf_counter()
                action = engine.select_action(message)
                end = time.perf_counter()
                for event in engine.drain_events():
                    append(requests, event)
                    if args.measure_inference and event["kind"] == "chunk_generated":
                        actions.send_pyobj(
                            {
                                "kind": "model_progress",
                                "episode_id": event["episode_id"],
                                "chunk_id": event["chunk_id"],
                            }
                        )
                if action is not None:
                    if action.shape != (1, 8) or not np.isfinite(action).all():
                        raise ValueError("Model returned invalid DOM action")
                    actions.send_pyobj(
                        {
                            "action": action,
                            "episode_id": episode,
                            "observation_index": message["index"],
                            **engine.last_action_metadata,
                        }
                    )
                append(
                    delivered,
                    {
                        "kind": "action_selection",
                        "episode_id": episode,
                        "observation_index": message["index"],
                        "observation_sim_time_s": message["sim_time_s"],
                        "started_wall_s": begin,
                        "finished_wall_s": end,
                        "action_sent": action is not None,
                        "action": action.tolist() if action is not None else None,
                        "selected_action": engine.last_action_metadata
                        if action is not None
                        else None,
                    },
                )
        write(output / "engine.json", engine.describe())
        write(
            output / "client-status.json",
            dict(
                status="completed",
                episodes=len(finished),
                elapsed_s=time.monotonic() - started,
            ),
        )
    finally:
        try:
            engine.close()
        finally:
            obs.close(linger=0)
            actions.close()
            context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--img-port", type=int, default=3186)
    parser.add_argument("--act-port", type=int, default=3188)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument(
        "--rotation", choices=("euler", "rotvec", "quat"), default="euler"
    )
    parser.add_argument("--measure-inference", action="store_true")
    parser.add_argument("--extra-delay-ms", type=float, default=0)
    parser.add_argument("--episode-seed-mode", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        write(
            args.output_dir / "client-status.json",
            dict(status="failed", error=str(error)),
        )
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
