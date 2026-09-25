"""Process lifecycle around DynamicVLA's native latest-observation streaming policy.

The indexed action merge remains in modeling_dynamicvla.py. This module owns
bounded result transport, episode isolation and generation evidence.
"""

from queue import Empty, Full
import time
import traceback


def worker(checkpoint, config, mailbox, results, events, stop):
    import torch

    from .modeling_dynamicvla import DynamicVLAPolicy

    try:
        if getattr(config, "runtime_seed", None) is not None:
            torch.manual_seed(config.runtime_seed)
        policy = (
            DynamicVLAPolicy.from_pretrained(
                checkpoint, config=config, local_files_only=True
            )
            .eval()
            .to(config.device)
        )
        dummy = {
            key: torch.zeros(1, config.n_obs_steps, *feat.shape, device=config.device)
            for key, feat in config.input_features.items()
        }
        dummy["task"] = ["dummy text input"]
        with torch.no_grad():
            policy._get_action_chunk(dummy)
        policy.generation_events.clear()
        results.put({"initialized": True})
        epoch = None
        while not stop.is_set():
            requested_epoch = mailbox.pop("reset", None)
            if requested_epoch is not None:
                policy.reset()
                epoch = requested_epoch
                events.put({"kind": "reset_ack", "episode_id": epoch})
            latest = mailbox.pop("obs", None)
            if latest is None:
                stop.wait(0.001)
                continue
            batch, noise = latest
            if epoch != batch["_episode_id"]:
                policy.reset()
                epoch = batch["_episode_id"]
            batch = {
                key: value.to(config.device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            noise = noise.to(config.device) if noise is not None else None
            index = batch["index"]
            state = batch["observation.state"][:, -1:, :]
            with torch.no_grad():
                actions = policy._get_action_chunk(policy._prepare_batch(batch), noise)
                if config.use_delta_action:
                    dim = actions.shape[-1] - 1
                    actions[..., :dim] += state[..., :dim]
                actions = actions.transpose(0, 1).cpu()
            event = policy.generation_events.pop()
            event.update(episode_id=epoch, result_ready_wall_s=time.perf_counter())
            result = {
                "actions": actions,
                "index": index,
                "episode_id": epoch,
                "chunk_id": event["chunk_id"],
            }
            try:
                results.put_nowait(result)
                event["result_enqueued"] = True
            except Full:
                event["result_enqueued"] = False
            events.put(event)
    except BaseException:
        events.put({"kind": "worker_error", "error": traceback.format_exc()})
        stop.set()


def start(policy, checkpoint):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    policy._manager = ctx.Manager()
    policy.q_in = policy._manager.dict()
    policy.q_out = ctx.Queue(maxsize=1)
    policy.q_events = ctx.Queue()
    policy._stop = ctx.Event()
    policy.worker = ctx.Process(
        target=worker,
        args=(
            checkpoint,
            policy.config,
            policy.q_in,
            policy.q_out,
            policy.q_events,
            policy._stop,
        ),
        daemon=True,
    )
    policy.worker.start()
    deadline = time.monotonic() + 600
    try:
        while time.monotonic() < deadline:
            check_worker(policy)
            try:
                message = policy.q_out.get(timeout=0.1)
            except Empty:
                continue
            if message.get("initialized"):
                return
            raise RuntimeError("Unexpected DynamicVLA worker initialization message")
        raise TimeoutError("DynamicVLA worker initialization exceeded 600 seconds")
    except BaseException:
        close(policy)
        raise


def drain(policy):
    collected, policy.generation_events = policy.generation_events, []
    if hasattr(policy, "q_events"):
        while True:
            try:
                event = policy.q_events.get_nowait()
            except Empty:
                break
            if event["kind"] == "worker_error":
                raise RuntimeError(event["error"])
            collected.append(event)
    return collected


def check_worker(policy):
    # Reading status must not consume ordinary generation events.
    if not policy.worker.is_alive() or policy._stop.is_set():
        detail = ""
        try:
            drain(policy)
        except RuntimeError as exc:
            detail = str(exc)
        raise RuntimeError("DynamicVLA inference worker stopped. " + detail)


def close(policy):
    if hasattr(policy, "worker"):
        policy._stop.set()
        policy.worker.join(timeout=30)
        if policy.worker.is_alive():
            policy.worker.terminate()
            policy.worker.join(timeout=5)
        for queue in (policy.q_out, policy.q_events):
            queue.close()
            queue.cancel_join_thread()
        policy._manager.shutdown()
        del policy.worker


def reset_episode(policy, episode_id):
    """Drain completed old-episode work before acknowledging a new episode."""
    policy.q_in.clear()
    policy.q_in["reset"] = episode_id
    deadline = time.monotonic() + 600
    collected = []
    while time.monotonic() < deadline:
        check_worker(policy)
        try:
            event = policy.q_events.get(timeout=0.1)
        except Empty:
            continue
        if event["kind"] == "worker_error":
            raise RuntimeError(event["error"])
        if event["kind"] == "reset_ack" and event["episode_id"] == episode_id:
            return collected
        collected.append(event)
    raise TimeoutError("DynamicVLA reset barrier exceeded 600 seconds")
