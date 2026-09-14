#!/usr/bin/env python3
"""Validate v1 manifests and serialized traces without running a simulator."""

import argparse
import json
import math
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


class ContractError(ValueError):
    """A serialized artifact violates the declared contract."""


def read_json(path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (OSError, ValueError) as exc:
        raise ContractError(f"{path}: {exc}") from exc


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number {value}")
    return result


def schema_validator(name):
    schema = read_json(ROOT / "schemas" / name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def check_schema(validator, value, location):
    error = next(validator.iter_errors(value), None)
    if error:
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ContractError(f"{location}: schema at {path}: {error.message}")


def validate_manifest(manifest):
    check_schema(schema_validator("run-manifest.schema.json"), manifest, "manifest")
    domains = manifest["clock_domains"]
    budget_domain = manifest["budget"]["clock_domain"]
    if budget_domain not in domains or domains[budget_domain]["kind"] == "device":
        raise ContractError("budget clock domain must name a simulation or host domain")
    watchdog = manifest.get("host_watchdog")
    if (
        watchdog
        and domains.get(watchdog["clock_domain"], {}).get("kind") != "host_monotonic"
    ):
        raise ContractError("host watchdog must use a declared host_monotonic domain")
    ids = [episode["episode_id"] for episode in manifest["episodes"]]
    if len(ids) != len(set(ids)):
        raise ContractError("episode IDs must be unique within a run")
    protocol = manifest["protocol"]
    if protocol["execution_length"] > protocol["action_chunk_length"]:
        raise ContractError("execution_length exceeds action_chunk_length")
    metrics = manifest.get("metrics")
    if metrics and metrics["successes"] > metrics["trials"]:
        raise ContractError("successes exceed trials")


def validate_events(manifest, events):
    """Validate logical association and per-domain order, never cross-domain ages."""
    validator = schema_validator("trace-event.schema.json")
    episodes = {item["episode_id"] for item in manifest["episodes"]}
    finished = set()
    observed_episodes = set()
    active_episode = None
    observations = {}
    requests = {}
    chunks = {}
    actions = {}
    clocks = {}
    last_seq = 0
    count = 0

    for count, event in enumerate(events, 1):
        location = f"event {count}"
        check_schema(validator, event, location)
        episode = event["episode_id"]
        kind = event["event_type"]

        def fail(message):
            raise ContractError(f"{location} ({kind}): {message}")

        if event["run_id"] != manifest["run_id"]:
            fail("run_id does not match manifest")
        if episode not in episodes:
            fail("undeclared episode")
        domain = event["clock_domain"]
        if domain not in manifest["clock_domains"]:
            fail("undeclared clock domain")
        if event["event_seq"] <= last_seq:
            fail("event_seq must increase across the run")
        last_seq = event["event_seq"]
        if event["timestamp_ns"] < clocks.get(domain, 0):
            fail("timestamp moved backwards in the same clock domain")
        clocks[domain] = event["timestamp_ns"]
        if episode in finished and kind in {
            "observation_sampled",
            "inference_submitted",
            "inference_started",
            "result_released",
            "action_started",
            "action_finished",
        }:
            fail("episode already finished")
        if episode not in observed_episodes:
            if active_episode is not None and active_episode not in finished:
                fail("new episode requires the previous episode's terminal boundary")
            active_episode = episode
            observed_episodes.add(episode)

        if kind == "observation_sampled":
            observation = event["observation_id"]
            if observation in observations:
                fail("duplicate observation")
            observations[observation] = episode
        elif kind == "inference_submitted":
            request = event["request_id"]
            observation = event["observation_id"]
            if observation not in observations:
                fail("unknown observation; sample before submitting a request")
            if observations[observation] != episode:
                fail("observation belongs to a different episode")
            if request in requests:
                fail("duplicate request")
            requests[request] = {
                "episode": episode,
                "observation": observation,
                "started": False,
                "completed": False,
                "released": False,
                "dropped": False,
                "chunk": None,
                "status": None,
            }
        elif kind in {
            "inference_started",
            "inference_completed",
            "result_released",
            "result_dropped",
        }:
            request = requests.get(event["request_id"])
            if request is None:
                fail("unknown request")
            if request["episode"] != episode:
                fail("request belongs to a different episode")
            if kind == "inference_started":
                if request["started"] or request["dropped"]:
                    fail("request already started or dropped")
                request["started"] = True
            elif kind == "inference_completed":
                if not request["started"]:
                    fail("request not started")
                if request["completed"]:
                    fail("request already completed")
                request.update(completed=True, status=event["status"])
                if event["status"] == "ok":
                    chunk = event["chunk_id"]
                    if chunk in chunks:
                        fail("duplicate chunk")
                    request["chunk"] = chunk
                    chunks[chunk] = request
            elif kind == "result_released":
                if not request["completed"] or request["status"] != "ok":
                    fail("request not completed successfully")
                if request["chunk"] != event["chunk_id"]:
                    fail("chunk does not match completed request")
                if request["released"] or request["dropped"]:
                    fail("request already released or dropped")
                protocol = manifest["protocol"]
                if event["delay_mode"] != protocol["delay_mode"]:
                    fail("delay mode does not match manifest")
                if event["profile_id"] != protocol.get("profile_id"):
                    fail("profile_id does not match manifest")
                request["released"] = True
            else:
                if request["dropped"]:
                    fail("request already dropped")
                if "chunk_id" in event and event["chunk_id"] != request["chunk"]:
                    fail("dropped chunk does not match request")
                request["dropped"] = True
        elif kind in {"action_started", "action_finished"}:
            chunk = event["chunk_id"]
            request = chunks.get(chunk)
            if request is None:
                fail("unknown chunk")
            if request["episode"] != episode:
                fail("chunk belongs to a different episode")
            if not request["released"] or request["dropped"]:
                fail("chunk not released or already dropped")
            if event["observation_id"] != request["observation"]:
                fail("observation does not match the chunk's source")
            if event["action_index"] >= manifest["protocol"]["execution_length"]:
                fail("action_index exceeds execution_length")
            key = (chunk, event["action_index"])
            if kind == "action_started":
                if key in actions:
                    fail("action already started")
                actions[key] = "started"
            else:
                if actions.get(key) != "started":
                    fail("action not started or already finished")
                actions[key] = "finished"
        elif kind == "episode_finished":
            if episode in finished:
                fail("episode already finished")
            finished.add(episode)

    if not count:
        raise ContractError("trace is empty")
    missing = episodes - finished
    if missing:
        raise ContractError(
            f"missing episode_finished for {', '.join(sorted(missing))}"
        )
    return count


def trace_events(path):
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(
                        line, parse_constant=reject_constant, parse_float=finite_float
                    )
                except ValueError as exc:
                    raise ContractError(f"{path}: line {line_number}: {exc}") from exc
    except OSError as exc:
        raise ContractError(f"{path}: {exc}") from exc


def validate_files(manifest_path, trace_path):
    manifest = read_json(manifest_path)
    validate_manifest(manifest)
    count = validate_events(manifest, trace_events(trace_path))
    return manifest["run_id"], count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", action="store_true", help="validate checked-in synthetic examples"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args(argv)
    if args.examples:
        if args.manifest or args.trace:
            parser.error("--examples cannot be combined with --manifest/--trace")
        folder = ROOT / "examples" / "contracts"
        pairs = [
            (path, path.with_name(path.name.replace(".manifest.json", ".trace.jsonl")))
            for path in sorted(folder.glob("*.manifest.json"))
        ]
        if not pairs:
            parser.error("no example manifests found")
    elif args.manifest and args.trace:
        pairs = [(args.manifest, args.trace)]
    else:
        parser.error("provide --examples or both --manifest and --trace")
    try:
        for manifest_path, trace_path in pairs:
            run_id, count = validate_files(manifest_path, trace_path)
            print(f"VALID {run_id}: {count} events")
    except ContractError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
