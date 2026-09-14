"""Behavior tests for serialized contracts; no policy or simulator is executed."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "validate_contracts.py"


def contract():
    manifest = {
        "schema_version": "1.0.0",
        "run_id": "synthetic-run",
        "synthetic": True,
        "protocol": {
            "schedule": "sync",
            "clock": "virtual",
            "delay_mode": "profile_replay",
            "profile_id": "synthetic-100ms",
            "physics_dt_ns": 10_000_000,
            "control_dt_ns": 50_000_000,
            "action_chunk_length": 1,
            "execution_length": 1,
        },
        "clock_domains": {
            "sim": {
                "kind": "simulation",
                "process_id": "fixture",
                "monotonic_origin": "run-start",
            },
            "host": {
                "kind": "host_monotonic",
                "process_id": "fixture",
                "monotonic_origin": "process-start",
            },
        },
        "episodes": [{"episode_id": "ep-1", "seed": 7}],
        "budget": {
            "clock_domain": "sim",
            "start_event": "first_observation_sampled",
            "limit_ns": 1_000_000_000,
        },
    }
    payloads = [
        (
            "observation_sampled",
            0,
            {
                "observation_id": "obs-1",
                "sim_tick": 0,
                "input_ref": "synthetic://blank",
            },
        ),
        ("inference_submitted", 0, {"request_id": "req-1", "observation_id": "obs-1"}),
        (
            "inference_started",
            0,
            {"request_id": "req-1", "backend": "synthetic", "worker_id": "worker-1"},
        ),
        (
            "inference_completed",
            80_000_000,
            {"request_id": "req-1", "chunk_id": "chunk-1", "status": "ok"},
        ),
        (
            "result_released",
            100_000_000,
            {
                "request_id": "req-1",
                "chunk_id": "chunk-1",
                "delay_mode": "profile_replay",
                "delay_ns": 100_000_000,
                "profile_id": "synthetic-100ms",
                "overshoot_ns": 0,
            },
        ),
        (
            "action_started",
            100_000_000,
            {"chunk_id": "chunk-1", "action_index": 0, "observation_id": "obs-1"},
        ),
        (
            "action_finished",
            150_000_000,
            {"chunk_id": "chunk-1", "action_index": 0, "observation_id": "obs-1"},
        ),
        (
            "episode_finished",
            150_000_000,
            {
                "terminal_reason": "success",
                "success": True,
                "budget_type": "fixed_budget",
            },
        ),
    ]
    events = [
        {
            "schema_version": "1.0.0",
            "run_id": manifest["run_id"],
            "episode_id": "ep-1",
            "event_seq": i,
            "clock_domain": "sim",
            "timestamp_ns": ts,
            "event_type": kind,
            **payload,
        }
        for i, (kind, ts, payload) in enumerate(payloads, 1)
    ]
    return manifest, events


def run_contract(tmp_path, manifest, events):
    manifest_path = tmp_path / "manifest.json"
    trace_path = tmp_path / "trace.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    trace_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--manifest",
            str(manifest_path),
            "--trace",
            str(trace_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def renumber(events):
    for i, event in enumerate(events, 1):
        event["event_seq"] = i


def test_valid_contract(tmp_path):
    result = run_contract(tmp_path, *contract())
    assert result.returncode == 0, result.stderr
    assert "8 events" in result.stdout


@pytest.mark.parametrize("schedule", ["sync", "history_observation", "async"])
@pytest.mark.parametrize(
    "clock,delay",
    [
        ("virtual", "zero"),
        ("virtual", "profile_replay"),
        ("realtime", "measured"),
        ("realtime", "additive"),
        ("realtime", "target_total"),
    ],
)
def test_schedule_and_clock_are_independent(tmp_path, schedule, clock, delay):
    manifest, events = contract()
    manifest["protocol"].update(schedule=schedule, clock=clock, delay_mode=delay)
    if schedule == "history_observation":
        manifest["protocol"]["history_steps"] = 1
    events[4]["delay_mode"] = delay
    if delay != "profile_replay":
        manifest["protocol"].pop("profile_id")
        events[4]["profile_id"] = None
    result = run_contract(tmp_path, manifest, events)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("wrong_version", "schema"),
        ("missing_clock", "schema"),
        ("invalid_clock_delay_pair", "schema"),
        ("unknown_clock_domain", "undeclared clock domain"),
        ("time_reversed", "timestamp moved backwards"),
        ("seq_duplicate", "event_seq must increase"),
        ("request_without_observation", "unknown observation"),
        ("completion_without_start", "not started"),
        ("release_without_completion", "not completed"),
        ("action_before_release", "not released"),
        ("wrong_action_observation", "observation does not match"),
        ("action_out_of_range", "action_index"),
        ("finish_without_start", "action not started"),
        ("duplicate_action", "action already started"),
        ("duplicate_terminal", "already finished"),
        ("action_after_terminal", "episode already finished"),
        ("request_after_terminal", "episode already finished"),
        ("cross_episode_chunk", "different episode"),
        ("release_mode_mismatch", "delay mode does not match"),
        ("duplicate_request", "duplicate request"),
        ("zero_success_mean", "schema"),
        ("too_many_successes", "successes exceed trials"),
        ("budget_unknown_clock", "budget clock domain"),
        ("missing_terminal", "missing episode_finished"),
    ],
)
def test_rejects_invalid_contracts(tmp_path, mutation, expected):
    manifest, events = contract()
    if mutation == "wrong_version":
        manifest["schema_version"] = "99.0.0"
    elif mutation == "missing_clock":
        events[0].pop("clock_domain")
    elif mutation == "invalid_clock_delay_pair":
        manifest["protocol"]["delay_mode"] = "measured"
    elif mutation == "unknown_clock_domain":
        events[0]["clock_domain"] = "undeclared"
    elif mutation == "time_reversed":
        events[5]["timestamp_ns"] = 99_000_000
    elif mutation == "seq_duplicate":
        events[1]["event_seq"] = events[0]["event_seq"]
    elif mutation == "request_without_observation":
        events[1]["observation_id"] = "missing"
    elif mutation == "completion_without_start":
        events.pop(2)
    elif mutation == "release_without_completion":
        events.pop(3)
    elif mutation == "action_before_release":
        events.pop(4)
    elif mutation == "wrong_action_observation":
        events[5]["observation_id"] = "unrelated"
    elif mutation == "action_out_of_range":
        events[5]["action_index"] = 1
    elif mutation == "finish_without_start":
        events.pop(5)
    elif mutation == "duplicate_action":
        events.insert(6, copy.deepcopy(events[5]))
    elif mutation == "duplicate_terminal":
        events.append(copy.deepcopy(events[-1]))
    elif mutation in {"action_after_terminal", "request_after_terminal"}:
        events.append(
            copy.deepcopy(events[5 if mutation == "action_after_terminal" else 1])
        )
        events[-1]["timestamp_ns"] = 150_000_000
    elif mutation == "cross_episode_chunk":
        manifest["episodes"].append({"episode_id": "ep-2", "seed": 8})
        events.append({**events[5], "episode_id": "ep-2", "timestamp_ns": 150_000_000})
    elif mutation == "release_mode_mismatch":
        events[4]["delay_mode"] = "zero"
    elif mutation == "duplicate_request":
        events.insert(2, copy.deepcopy(events[1]))
    elif mutation == "zero_success_mean":
        manifest["metrics"] = {
            "trials": 1,
            "successes": 0,
            "successful_task_time_mean_ns": 12,
            "successful_chunk_count_mean": None,
        }
    elif mutation == "too_many_successes":
        manifest["metrics"] = {
            "trials": 1,
            "successes": 2,
            "successful_task_time_mean_ns": 12,
            "successful_chunk_count_mean": 1,
        }
    elif mutation == "budget_unknown_clock":
        manifest["budget"]["clock_domain"] = "unknown"
    elif mutation == "missing_terminal":
        events.pop()
    if mutation != "seq_duplicate":
        renumber(events)
    result = run_contract(tmp_path, manifest, events)
    assert result.returncode == 1, result.stderr
    assert expected in result.stderr


def test_clock_domains_do_not_share_an_epoch(tmp_path):
    manifest, events = contract()
    for index, timestamp in [(2, 900_000_000), (3, 980_000_000)]:
        events[index].update(clock_domain="host", timestamp_ns=timestamp)
    result = run_contract(tmp_path, manifest, events)
    assert result.returncode == 0, result.stderr


def test_pending_request_can_complete_and_drop_after_terminal(tmp_path):
    manifest, events = contract()
    terminal = {
        **events[-1],
        "timestamp_ns": 50_000_000,
        "terminal_reason": "timeout",
        "success": False,
    }
    dropped = {
        **events[0],
        "event_type": "result_dropped",
        "timestamp_ns": 80_000_000,
        "request_id": "req-1",
        "chunk_id": "chunk-1",
        "reason": "episode_finished",
    }
    for key in ["observation_id", "sim_tick", "input_ref"]:
        dropped.pop(key)
    events = events[:3] + [terminal, events[3], dropped]
    renumber(events)
    result = run_contract(tmp_path, manifest, events)
    assert result.returncode == 0, result.stderr


def test_zero_success_has_null_success_means(tmp_path):
    manifest, events = contract()
    events[-1].update(success=False, terminal_reason="task_failure")
    manifest["metrics"] = {
        "trials": 1,
        "successes": 0,
        "successful_task_time_mean_ns": None,
        "successful_chunk_count_mean": None,
    }
    result = run_contract(tmp_path, manifest, events)
    assert result.returncode == 0, result.stderr


def test_examples_cli():
    result = subprocess.run(
        [sys.executable, str(CLI), "--examples"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "synthetic" in result.stdout


def test_malformed_json_is_reported_without_traceback(tmp_path):
    manifest, events = contract()
    run_contract(tmp_path, manifest, events)
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("{bad json}\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--trace",
            str(trace_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "line 1" in result.stderr
    assert "Traceback" not in result.stderr


def test_new_episode_requires_previous_terminal_boundary(tmp_path):
    manifest, events = contract()
    manifest["episodes"].append({"episode_id": "ep-2", "seed": 8})
    new_observation = {
        **events[0],
        "episode_id": "ep-2",
        "observation_id": "obs-2",
        "timestamp_ns": 150_000_000,
    }
    events.insert(-1, new_observation)
    events.append({**events[-1], "episode_id": "ep-2"})
    renumber(events)
    result = run_contract(tmp_path, manifest, events)
    assert result.returncode == 1
    assert "terminal boundary" in result.stderr


def test_sequential_episode_reset_does_not_reset_clock(tmp_path):
    manifest, events = contract()
    manifest["episodes"].append({"episode_id": "ep-2", "seed": 8})
    _, second = contract()
    for event in second:
        event["episode_id"] = "ep-2"
        event["timestamp_ns"] += 150_000_000
        for key in ["observation_id", "request_id", "chunk_id"]:
            if key in event:
                event[key] = event[key].replace("-1", "-2")
    events.extend(second)
    renumber(events)
    result = run_contract(tmp_path, manifest, events)
    assert result.returncode == 0, result.stderr


def test_mislabelled_device_watchdog_is_rejected(tmp_path):
    manifest, events = contract()
    manifest["host_watchdog"] = {"clock_domain": "sim", "timeout_ns": 100}
    result = run_contract(tmp_path, manifest, events)
    assert result.returncode == 1
    assert "host watchdog" in result.stderr


def test_release_after_terminal_is_rejected(tmp_path):
    manifest, events = contract()
    terminal = {
        **events[-1],
        "timestamp_ns": 90_000_000,
        "terminal_reason": "timeout",
        "success": False,
    }
    events = events[:4] + [terminal, events[4]]
    renumber(events)
    result = run_contract(tmp_path, manifest, events)
    assert result.returncode == 1
    assert "episode already finished" in result.stderr


def test_overflowing_json_number_is_rejected(tmp_path):
    manifest, events = contract()
    manifest["metrics"] = {
        "trials": 1,
        "successes": 1,
        "successful_task_time_mean_ns": 1.0,
        "successful_chunk_count_mean": 1,
    }
    run_contract(tmp_path, manifest, events)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        manifest_path.read_text().replace(
            '"successful_task_time_mean_ns": 1.0',
            '"successful_task_time_mean_ns": 1e999',
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--manifest",
            str(manifest_path),
            "--trace",
            str(tmp_path / "trace.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "non-finite JSON number" in result.stderr
    assert "Traceback" not in result.stderr
