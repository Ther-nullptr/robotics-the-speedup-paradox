"""Independent hand calculations and invalid inputs for the CPU speedup CLI."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "compare_speedups.py"


def case():
    return {
        "format_version": "1.0",
        "synthetic": True,
        "case_id": "hand-calculated",
        "timing_scope": "policy_inference",
        "clock_domain": "simulation",
        "n_actions": 10,
        "action_time_ms": 20,
        "baseline_id": "baseline",
        "runs": [
            {
                "id": name,
                "inference_time_ms": latency,
                "overlap_actions": overlap,
                "trials": 100,
                "successes": 80,
                "successful_chunk_count_mean": chunks,
            }
            for name, latency, overlap, chunks in [
                ("baseline", 100, 0, 10),
                ("quantization", 50, 0, 12),
                ("async", 100, 3, 12),
                ("combined", 50, 3, 16),
            ]
        ],
    }


def invoke(tmp_path, payload, *flags):
    source = tmp_path / "case.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(CLI), "--input", str(source), *flags],
        capture_output=True,
        text=True,
        check=False,
    )


def parsed(tmp_path, payload):
    result = invoke(tmp_path, payload)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return json.loads(result.stdout)


def test_four_hand_calculated_cycles_and_task_slowdown(tmp_path):
    output = parsed(tmp_path, case())
    assert output["estimate_model"] == "paper_steady_state"
    assert output["synthetic"] is True
    rows = output["runs"]
    assert [r["paper_steady_state"]["cycle_time_ms"] for r in rows] == [
        300,
        250,
        240,
        200,
    ]
    assert [r["paper_steady_state"]["successful_task_time_mean_ms"] for r in rows] == [
        3000,
        3000,
        2880,
        3200,
    ]
    combined = rows[3]
    assert combined["latency_ratio_alpha"] == 0.5
    assert combined["speedup_inference"] == 2
    assert combined["speedup_chunk_estimated"] == 1.5
    assert combined["speedup_task_estimated"] == 0.9375
    assert combined["speedup_task_success"] is None
    assert combined["missing_reasons"]["speedup_task_success"]


def test_independent_speedups_cannot_generally_be_multiplied(tmp_path):
    payload = case()
    payload["runs"][1]["inference_time_ms"] = 60
    payload["runs"][3]["inference_time_ms"] = 60
    rows = parsed(tmp_path, payload)["runs"]
    independent_product = (
        rows[1]["speedup_chunk_estimated"] * rows[2]["speedup_chunk_estimated"]
    )
    assert independent_product == pytest.approx(1.4423076923076923)
    assert rows[3]["speedup_chunk_estimated"] == 1.5
    assert independent_product != rows[3]["speedup_chunk_estimated"]


def test_provided_task_summary_is_separate_from_cycle_estimate(tmp_path):
    payload = case()
    payload["runs"][0]["successful_task_time_mean_ms"] = 4500
    payload["runs"][1]["successful_task_time_mean_ms"] = 3750
    payload["runs"][1]["successes"] = 75
    row = parsed(tmp_path, payload)["runs"][1]
    assert row["speedup_task_success"] == 1.2
    assert row["speedup_task_estimated"] == 1
    assert row["provided_summary"]["successful_task_time_mean_ms"] == 3750
    assert row["success_rate"] == 0.75
    assert row["success_rate_delta_pp"] == pytest.approx(-5)


def test_missing_chunk_mean_does_not_invent_task_estimate(tmp_path):
    payload = case()
    del payload["runs"][1]["successful_chunk_count_mean"]
    row = parsed(tmp_path, payload)["runs"][1]
    assert row["paper_steady_state"]["successful_task_time_mean_ms"] is None
    assert row["speedup_task_estimated"] is None
    assert row["missing_reasons"]["speedup_task_estimated"]
    assert row["speedup_chunk_estimated"] == 1.2


def test_missing_baseline_mean_keeps_individual_estimate(tmp_path):
    payload = case()
    payload["runs"][0]["successful_chunk_count_mean"] = None
    row = parsed(tmp_path, payload)["runs"][1]
    assert row["paper_steady_state"]["successful_task_time_mean_ms"] == 3000
    assert row["speedup_task_estimated"] is None
    assert "baseline" in row["missing_reasons"]["speedup_task_estimated"]


@pytest.mark.parametrize("run_index", [0, 1])
def test_no_success_on_either_side_means_no_task_ratio(tmp_path, run_index):
    payload = case()
    payload["runs"][run_index]["successes"] = 0
    payload["runs"][run_index]["successful_chunk_count_mean"] = None
    payload["runs"][1 - run_index]["successful_task_time_mean_ms"] = 1000
    rows = parsed(tmp_path, payload)["runs"]
    assert rows[run_index]["success_rate"] == 0
    assert rows[1]["speedup_task_estimated"] is None
    assert rows[1]["speedup_task_success"] is None
    assert rows[1]["missing_reasons"]["speedup_task_success"]


def test_full_overlap_saturates_at_action_execution_time(tmp_path):
    payload = case()
    payload["runs"][1]["overlap_actions"] = 10
    row = parsed(tmp_path, payload)["runs"][1]
    assert row["paper_steady_state"]["cycle_time_ms"] == 200
    assert row["speedup_chunk_estimated"] == 1.5


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("n_actions",), True, "n_actions"),
        (("n_actions",), 0, "n_actions"),
        (("n_actions",), 1.5, "n_actions"),
        (("action_time_ms",), False, "action_time_ms"),
        (("action_time_ms",), 0, "action_time_ms"),
        (("action_time_ms",), -1, "action_time_ms"),
        (("synthetic",), 1, "synthetic"),
        (("format_version",), "2.0", "format_version"),
        (("case_id",), " ", "case_id"),
        (("timing_scope",), None, "timing_scope"),
        (("clock_domain",), 0, "clock_domain"),
        (("baseline_id",), "absent", "baseline"),
        (("runs",), [], "runs"),
        (("runs",), {}, "runs"),
        (("runs", 0, "overlap_actions"), 1, "baseline"),
        (("runs", 1, "id"), "baseline", "duplicate"),
        (("runs", 1, "id"), "", "id"),
        (("runs", 1, "overlap_actions"), -1, "overlap_actions"),
        (("runs", 1, "overlap_actions"), 11, "overlap_actions"),
        (("runs", 1, "overlap_actions"), True, "overlap_actions"),
        (("runs", 1, "overlap_actions"), 1.5, "overlap_actions"),
        (("runs", 1, "inference_time_ms"), 0, "inference_time_ms"),
        (("runs", 1, "inference_time_ms"), -1, "inference_time_ms"),
        (("runs", 1, "inference_time_ms"), True, "inference_time_ms"),
        (("runs", 1, "trials"), 0, "trials"),
        (("runs", 1, "trials"), False, "trials"),
        (("runs", 1, "successes"), -1, "successes"),
        (("runs", 1, "successes"), 101, "successes"),
        (("runs", 1, "successes"), True, "successes"),
        (("runs", 1, "successful_chunk_count_mean"), 0, "successful_chunk"),
        (("runs", 1, "successful_chunk_count_mean"), False, "successful_chunk"),
        (("runs", 1, "successful_task_time_mean_ms"), -1, "successful_task"),
        (("runs", 1, "successful_task_time_mean_ms"), 0, "successful_task"),
        (("runs", 1, "successful_task_time_mean_ms"), True, "successful_task"),
    ],
)
def test_rejects_invalid_fields(tmp_path, path, value, message):
    payload = case()
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    result = invoke(tmp_path, payload)
    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_json_numbers(tmp_path, value):
    payload = case()
    payload["runs"][1]["inference_time_ms"] = value
    result = invoke(tmp_path, payload)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "finite" in result.stderr


@pytest.mark.parametrize(
    "field", ["successful_chunk_count_mean", "successful_task_time_mean_ms"]
)
def test_rejects_success_mean_when_no_success_exists(tmp_path, field):
    payload = case()
    row = payload["runs"][1]
    row["successes"] = 0
    row["successful_chunk_count_mean"] = None
    row[field] = 12
    result = invoke(tmp_path, payload)
    assert result.returncode == 2
    assert "zero successes" in result.stderr


def test_rejects_overflow_in_derived_cycle(tmp_path):
    payload = case()
    payload["action_time_ms"] = 1e308
    result = invoke(tmp_path, payload)
    assert result.returncode == 2
    assert "finite" in result.stderr
    assert result.stdout == ""


def test_rejects_missing_required_field(tmp_path):
    payload = case()
    del payload["runs"][1]["trials"]
    result = invoke(tmp_path, payload)
    assert result.returncode == 2
    assert "trials" in result.stderr


def test_accepts_baseline_in_any_row_and_does_not_reorder(tmp_path):
    payload = case()
    payload["runs"] = payload["runs"][1:] + payload["runs"][:1]
    output = parsed(tmp_path, payload)
    assert output["baseline_id"] == "baseline"
    assert [row["id"] for row in output["runs"]] == [
        "quantization",
        "async",
        "combined",
        "baseline",
    ]
    assert output["runs"][3]["speedup_task_estimated"] == 1


def test_markdown_reports_four_speedups_and_synthetic_status(tmp_path):
    result = invoke(tmp_path, case(), "--markdown")
    assert result.returncode == 0, result.stderr
    for label in [
        "synthetic=true",
        "timing_scope=policy_inference",
        "clock_domain=simulation",
        "n_actions=10",
        "action_time_ms=20",
        "speedup_inference",
        "speedup_chunk_estimated",
        "speedup_task_estimated",
        "speedup_task_success",
        "0.9375",
        "null",
    ]:
        assert label in result.stdout


def test_cli_reports_malformed_json_without_traceback(tmp_path):
    source = tmp_path / "broken.json"
    source.write_text('{"runs":', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CLI), "--input", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_example_is_marked_synthetic_and_runs_without_other_assets():
    example = ROOT / "examples" / "speedups" / "synthetic.json"
    result = subprocess.run(
        [sys.executable, str(CLI), "--input", str(example)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["synthetic"] is True
    assert [row["paper_steady_state"]["cycle_time_ms"] for row in output["runs"]] == [
        300,
        250,
        240,
        200,
    ]
