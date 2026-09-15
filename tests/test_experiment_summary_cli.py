"""The summary CLI reads existing ledgers and never launches an evaluator."""

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ENTRY = Path(__file__).resolve().parents[1] / "tools/summarize_experiment.py"


def run(*args):
    return subprocess.run(
        [sys.executable, str(ENTRY), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "run-a"
    root.mkdir()
    records = [
        {
            "task": "A",
            "init_state_id": 0,
            "env_seed": 42,
            "success": True,
            "primitive_steps": 10,
        },
        {
            "task": "A",
            "init_state_id": 1,
            "env_seed": 42,
            "success": False,
            "primitive_steps": 4,
        },
    ]
    (root / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (root / "case-manifest.json").write_text(
        json.dumps({"status": "completed", "identity": {"case": {"episodes": 2}}})
    )
    (root / "coverage.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "expected_episodes": 2,
                "completed_episodes": 2,
                "successes": 1,
            }
        )
    )
    return root


def test_summary_distinguishes_actual_success_and_penalized_counts(experiment):
    result = run("--input", experiment, "--max-steps", 100, "--format", "json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    record = report["runs"][0]
    assert record["completion"] == "complete"
    overall = record["statistics"]["overall"]
    assert overall["all_episodes"]["total_control_steps"] == 14
    assert overall["all_episodes"]["mean_control_steps"] == 7
    assert overall["successful_episodes"]["mean_control_steps"] == 10
    assert overall["failure_penalized"]["mean_control_steps"] == 55


def test_missing_budget_is_not_inferred_from_longest_episode(experiment):
    result = run("--input", experiment, "--format", "json")
    assert result.returncode == 0, result.stderr
    penalty = json.loads(result.stdout)["runs"][0]["statistics"]["overall"][
        "failure_penalized"
    ]
    assert penalty["status"] == "missing_failure_budget"
    assert penalty["mean_control_steps"] is None


def test_recorded_runtime_budget_is_used_without_a_cli_argument(experiment):
    path = experiment / "coverage.json"
    coverage = json.loads(path.read_text())
    coverage["max_primitive_steps"] = 80
    path.write_text(json.dumps(coverage))
    result = run("--input", experiment, "--format", "json")
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)["runs"][0]
    assert record["budgets"]["max_steps"] == 80
    assert record["budgets"]["source"] == "coverage.json"
    assert (
        record["statistics"]["overall"]["failure_penalized"]["mean_control_steps"] == 45
    )


def test_runtime_can_write_the_same_summary_without_model_imports(experiment, tmp_path):
    spec = importlib.util.spec_from_file_location("runtime_summary_test", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        json.loads(line)
        for line in (experiment / "episodes.jsonl").read_text().splitlines()
    ]
    output = tmp_path / "auto-run"
    output.mkdir()
    report = module.write_run_summary(output, rows, max_steps=100)
    written = json.loads((output / "episode-summary.json").read_text())
    assert written == report
    assert "Success only" in (output / "episode-summary.md").read_text()
    standalone = run("--input", experiment, "--max-steps", 100, "--format", "json")
    assert standalone.returncode == 0, standalone.stderr
    assert (
        report["runs"][0]["statistics"]
        == json.loads(standalone.stdout)["runs"][0]["statistics"]
    )


def test_standalone_jsonl_is_supported_with_unverified_completion(experiment):
    (experiment / "case-manifest.json").unlink()
    (experiment / "coverage.json").unlink()
    result = run("--input", experiment / "episodes.jsonl", "--format", "json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["runs"][0]["completion"] == "unverified"


def test_unfinished_run_requires_explicit_partial_flag(experiment):
    (experiment / "case-manifest.json").write_text('{"status":"running"}')
    rejected = run("--input", experiment)
    assert rejected.returncode != 0
    assert "allow-partial" in rejected.stderr
    accepted = run("--input", experiment, "--allow-partial", "--format", "json")
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(accepted.stdout)["runs"][0]["completion"] == "partial"


def test_completed_metadata_must_agree_with_actual_rows(experiment):
    (experiment / "coverage.json").write_text(
        '{"status":"passed","expected_episodes":3,"completed_episodes":3,"successes":1}'
    )
    result = run("--input", experiment, "--allow-partial")
    assert result.returncode != 0
    assert "count" in result.stderr.lower()


def test_default_markdown_and_csv_formats(experiment):
    markdown = run("--input", experiment, "--max-steps", 100)
    assert markdown.returncode == 0, markdown.stderr
    assert (
        "All episodes (observed)" in markdown.stdout
        and "Success only" in markdown.stdout
    )
    assert markdown.stdout.isascii()
    csv = run("--input", experiment, "--max-steps", 100, "--format", "csv")
    assert csv.returncode == 0, csv.stderr
    assert "all_total_control_steps" in csv.stdout
    assert "failure_penalized_mean_control_steps" in csv.stdout


def test_output_is_created_exclusively_and_inputs_are_preserved(experiment, tmp_path):
    output = tmp_path / "report.json"
    result = run("--input", experiment, "--max-steps", 100, "--output", output)
    assert result.returncode == 0, result.stderr
    original = output.read_text()
    assert json.loads(original)["runs"][0]["name"] == "run-a"
    rejected = run("--input", experiment, "--output", output)
    assert rejected.returncode != 0
    assert output.read_text() == original
    assert (
        run("--input", experiment, "--output", experiment / "episodes.jsonl").returncode
        != 0
    )


def test_multiple_runs_are_reported_separately(experiment, tmp_path):
    other = tmp_path / "run-b"
    other.mkdir()
    (other / "episodes.jsonl").write_text((experiment / "episodes.jsonl").read_text())
    result = run("--input", experiment, other, "--max-steps", 100, "--format", "json")
    assert result.returncode == 0, result.stderr
    records = json.loads(result.stdout)["runs"]
    assert [record["name"] for record in records] == ["run-a", "run-b"]
    assert all(record["statistics"]["overall"]["episodes"] == 2 for record in records)


def test_line_number_is_reported_for_invalid_json(experiment):
    with (experiment / "episodes.jsonl").open("a") as stream:
        stream.write("{broken}\n")
    result = run("--input", experiment)
    assert result.returncode != 0
    assert "episodes.jsonl:3" in result.stderr


def test_per_task_budget_override_is_used(experiment):
    result = run(
        "--input",
        experiment,
        "--max-steps",
        100,
        "--task-max-steps",
        "A=40",
        "--format",
        "json",
    )
    assert result.returncode == 0, result.stderr
    stats = json.loads(result.stdout)["runs"][0]["statistics"]
    assert stats["overall"]["failure_penalized"]["mean_control_steps"] == 25
    assert run("--input", experiment, "--task-max-steps", "typo=40").returncode != 0


def test_budget_provenance_distinguishes_recorded_budget_and_cli_override(experiment):
    coverage_path = experiment / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    coverage["max_primitive_steps"] = 80
    coverage_path.write_text(json.dumps(coverage))
    result = run("--input", experiment, "--task-max-steps", "A=40", "--format", "json")
    assert result.returncode == 0, result.stderr
    budgets = json.loads(result.stdout)["runs"][0]["budgets"]
    assert budgets["source"] == "mixed"
    assert budgets["global_source"] == "coverage.json"
    assert budgets["task_override_source"] == "cli"


def test_task_budget_metadata_only_lists_overrides_used_by_that_run(
    experiment, tmp_path
):
    other = tmp_path / "run-b"
    other.mkdir()
    (other / "episodes.jsonl").write_text(
        json.dumps(
            {"task": "B", "init_state_id": 0, "success": False, "primitive_steps": 4}
        )
        + "\n"
    )
    result = run(
        "--input",
        experiment,
        other,
        "--task-max-steps",
        "A=40",
        "--task-max-steps",
        "B=50",
        "--format",
        "json",
    )
    assert result.returncode == 0, result.stderr
    records = json.loads(result.stdout)["runs"]
    assert records[0]["budgets"]["task_max_steps"] == {"A": 40}
    assert records[1]["budgets"]["task_max_steps"] == {"B": 50}
    assert all(record["budgets"]["source"] == "cli" for record in records)
