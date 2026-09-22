"""CPU checks for the standalone LingBot evaluator report scorer."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "benchmarks/static/lingbot_robotwin/score_reports.py"
SCORE_EXAMPLE = ROOT / "benchmarks/static/lingbot_robotwin/scoring.synthetic.json"


def load_scoring():
    spec = importlib.util.spec_from_file_location("lingbot_scoring", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scoring = load_scoring()


def score_example():
    groups, baseline, synthetic = scoring.load_manifest(SCORE_EXAMPLE)
    episodes = [
        scoring.score_episode(report, config, digest)
        for config, reports in groups
        for report, digest in reports
    ]
    return groups, episodes, scoring.summarize(episodes, baseline), synthetic


def test_score_example_exposes_chunk_task_paradox():
    _, episodes, rows, synthetic = score_example()
    base, p4 = rows
    assert synthetic is True
    assert len(episodes) == 4
    assert base["success_rate"] == p4["success_rate"] == 0.5
    assert base["mean_success_chunks"] == 1
    assert p4["mean_success_chunks"] == 2
    assert p4["speedup_endpoint_compute_per_chunk"] > 1
    assert p4["speedup_task_endpoint_compute"] < 1
    assert base["mean_success_stability_score"] == pytest.approx(69.5)


def test_score_cli_writes_machine_and_human_outputs_without_overwrite(tmp_path):
    output = tmp_path / "score"
    command = [
        sys.executable,
        str(SCORER),
        "--input",
        str(SCORE_EXAMPLE),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert {path.name for path in output.iterdir()} == {
        "episodes.csv",
        "summary.csv",
        "summary.json",
        "summary.md",
    }
    payload = json.loads((output / "summary.json").read_text())
    assert payload["format"] == "lingbot-evaluator-score-summary-v1"
    assert payload["synthetic"] is True
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode == 2


def test_score_rejects_duplicate_episode_and_inconsistent_stability():
    groups, episodes, _, _ = score_example()
    with pytest.raises(ValueError, match="duplicate episode"):
        scoring.summarize([*episodes, episodes[0]], "base")

    bad_report = deepcopy(groups[0][1][0][0])
    bad_report["action_stability"]["action_stability_score"] = 0.0
    with pytest.raises(ValueError, match="disagrees with its components"):
        scoring.score_episode(bad_report, "base", "fixture")


def test_score_zero_success_is_explicitly_missing():
    groups, _, _, _ = score_example()
    episodes = []
    for config, reports in groups:
        for report, digest in reports:
            report = deepcopy(report)
            if config == "p4":
                report["success"] = False
            episodes.append(scoring.score_episode(report, config, digest))
    p4 = scoring.summarize(episodes, "base")[1]
    assert p4["success_rate"] == 0
    assert p4["mean_success_chunks"] is None
    assert p4["speedup_task_endpoint_compute"] is None
    assert p4["missing_reasons"] == ["candidate or baseline has no successful episodes"]


def test_optional_control_timing_is_not_inferred_from_endpoint_timing():
    groups, _, _, _ = score_example()
    report = deepcopy(groups[0][1][0][0])
    action = report["denoise"]["action_infer_reports"][0]
    del action["execute_actions_equivalent_s"]
    del action["encode_obs_s"]
    episode = scoring.score_episode(report, "base", "fixture")
    assert episode["task_endpoint_compute_s"] == pytest.approx(1.0)
    assert episode["task_endpoint_plus_control_s"] is None
    assert episode["server_compute_per_chunk_s"] == pytest.approx(0.65)


def test_partial_control_timing_within_episode_is_rejected():
    groups, _, _, _ = score_example()
    report = deepcopy(groups[1][1][0][0])
    del report["denoise"]["action_infer_reports"][1]["execute_actions_equivalent_s"]
    with pytest.raises(ValueError, match="every chunk or none"):
        scoring.score_episode(report, "p4", "fixture")
