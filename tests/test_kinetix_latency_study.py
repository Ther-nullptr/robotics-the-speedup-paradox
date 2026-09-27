"""Coverage and provenance boundaries for hardware-delay studies."""

import importlib.util
from pathlib import Path

import pytest


def load_study():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/dynamic/kinetix/latency_study.py"
    )
    spec = importlib.util.spec_from_file_location("kinetix_latency_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_preserves_logical_conditions_without_inflating_samples():
    study = load_study()
    profiles = study.read_profiles(study.DEFAULT_ARCHIVE)
    matrix = study.make_matrix(["catcher_v3"], profiles)
    assert len(matrix["conditions"]) == 57
    assert len(matrix["executions"]) == 31
    assert len(matrix["jobs"]) == 6
    assert sum(len(job["execution_ids"]) for job in matrix["jobs"]) == 31
    coarse = [
        row
        for row in matrix["conditions"]
        if row["role"] == "validation" and row["mode"] == "coarse"
    ]
    fast = [row for row in coarse if row["hardware"] in {"local_ada", "rtx3090"}]
    assert len(fast) == 10
    assert all(row["source"]["kind"] == "baseline" for row in fast)
    assert all(row["calibration_overlap"] for row in fast)
    fine = [
        row
        for row in matrix["conditions"]
        if row["role"] == "validation" and row["mode"] == "fine"
    ]
    assert len(fine) == 20
    assert not any(row["calibration_overlap"] for row in fine)
    ids = {row["id"] for row in matrix["executions"]}
    assert all(
        row["source"]["id"] in ids
        for row in matrix["conditions"]
        if row["source"]["kind"] == "execution"
    )


def test_duplicate_or_missing_seed_is_not_accepted_as_complete_coverage():
    study = load_study()
    rows = [{"flow_steps": 1, "env_seed": seed} for seed in range(3)]
    study.check_coverage(rows, [1], start_seed=0, episodes=3)
    with pytest.raises(ValueError, match="coverage|Duplicate"):
        study.check_coverage(rows + rows[:1], [1], start_seed=0, episodes=3)
    with pytest.raises(ValueError, match="coverage"):
        study.check_coverage(rows[:-1], [1], start_seed=0, episodes=3)


def test_profile_rejects_excluded_or_ambiguous_hardware(tmp_path):
    study = load_study()
    original = study.DEFAULT_ARCHIVE.read_text()
    path = tmp_path / "profiles.csv"
    path.write_text(original + original.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        study.read_profiles(path)
    path.write_text(original.replace("agx30", "agx50"))
    with pytest.raises(ValueError, match="profile|hardware"):
        study.read_profiles(path)
