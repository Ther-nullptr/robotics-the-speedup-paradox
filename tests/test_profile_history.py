"""History exports explicit paired measurements without multiplying rounds."""

from pathlib import Path
import importlib


def test_low_precision_quality_and_matched_baseline_are_separate():
    assert (
        Path(__file__).parents[1] / "src/robotics_bench/profiling/history.py"
    ).exists(), "History exporter is missing"
    api = importlib.import_module("robotics_bench.profiling.history")
    records = [
        {
            "id": "base",
            "precision": "bf16",
            "switches": [],
            "samples_ms": [100, 102],
            "max_abs": 0.0,
            "exact": True,
        },
        {
            "id": "fused",
            "precision": "bf16",
            "switches": ["norm"],
            "samples_ms": [80, 82],
            "max_abs": 0.0,
            "exact": True,
        },
        {
            "id": "int8",
            "precision": "int8",
            "switches": ["norm"],
            "samples_ms": [60, 62],
            "max_abs": 0.01,
            "exact": False,
        },
    ]
    data = api.build_history(
        records,
        protocol={
            "id": "case",
            "label": "Case",
            "workload": {"input": "fixed"},
            "environment": {"gpu": "test"},
            "metric": {"name": "Service", "unit": "ms", "statistic": "median"},
        },
        cohort="run",
        source="measurements.json",
        revision="abc",
        current="fused",
    )
    assert data["current_run_id"] == "fused"
    assert data["runs"][2]["quality"]["status"] == "unmeasured"
    assert any(
        c["kind"] == "matched_precision"
        and c["baseline_run_id"] == "fused"
        and c["candidate_run_id"] == "int8"
        for c in data["comparisons"]
    )
    assert len([c for c in data["comparisons"] if c["kind"] == "cumulative"]) == 2
