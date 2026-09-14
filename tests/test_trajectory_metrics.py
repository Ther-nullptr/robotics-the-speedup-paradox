"""CPU trajectory checks using independent polynomial derivatives and CSV inputs."""

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "embodied" / "trajectory_metrics.py"


def api():
    assert CLI.is_file(), "trajectory_metrics.py is not implemented"
    spec = importlib.util.spec_from_file_location("trajectory_metrics_tested", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cubic_csv(tmp_path):
    source = tmp_path / "trajectory.csv"
    source.write_text("t,x,y,z\n0,0,0,0\n1,1,2,0\n2,8,16,0\n3,27,54,0\n")
    return source


def run_cli(source, *args):
    return subprocess.run(
        [sys.executable, str(CLI), "--input", str(source), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_cubic_vector_summary_uses_sum_over_xyz(tmp_path):
    result = run_cli(cubic_csv(tmp_path), "--frame", "world")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "series" not in payload
    assert payload["summary"]["jerk_mean_squared_m2_s6"] == pytest.approx(180)
    assert payload["summary"]["jerk_rms_m_s3"] == pytest.approx(math.sqrt(180))
    assert payload["metadata"]["frame"] == "world"
    assert payload["metadata"]["units"]["position"] == "m"
    assert payload["metadata"]["units"]["time"] == "s"
    assert result.stderr == ""


def test_constant_velocity_geometry_and_midpoint_times():
    result = api().analyze_trajectory(
        [0, 0.5, 1, 1.5], [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    )
    summary = result["summary"]
    assert summary["sample_count"] == 4
    assert summary["duration_s"] == 1.5
    assert summary["path_length_m"] == 3
    assert summary["displacement_m"] == 3
    assert summary["mean_speed_m_s"] == 2
    assert summary["peak_speed_m_s"] == 2
    assert summary["uniform_sampling"] is True
    assert summary["dt_s"] == 0.5
    assert summary["acceleration_rms_m_s2"] == 0
    assert summary["jerk_mean_squared_m2_s6"] == 0
    assert result["series"]["velocity"]["time_s"] == [0.25, 0.75, 1.25]
    assert result["series"]["velocity"]["norm"] == [2, 2, 2]
    assert result["series"]["acceleration"]["time_s"] == [0.5, 1]
    assert result["series"]["jerk"]["time_s"] == [0.75]


def test_quadratic_has_constant_acceleration_and_zero_jerk():
    result = api().analyze_trajectory([0, 1, 2, 3], [(t * t, 0, 0) for t in range(4)])
    assert result["series"]["velocity"]["norm"] == [1, 3, 5]
    assert result["series"]["acceleration"]["norm"] == [2, 2]
    assert result["summary"]["acceleration_rms_m_s2"] == 2
    assert result["summary"]["jerk_mean_squared_m2_s6"] == 0


def test_cubic_derivatives_and_stencil_timestamps():
    result = api().analyze_trajectory(
        [10, 11, 12, 13, 14], [(t**3, 0, 0) for t in range(5)]
    )
    series = result["series"]
    assert series["position"]["time_s"] == [10, 11, 12, 13, 14]
    assert series["velocity"]["time_s"] == [10.5, 11.5, 12.5, 13.5]
    assert series["velocity"]["norm"] == [1, 7, 19, 37]
    assert series["acceleration"]["time_s"] == [11, 12, 13]
    assert series["acceleration"]["norm"] == [6, 12, 18]
    assert series["jerk"]["time_s"] == [11.5, 12.5]
    assert series["jerk"]["norm"] == [6, 6]
    assert result["summary"]["jerk_mean_squared_m2_s6"] == 36
    assert result["summary"]["jerk_rms_m_s3"] == 6
    assert result["summary"]["jerk_peak_m_s3"] == 6


def test_vector_jerk_is_180_not_coordinate_averaged_60():
    result = api().analyze_trajectory(
        [0, 1, 2, 3, 4], [(t**3, 2 * t**3, 0) for t in range(5)]
    )
    assert result["summary"]["jerk_mean_squared_m2_s6"] == 180
    assert result["series"]["jerk"]["xyz"] == [(6, 12, 0), (6, 12, 0)]
    assert result["summary"]["jerk_rms_m_s3"] == pytest.approx(math.sqrt(180))


def test_nonuniform_retains_interval_velocities_and_path():
    result = api().analyze_trajectory(
        [0, 1, 3, 4], [(0, 0, 0), (2, 0, 0), (8, 0, 0), (9, 0, 0)]
    )
    summary = result["summary"]
    assert summary["uniform_sampling"] is False
    assert summary["dt_s"] is None
    assert summary["path_length_m"] == 9
    assert summary["mean_speed_m_s"] == 2.25
    assert summary["peak_speed_m_s"] == 3
    assert result["series"]["velocity"]["time_s"] == [0.5, 2, 3.5]
    assert result["series"]["velocity"]["norm"] == [2, 3, 1]
    for name in ("acceleration", "jerk"):
        assert result["series"][name] is None
        assert "nonuniform" in result["missing_reasons"][name]
    for name in (
        "acceleration_rms_m_s2",
        "jerk_mean_squared_m2_s6",
        "jerk_rms_m_s3",
        "jerk_peak_m_s3",
    ):
        assert summary[name] is None
        assert result["missing_reasons"][name]


def test_uniform_tolerance_uses_first_dt_but_velocity_uses_actual_dt():
    result = api().analyze_trajectory(
        [0, 1, 2.000005], [(0, 0, 0), (1, 0, 0), (4, 0, 0)]
    )
    assert result["summary"]["uniform_sampling"] is True
    assert result["summary"]["dt_s"] == 1
    assert result["series"]["acceleration"]["norm"] == [2]
    assert result["series"]["velocity"]["norm"][1] == pytest.approx(3 / 1.000005)
    strict = api().analyze_trajectory(
        [0, 1, 2.000005],
        [(0, 0, 0), (1, 0, 0), (4, 0, 0)],
        uniform_rtol=0,
        uniform_atol_s=0,
    )
    assert strict["summary"]["uniform_sampling"] is False


def test_nonuniform_finite_geometry_does_not_require_higher_derivatives():
    result = api().analyze_trajectory(
        [0, 1, 3, 4], [(0, 0, 0), (5e307, 0, 0), (0, 0, 0), (5e307, 0, 0)]
    )
    assert math.isfinite(result["summary"]["path_length_m"])
    assert result["series"]["acceleration"] is None
    assert result["series"]["jerk"] is None


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_short_trajectories_do_not_pad_missing_derivatives(count):
    result = api().analyze_trajectory(
        list(range(count)), [(t, 0, 0) for t in range(count)]
    )
    assert len(result["series"]["position"]["xyz"]) == count
    for name, required in (("velocity", 2), ("acceleration", 3), ("jerk", 4)):
        if count < required:
            assert result["series"][name] is None
            assert result["missing_reasons"][name]
        else:
            assert len(result["series"][name]["time_s"]) == count - required + 1
    if count == 1:
        assert result["summary"]["duration_s"] == 0
        assert result["summary"]["path_length_m"] == 0
        assert result["summary"]["displacement_m"] == 0
        assert result["summary"]["uniform_sampling"] is None
        assert result["summary"]["dt_s"] is None
        assert result["summary"]["mean_speed_m_s"] is None
        assert result["summary"]["peak_speed_m_s"] is None


@pytest.mark.parametrize(("time_unit", "tf"), [("s", 1), ("ms", 1000)])
@pytest.mark.parametrize(("position_unit", "pf"), [("m", 1), ("cm", 100), ("mm", 1000)])
def test_csv_unit_conversion_and_extra_columns(
    tmp_path, time_unit, tf, position_unit, pf
):
    source = tmp_path / "units.csv"
    source.write_text(
        f"t,x,y,z,episode_id,note\n0,0,0,0,ep1,start\n{tf},{3 * pf},{4 * pf},0,ep1,end\n"
    )
    times, positions = api().load_trajectory(
        source, time_unit=time_unit, position_unit=position_unit
    )
    assert times == [0, 1]
    assert positions == [(0, 0, 0), (3, 4, 0)]
    assert all(isinstance(vector, tuple) for vector in positions)
    assert api().analyze_trajectory(times, positions)["summary"]["path_length_m"] == 5


@pytest.mark.parametrize(
    ("times", "positions", "message"),
    [
        ([], [], "at least one"),
        ([0, 1], [(0, 0, 0)], "length"),
        ([0, 0], [(0, 0, 0), (1, 0, 0)], "increasing"),
        ([1, 0], [(0, 0, 0), (1, 0, 0)], "increasing"),
        ([float("nan")], [(0, 0, 0)], "finite"),
        ([float("inf")], [(0, 0, 0)], "finite"),
        ([0], [(float("inf"), 0, 0)], "finite"),
        ([0], [(0, 0)], "three"),
        ([0], [(0, 0, 0, 0)], "three"),
        ([True], [(0, 0, 0)], "number"),
        ([0], [(True, 0, 0)], "number"),
    ],
)
def test_rejects_invalid_samples(times, positions, message):
    with pytest.raises(ValueError, match=message):
        api().analyze_trajectory(times, positions)


@pytest.mark.parametrize(
    ("field", "value"),
    [("uniform_rtol", -1), ("uniform_atol_s", float("nan")), ("uniform_rtol", True)],
)
def test_rejects_invalid_uniformity_tolerance(field, value):
    with pytest.raises(ValueError, match=field):
        api().analyze_trajectory([0], [(0, 0, 0)], **{field: value})


@pytest.mark.parametrize(
    "contents",
    [
        "t,x,y\n0,0,0\n",
        "t,x,y,z\n",
        "t,x,y,z\n0,nan,0,0\n",
        "t,x,y,z\n0,0,0,0\n0,1,0,0\n",
        "t,x,y,z,episode_id\n0,0,0,0,ep1\n1,1,0,0,ep2\n",
        "t,x,y,z,episode_id\n0,0,0,0,\n",
        "t,x,y,z,episode_id\n0,0,0,0,ep1\n1,1,0,0,\n",
    ],
)
def test_loader_rejects_bad_csv_or_cross_episode_input(tmp_path, contents):
    source = tmp_path / "bad.csv"
    source.write_text(contents)
    with pytest.raises(ValueError):
        api().load_trajectory(source)


@pytest.mark.parametrize("units", [{"time_unit": "minutes"}, {"position_unit": "inch"}])
def test_loader_rejects_unknown_units(tmp_path, units):
    with pytest.raises(ValueError, match="unit"):
        api().load_trajectory(cubic_csv(tmp_path), **units)


def test_cli_optional_series_and_output_file(tmp_path):
    target = tmp_path / "output" / "metrics.json"
    result = run_cli(cubic_csv(tmp_path), "--include-series", "--output", str(target))
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    payload = json.loads(target.read_text())
    assert payload["series"]["jerk"]["time_s"] == [1.5]
    assert payload["series"]["jerk"]["xyz"] == [[6, 12, 0]]


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_cli_never_overwrites_input_even_through_an_alias(tmp_path, alias):
    source = cubic_csv(tmp_path)
    original = source.read_bytes()
    target = source
    if alias != "same":
        target = tmp_path / "alias.json"
        if alias == "symlink":
            target.symlink_to(source)
        else:
            os.link(source, target)
    result = run_cli(source, "--output", str(target))
    assert result.returncode == 2
    assert "input" in result.stderr
    assert source.read_bytes() == original


def test_cli_invalid_data_reports_error_without_json_or_traceback(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("t,x,y,z\n1,0,0,0\n0,1,0,0\n")
    result = run_cli(source)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "increasing" in result.stderr
    assert "Traceback" not in result.stderr
