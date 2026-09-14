"""Plot semantics: real units, derivative timestamps, and unavailable data."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")


def report():
    return {
        "summary": {"uniform_sampling": True},
        "missing_reasons": {},
        "series": {
            "position": {
                "time_s": [10, 11, 12, 13],
                "xyz": [(1, 10, 100), (2, 11, 101), (3, 12, 102), (4, 13, 103)],
                "norm": [],
            },
            "velocity": {
                "time_s": [10.5, 11.5, 12.5],
                "xyz": [(1, 1, 1)] * 3,
                "norm": [3**0.5] * 3,
            },
            "acceleration": {
                "time_s": [11, 12],
                "xyz": [(0, 0, 0)] * 2,
                "norm": [0, 0],
            },
            "jerk": {"time_s": [11.5], "xyz": [(0, 0, 0)], "norm": [0]},
        },
    }


def plotting():
    path = Path(__file__).resolve().parents[1] / "tools/embodied/plot_trajectory.py"
    spec = importlib.util.spec_from_file_location("embodied_plot_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_3d_uses_equal_meter_scale_and_correct_derivative_times():
    module = plotting()
    fig = module.create_figure([report()], ["example"], frame="world")
    try:
        path, speed, _, jerk = fig.axes
        spans = [
            limits[1] - limits[0]
            for limits in [path.get_xlim(), path.get_ylim(), path.get_zlim()]
        ]
        assert spans == pytest.approx([spans[0]] * 3)
        assert list(speed.lines[0].get_xdata()) == [0.5, 1.5, 2.5]
        assert list(jerk.lines[0].get_xdata()) == [1.5]
        assert "m" in path.get_xlabel()
        assert "world" in fig._suptitle.get_text()
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_unavailable_jerk_is_not_drawn_as_zero():
    module = plotting()
    data = report()
    data["series"]["jerk"] = None
    data["missing_reasons"]["jerk"] = "nonuniform_sampling"
    fig = module.create_figure([data], ["irregular"], projection="xy")
    try:
        jerk = fig.axes[-1]
        assert len(jerk.lines) == 0
        assert any("unavailable" in text.get_text().lower() for text in jerk.texts)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_label_count_must_match_trajectories():
    with pytest.raises(ValueError, match="label"):
        plotting().create_figure([report()], [])
