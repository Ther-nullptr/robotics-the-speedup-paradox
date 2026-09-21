"""Compatibility entry for the shared, installable CPU episode statistics API."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robotics_bench.statistics import summarize_episodes as summarize_episodes  # noqa: E402
