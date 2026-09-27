"""Run native DynamicVLA + DOM from a checkout without external source imports."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from robotics_bench.dynamicvla_dom.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
