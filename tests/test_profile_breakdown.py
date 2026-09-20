"""Use measured GPU events only, without adding overlapping CPU ranges."""

from pathlib import Path
import importlib


def test_breakdown_excludes_cpu_ranges_and_retains_unknown_kernels():
    assert (
        Path(__file__).parents[1] / "src/robotics_bench/profiling/breakdown.py"
    ).exists()
    api = importlib.import_module("robotics_bench.profiling.breakdown")
    trace = {
        "traceEvents": [
            {"cat": "cpu_op", "name": "aten::mm", "dur": 1000},
            {"cat": "kernel", "name": "cutlass_gemm", "dur": 20},
            {"cat": "kernel", "name": "unknown_kernel", "dur": 5},
            {"cat": "gpu_memcpy", "name": "Memcpy HtoD", "dur": 3},
        ]
    }
    result = api.summarize(trace, "candidate")
    assert result["total"] == 0.028
    assert {x["name"]: x["value"] for x in result["components"]} == {
        "Matrix multiplication": 0.020,
        "Unclassified GPU work": 0.005,
        "GPU transfers": 0.003,
    }
