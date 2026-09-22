"""Replay cases preserve recorded convolution semantics and input layouts."""

import importlib.util
from pathlib import Path


def test_extract_convolution_cases_retains_strides_and_unsupported_coverage():
    path = Path(__file__).parents[1] / "benchmarks/inference/bench_convolution.py"
    assert path.is_file(), "Convolution replay entry is missing"
    spec = importlib.util.spec_from_file_location("conv_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def event(channels):
        return {
            "name": "aten::convolution",
            "args": {
                "Input Dims": [[1, channels, 3, 10, 10], [16, channels, 3, 3, 3], [16]],
                "Input Strides": [
                    [300 * channels, 1, 100 * channels, 10 * channels, channels]
                ],
                "Input type": ["c10::BFloat16", "c10::BFloat16", "c10::BFloat16"],
                "Concrete Inputs": [
                    "",
                    "",
                    "",
                    "[1, 1, 1]",
                    "[0, 0, 0]",
                    "[1, 1, 1]",
                    "False",
                    "[0, 0, 0]",
                    "1",
                ],
            },
        }

    cases = module.extract_cases({"traceEvents": [event(16), event(16), event(3)]})
    assert len(cases) == 2
    supported = next(c for c in cases if c["input_shape"][1] == 16)
    assert supported["calls"] == 2
    assert supported["input_strides"] == [4800, 1, 1600, 160, 16]
    assert supported["padding"] == [0, 0, 0]
    assert supported["supported"]
    unsupported = next(c for c in cases if c["input_shape"][1] == 3)
    assert not unsupported["supported"]
    assert "multiple of 8" in unsupported["reason"]
