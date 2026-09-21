"""Use measured GPU events only, without adding overlapping CPU ranges."""

from pathlib import Path
import importlib
import pytest


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "cutlass::gemm::DefaultGemmWithVisitor<signed char",
            "Integer matrix multiplication",
        ),
        (
            "cutlass::gemm::DefaultGemmWithVisitor<cutlass::integer_subbyte<4, true>",
            "Integer matrix multiplication",
        ),
        ("ampere_bf16_gemm", "Floating matrix multiplication"),
        ("cublasLt::splitKreduce_kernel", "Matrix reduction and epilogue"),
        ("elementwise_kernel direct_copy_kernel_cuda", "Copies and dtype conversions"),
        ("vectorized_layer_norm_kernel", "Normalization and reductions"),
        ("CatArrayBatchedCopy", "Copies and dtype conversions"),
        ("_vae_norm_affine", "VAE pointwise fusion"),
        ("fmha_cutlassF_bf16_aligned", "Attention and softmax"),
        (
            "distribution_elementwise normal_and_transform",
            "Other elementwise arithmetic",
        ),
    ],
)
def test_separates_actual_integer_compute_and_surrounding_work(name, expected):
    api = importlib.import_module("robotics_bench.profiling.breakdown")
    assert api.category({"cat": "kernel", "name": name}) == expected


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


def test_large_sweeps_paginate_without_losing_candidates_or_anchor():
    api = importlib.import_module("robotics_bench.profiling.breakdown")
    assert hasattr(api, "pages"), "Breakdown pagination is missing"
    document = {
        "title": "test",
        "profiles": [{"name": str(i)} for i in range(8)],
        "sources": [],
    }
    pages = api.pages(document)
    assert [len(p["profiles"]) for p in pages] == [4, 4, 2]
    assert all(p["profiles"][0]["name"] == "0" for p in pages)
    assert [r["name"] for p in pages for r in p["profiles"][1:]] == [
        str(i) for i in range(1, 8)
    ]
    assert len(document["profiles"]) == 8
