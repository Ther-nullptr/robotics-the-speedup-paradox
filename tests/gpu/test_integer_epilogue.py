"""Specialized epilogues must preserve the integer mathematical result."""

import os
import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)
import torch


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("tactic", range(8))
def test_specialized_no_bias_epilogue_keeps_quantized_results(bits, use_bias, tactic):
    from robotics_kernels.ampere_ada.integer import IntegerLinear

    original = torch.nn.Linear(
        258, 37, bias=use_bias, device="cuda", dtype=torch.bfloat16
    ).eval()
    candidate = IntegerLinear.from_linear(
        original, bits=bits, tactic=tactic, biasless_epilogue=True
    )
    reference = IntegerLinear.from_linear(original, bits=bits, tactic=tactic)
    x = torch.randn(33, 258, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(candidate(x), reference(x))
