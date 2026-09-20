"""Graph replay updates values and does not alias previously returned output."""

import os
from pathlib import Path
import pytest

if os.environ.get("ROBOTICS_GPU_TESTS") != "1":
    pytest.skip("Explicit CUDA opt-in required", allow_module_level=True)
import torch


def test_graph_updates_inputs_and_preserves_outputs():
    assert (Path(__file__).parents[2] / "src/robotics_kernels/graph.py").exists(), (
        "Graph helper is not implemented"
    )
    from robotics_kernels.graph import CudaGraphCall

    call = CudaGraphCall(lambda x, y: x + y * 2)
    x = torch.ones(3, 7, device="cuda")
    y = torch.full_like(x, 3)
    first = call(x, y)
    x.fill_(4)
    y.fill_(5)
    second = call(x, y)
    assert torch.equal(first, torch.full_like(x, 7))
    assert torch.equal(second, x + y * 2)
    assert call.captures == 1 and call.replays == 2
    third = call(torch.ones(1, 3, device="cuda"), torch.ones(1, 3, device="cuda"))
    assert torch.equal(third, torch.full_like(third, 3))
    assert call.captures == 2
