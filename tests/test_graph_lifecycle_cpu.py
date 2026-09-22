"""Failure recovery for graph ownership without loading a CUDA runtime."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import sys
import weakref

import pytest


def test_failed_recapture_keeps_previous_working_graph(monkeypatch):
    class Tensor:
        is_cuda = True
        dtype = "float"
        device = "cuda:1"

        def __init__(self, shape):
            self.shape = shape

        def stride(self):
            return (1,)

        def clone(self):
            return Tensor(self.shape)

        def copy_(self, source):
            pass

        def record_stream(self, stream):
            pass

    class Stream:
        def wait_stream(self, other):
            pass

    class Graph:
        def replay(self):
            pass

    captured = []

    def capture(graph, *, stream=None):
        captured.append(stream)
        return nullcontext()

    cuda = SimpleNamespace(
        device=lambda device: nullcontext(),
        Stream=lambda device: Stream(),
        current_stream=lambda device: Stream(),
        stream=lambda stream: nullcontext(),
        CUDAGraph=Graph,
        graph=capture,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Tensor=Tensor, cuda=cuda))
    path = Path(__file__).parents[1] / "src/robotics_kernels/common/graph.py"
    spec = importlib.util.spec_from_file_location("_graph_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Owner:
        pass

    cache = {}

    def function(x):
        if x.shape == (2,):
            cache.clear()
            raise ValueError("Rejected input")
        cache.setdefault("constant", Owner())
        return x.clone()

    call = module.CudaGraphCall(
        function, retained_tensors=lambda: tuple(cache.values())
    )
    call(Tensor((1,)))
    owner_ref = weakref.ref(cache["constant"])
    with pytest.raises(ValueError, match="Rejected"):
        call(Tensor((2,)))
    assert owner_ref() is not None, (
        "Old graph constants must remain alive after failed recapture"
    )
    assert call(Tensor((1,))).shape == (1,)
    assert all(s is not None for s in captured), (
        "Capture must bind the target-device stream"
    )
