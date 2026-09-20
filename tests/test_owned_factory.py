"""A freshly loaded policy must enter eval mode before optimization installation."""

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import sys


def test_factory_sets_eval_before_optimization(monkeypatch):
    from robotics_bench.optimizations.entry import owned_pi05_factory
    from robotics_bench.optimizations.config import OptimizationConfig

    class Policy:
        def __init__(self):
            self.model = SimpleNamespace(training=True)

        def eval(self):
            self.model.training = False
            return self

    owned = ModuleType("robotics_bench.models.pi05.modeling_pi05")
    owned.PI05Policy = Policy
    optimizer = ModuleType("robotics_bench.optimizations.pi05")

    @contextmanager
    def optimize(model, config):
        assert not model.training, "Optimizer received a training-mode policy"
        yield {"installed": True}

    optimizer.optimize_pi05 = optimize
    monkeypatch.setitem(sys.modules, owned.__name__, owned)
    monkeypatch.setitem(sys.modules, optimizer.__name__, optimizer)
    namespace = {"get_policy_class": lambda name: Policy}
    exec('def make_policy():\n    return get_policy_class("pi05")()\n', namespace)
    original = namespace["make_policy"]
    evaluator = SimpleNamespace(make_lerobot_policy=original)
    with owned_pi05_factory(evaluator, OptimizationConfig()) as reports:
        policy = evaluator.make_lerobot_policy()
        assert not policy.model.training
        assert reports == [{"installed": True}]
    assert evaluator.make_lerobot_policy is original
