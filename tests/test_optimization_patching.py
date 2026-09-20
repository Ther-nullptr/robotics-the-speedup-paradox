"""Restore only an instance's changed behavior, including on error."""

from robotics_bench.optimizations.patching import Patches


def value():
    return "original"


class Model:
    def forward(self):
        return value()


def test_instance_patch_does_not_mutate_module_globals_or_other_instances():
    one, two = Model(), Model()
    patches = Patches()
    patches.globals(one, "forward", {"value": lambda: "optimized"})
    assert one.forward() == "optimized"
    assert two.forward() == "original"
    assert value() == "original"
    patches.restore()
    assert "forward" not in vars(one)
    assert one.forward() == "original"


def test_patch_restores_attributes_held_in_a_module_registry():
    class Registry:
        def __init__(self):
            object.__setattr__(self, "_modules", {"layer": "original"})

        def __getattr__(self, name):
            return self._modules[name]

        def __setattr__(self, name, value):
            self._modules[name] = value

        def __delattr__(self, name):
            del self._modules[name]

    module = Registry()
    patches = Patches()
    patches.set(module, "layer", "optimized")
    patches.restore()
    assert module.layer == "original"
