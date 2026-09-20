"""Instance-local patches with deterministic restoration."""

from types import FunctionType, MethodType


class Patches:
    def __init__(self):
        self._undo = []

    def set(self, obj, name, value):
        registered = any(
            name in vars(obj).get(key, {})
            for key in ("_modules", "_parameters", "_buffers")
        )
        existed = name in vars(obj) or registered
        previous = getattr(obj, name) if existed else None
        self._undo.append((obj, name, existed, previous))
        setattr(obj, name, value)

    def bind(self, obj, name, function):
        self.set(obj, name, MethodType(function, obj))

    def globals(self, obj, name, replacements):
        original = getattr(obj, name).__func__
        namespace = dict(original.__globals__)
        for key in replacements:
            if key not in namespace:
                raise ValueError(f"Unsupported callable: missing global {key}")
        namespace.update(replacements)
        function = FunctionType(
            original.__code__,
            namespace,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        function.__kwdefaults__ = original.__kwdefaults__
        self.bind(obj, name, function)

    def restore(self):
        for obj, name, existed, previous in reversed(self._undo):
            if existed:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self._undo.clear()
