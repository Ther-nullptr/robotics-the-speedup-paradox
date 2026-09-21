"""Projection reuse restricted to caller-registered immutable condition tensors."""


class ConstantProjectionCache:
    def __init__(self, function, eligible):
        self.function = function
        self.eligible = eligible
        self._values = {}

    def __call__(self, value):
        if not self.eligible(value):
            return self.function(value)
        key = id(value)
        if key not in self._values:
            # Keep the input alive so Python cannot recycle this identity.
            self._values[key] = (value, self.function(value))
        return self._values[key][1]

    def __len__(self):
        return len(self._values)

    def retained_tensors(self):
        return tuple(output for _, output in self._values.values())
