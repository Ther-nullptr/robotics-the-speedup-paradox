"""One instance-owned projection shared by Q/K/V or gate/up views.

Adapted in structure from VLM Fp8FusedProjectionGroup. All slices consume the
same input within one forward; the result is released after the last slice.
"""

import weakref
import torch


class ProjectionGroup(torch.nn.Module):
    def __init__(self, linears):
        super().__init__()
        first = linears[0]
        if any(
            linear.in_features != first.in_features
            or linear.weight.dtype != first.weight.dtype
            or linear.weight.device != first.weight.device
            for linear in linears
        ):
            raise ValueError("Projection inputs must share width, dtype and device")
        self.sizes = tuple(linear.out_features for linear in linears)
        self.linear = torch.nn.Linear(
            first.in_features,
            sum(self.sizes),
            bias=any(linear.bias is not None for linear in linears),
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        with torch.no_grad():
            self.linear.weight.copy_(torch.cat([linear.weight for linear in linears]))
            if self.linear.bias is not None:
                self.linear.bias.copy_(
                    torch.cat(
                        [
                            linear.bias
                            if linear.bias is not None
                            else linear.weight.new_zeros(linear.out_features)
                            for linear in linears
                        ]
                    )
                )
        self.linear.requires_grad_(False)
        self._input = None
        self._output = None
        self._remaining = set()

    def project(self, x, index):
        if self._input is not x or index not in self._remaining:
            self._input = x
            self._output = self.linear(x)
            self._remaining = set(range(len(self.sizes)))
        output = self._output.narrow(-1, sum(self.sizes[:index]), self.sizes[index])
        self._remaining.remove(index)
        if not self._remaining:
            self._input = None
            self._output = None
        return output


class ProjectionSlice(torch.nn.Module):
    def __init__(self, group, index):
        super().__init__()
        object.__setattr__(self, "_group", weakref.ref(group))
        self.index = index
        self.in_features = group.linear.in_features
        self.out_features = group.sizes[index]

    @property
    def weight(self):
        return self._group().linear.weight

    @property
    def bias(self):
        group = self._group()
        if group.linear.bias is None:
            return None
        return group.linear.bias.narrow(
            0, sum(group.sizes[: self.index]), self.out_features
        )

    def forward(self, x):
        return self._group().project(x, self.index)
