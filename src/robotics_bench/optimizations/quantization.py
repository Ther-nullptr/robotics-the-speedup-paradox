"""Model-specific selection above independently owned precision backends."""

from contextlib import contextmanager
from .patching import Patches


class ActivationCache:
    """Reuse a pack only across one known sibling projection group."""

    def __init__(self, prepare, fanout):
        self.prepare = prepare
        self.fanout = fanout
        self.input = None
        self.packed = None
        self.remaining = 0

    def get(self, x):
        if self.input is not x or self.remaining == 0:
            self.packed = self.prepare(x)
            self.input = x
            self.remaining = self.fanout
        packed = self.packed
        self.remaining -= 1
        if not self.remaining:
            self.input = None
            self.packed = None
        return packed


def pi05_scope(name):
    if ".vision_model.encoder.layers." in name:
        return "vision"
    if ".paligemma.model.language_model.layers." in name:
        return "text"
    if ".gemma_expert.model.layers." in name:
        return "expert"
    if ".multi_modal_projector." in name:
        return "projector"
    return None


@contextmanager
def quantize_linears(
    model,
    precision,
    *,
    scopes=("text",),
    shared_quant=False,
    tactic=0,
    selector=pi05_scope,
    grouped=False,
    pack_reuse=False,
    site_bits=None,
    group_qkv=False,
    group_gate_up=False,
    group_views=False,
):
    import torch

    if precision not in ("int8", "int4", "fp8", "fp4"):
        raise ValueError("Unknown low-precision backend")
    if not scopes or any(
        s not in ("text", "expert", "vision", "projector", "dit") for s in scopes
    ):
        raise ValueError("Explicit supported module scopes are required")
    if precision.startswith("int"):
        from robotics_kernels.integer import IntegerLinear

        bits = 8 if precision == "int8" else 4

        def make(m, n):
            return IntegerLinear.from_linear(
                m,
                bits=site_bits[n] if site_bits is not None else bits,
                tactic=tactic,
                pack_reuse=pack_reuse,
            )
    elif precision == "fp8":
        from robotics_kernels.blackwell.fp8_linear import Fp8CutlassLinear

        def make(m, n):
            return Fp8CutlassLinear.from_linear(m, role="linear", target_name=n)
    else:
        from robotics_kernels.blackwell.fp4_linear import Fp4CutlassLinear

        def make(m, n):
            return Fp4CutlassLinear.from_linear(m, role="linear", target_name=n)

    p = Patches()
    report = {
        "precision": precision,
        "scopes": list(scopes),
        "modules": {},
        "skipped": {},
        "shared_quant": shared_quant,
        "tactic": tactic,
        "grouped": {},
        "pack_reuse": pack_reuse,
    }
    try:
        replacements = {}
        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear) or selector(name) not in scopes:
                continue
            if module.weight.dtype != torch.bfloat16:
                report["skipped"][name] = "Original parameter dtype is not BF16"
                continue
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            replacement = make(module, name)
            p.set(parent, attr, replacement)
            replacements[name] = replacement
            report["modules"][name] = {
                "backend": getattr(
                    replacement,
                    "backend",
                    getattr(replacement, "fp8_backend", "cutlass_fp4"),
                ),
                "in_features": module.in_features,
                "out_features": module.out_features,
            }
        if not replacements:
            raise ValueError("No eligible Linear modules in the selected scope")
        if shared_quant or grouped or group_qkv or group_gate_up:
            if not precision.startswith("int"):
                raise ValueError(
                    "Shared preparation currently supports integer backends"
                )
            from robotics_kernels.ampere_ada.integer import (
                IntegerProjectionGroup,
                IntegerProjectionSlice,
            )

            for module_name, module in list(model.named_modules()):
                attention_names = (
                    ("q_proj", "k_proj", "v_proj")
                    if getattr(module, "is_selfattn", True)
                    else ("k_proj", "v_proj")
                )
                for group_index, names in enumerate(
                    (attention_names, ("gate_proj", "up_proj"))
                ):
                    group_this = grouped or (
                        group_qkv if group_index == 0 else group_gate_up
                    )
                    compatible = {}
                    for attr in names:
                        m = getattr(module, attr, None)
                        if isinstance(m, IntegerLinear):
                            compatible.setdefault((m.bits, m.in_features), []).append(
                                (attr, m)
                            )
                    for (bits, _), pairs in compatible.items():
                        if len(pairs) < 2:
                            continue
                        siblings = [m for _, m in pairs]
                        if group_this:
                            group = IntegerProjectionGroup(
                                siblings, contiguous_outputs=not group_views
                            )
                            key = f"_robotics_integer_group_{group_index}_{bits}"
                            p.set(module, key, group)
                            for index, (attr, _) in enumerate(pairs):
                                p.set(
                                    module, attr, IntegerProjectionSlice(group, index)
                                )
                            report["grouped"][module_name + "." + key] = {
                                "members": [attr for attr, _ in pairs],
                                "bits": bits,
                                "in_features": group.linear.in_features,
                                "out_features": list(group.sizes),
                                "contiguous_outputs": not group_views,
                            }
                            continue
                        if not shared_quant:
                            continue
                        cache = ActivationCache(siblings[0].pack_input, len(siblings))
                        for m in siblings:

                            def forward(linear, x, cache=cache):
                                return linear.forward_packed(cache.get(x))

                            p.bind(m, "forward", forward)
        yield report
    finally:
        p.restore()
