"""Experimental recipe definitions; these do not install a quantized kernel."""

FAMILIES = ("fp16", "cmix", "ptmix", "a4mix")
CHANNEL_MODULES = tuple(
    f"mlp_stack.{block}.channel_mix_{side}"
    for block in (3, 2, 1, 0)
    for side in ("in", "out")
)


def describe_recipe(family, layers, *, flow_steps=5):
    """Describe requested coverage independently from actual backend evidence.

    PTMix here means selected per-tensor W4 layers with the remainder FP16.
    It does not mean the historical all-W4 mixed-scale tail.
    A4Mix describes the later per-channel W4 / calibrated per-block A4 path.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown quantization family: {family}")
    if type(layers) is not int or not 0 <= layers <= len(CHANNEL_MODULES):
        raise ValueError("layers must be an integer from 0 through 8")
    if type(flow_steps) is not int or flow_steps < 1:
        raise ValueError("flow_steps must be a positive integer")
    if family == "fp16" and layers:
        raise ValueError("FP16 must request zero quantized layers")
    if family == "a4mix" and layers % 2:
        raise ValueError("A4Mix requires complete blocks: 0, 2, 4, 6 or 8 layers")
    if layers == 0:
        family = "fp16"
    selected = CHANNEL_MODULES[:layers]
    stage = "fp16" if not layers else f"{family}{layers}"
    return {
        "recipe_id": f"flow{flow_steps}-{stage}",
        "family": family,
        "flow_steps": flow_steps,
        "requested_quantized_channel_linears": layers,
        "requested_modules": list(selected),
        "remaining_fp16_channel_modules": list(CHANNEL_MODULES[layers:]),
        "selected_blocks": list(
            dict.fromkeys(int(name.split(".")[1]) for name in selected)
        ),
        "weight_bits": 4 if layers else 16,
        "activation_bits": 4 if family == "a4mix" else 16,
        "weight_scale_granularity": (
            "none"
            if not layers
            else "per_tensor"
            if family == "ptmix"
            else "per_output_channel"
        ),
        "activation_scale_granularity": "static_per_block"
        if family == "a4mix"
        else "none",
        "activation_calibration_required": family == "a4mix",
        "requested_policy_backend": "torch_native",
        "execution_available": False,
        "implementation_status": "recipe_only",
    }
