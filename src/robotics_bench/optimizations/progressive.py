"""Cosmos B.1/Table 5 site selection; BF16 reference adaptation, no calibration."""


def cosmos_candidate_sites():
    roles = [
        f"{attention}.{projection}"
        for attention in ("self_attn", "cross_attn")
        for projection in ("q_proj", "k_proj", "v_proj", "output_proj")
    ]
    roles += ["mlp.layer1", "mlp.layer2"]
    return [f"blocks.{block}.{role}" for block in range(28) for role in roles]


def cosmos_precision_map(names, tier):
    if type(tier) is not int or not 0 <= tier <= 10:
        raise ValueError("Cosmos quantization tier must be an integer from 0 to 10")
    names = list(names)
    canonical = [n.replace("._checkpoint_wrapped_module", "") for n in names]
    expected = cosmos_candidate_sites()
    if len(canonical) != len(expected) or set(canonical) != set(expected):
        raise ValueError(
            "Cosmos tier protocol requires the exact 280 candidate sites in 28 blocks"
        )
    result = {}
    for original, name in zip(names, canonical):
        _, block, family, role = name.split(".")
        block = int(block)
        early, late = block < 10, block >= 20
        low = (
            (
                family == "cross_attn"
                and role == "output_proj"
                and ((tier >= 1 and late) or (tier >= 3 and early) or tier >= 6)
            )
            or (
                family == "mlp"
                and role == "layer1"
                and ((tier >= 2 and early) or (tier >= 4 and late) or tier >= 5)
            )
            or (tier >= 7 and family == "mlp" and role == "layer2")
            or (tier >= 8 and family == "self_attn" and role != "v_proj")
            or (
                tier >= 9
                and (
                    (family == "self_attn" and role == "v_proj")
                    or (family == "cross_attn" and role == "q_proj")
                )
            )
            or tier >= 10
        )
        result[original] = 4 if low else 8
    return result
