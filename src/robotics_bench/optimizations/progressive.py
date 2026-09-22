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


def cosmos_tier_variants(template, tiers=None):
    """Construct a reproducible W8A8-to-W4A4 sweep without changing the workload."""
    tiers = list(range(11) if tiers is None else tiers)
    if (
        template.scopes != ("dit",)
        or template.precision not in ("bf16", "int8")
        or template.quant_tier is not None
    ):
        raise ValueError(
            "A progressive sweep requires a BF16/INT8 Cosmos dit template without a single tier"
        )
    if (
        not tiers
        or len(set(tiers)) != len(tiers)
        or any(type(t) is not int or not 0 <= t <= 10 for t in tiers)
    ):
        raise ValueError("Select unique integer tiers from 0 to 10")
    switches = list(template.enabled)
    for switch in ("shared_quant", "activation_quant_fusion", "integer_pack_reuse"):
        if switch not in switches:
            switches.append(switch)
    return [
        {
            "id": "w8a8" if tier == 0 else f"w4-t{tier}",
            "precision": "int8",
            "switches": switches,
            "scopes": ["dit"],
            "tactic": template.tactic,
            "quant_tier": tier,
        }
        for tier in tiers
    ]
