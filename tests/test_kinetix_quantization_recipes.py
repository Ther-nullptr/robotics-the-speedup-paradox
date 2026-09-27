"""Keep layer selection and scale semantics distinct during quantization work."""

import pytest


def test_weight_only_families_keep_distinct_scale_semantics():
    from robotics_bench.kinetix.quantization_recipes import describe_recipe

    channel = describe_recipe("cmix", 3)
    tensor = describe_recipe("ptmix", 3)
    assert (
        channel["requested_modules"]
        == tensor["requested_modules"]
        == [
            "mlp_stack.3.channel_mix_in",
            "mlp_stack.3.channel_mix_out",
            "mlp_stack.2.channel_mix_in",
        ]
    )
    assert channel["weight_scale_granularity"] == "per_output_channel"
    assert tensor["weight_scale_granularity"] == "per_tensor"
    assert channel["recipe_id"] != tensor["recipe_id"]
    assert len(tensor["remaining_fp16_channel_modules"]) == 5
    assert not tensor["execution_available"]


def test_activation_quantization_requires_complete_blocks_and_calibration():
    from robotics_bench.kinetix.quantization_recipes import describe_recipe

    with pytest.raises(ValueError, match="complete blocks"):
        describe_recipe("a4mix", 3)
    recipe = describe_recipe("a4mix", 4, flow_steps=3)
    assert recipe["selected_blocks"] == [3, 2]
    assert recipe["activation_bits"] == 4
    assert recipe["activation_calibration_required"]
    assert recipe["flow_steps"] == 3
    assert describe_recipe("ptmix", 0) == describe_recipe("cmix", 0)
    with pytest.raises(ValueError, match="layers"):
        describe_recipe("ptmix", 9)
    with pytest.raises(ValueError, match="flow_steps"):
        describe_recipe("ptmix", 4, flow_steps=0)
