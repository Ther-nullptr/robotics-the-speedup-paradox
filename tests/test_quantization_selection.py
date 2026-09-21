"""Shared activation packs expire after their declared sibling consumers."""

from robotics_bench.optimizations.quantization import ActivationCache, pi05_scope


def test_pack_is_reused_only_within_one_projection_group():
    calls = []

    def prepare(x):
        value = object()
        calls.append(value)
        return value

    cache = ActivationCache(prepare, 3)
    x = object()
    other = object()
    first = cache.get(x)
    assert cache.get(x) is first and cache.get(x) is first
    assert cache.get(x) is not first
    assert cache.get(other) is not calls[-2]
    assert len(calls) == 3


def test_scope_does_not_quantize_unused_language_head_or_action_output():
    assert (
        pi05_scope(
            "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.up_proj"
        )
        == "text"
    )
    assert (
        pi05_scope("paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj")
        == "expert"
    )
    assert pi05_scope("paligemma_with_expert.paligemma.lm_head") is None
    assert pi05_scope("action_out_proj") is None
