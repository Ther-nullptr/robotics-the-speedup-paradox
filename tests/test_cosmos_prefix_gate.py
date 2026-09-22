"""Skipped suffix chunks do not replace the native final projection or RNG path."""

import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("layout", ["contiguous", "channels_last"])
def test_encoder_gate_only_skips_suffix_and_preserves_prefix(layout):
    from robotics_bench.optimizations.cosmos_prefix import (
        EncoderPrefixGate,
        prefix_plan,
    )

    calls = []

    def encode(x, **kwargs):
        calls.append(x.shape[2])
        return (x[:, :2, ::4] + 2).contiguous(
            memory_format=torch.channels_last_3d
            if layout == "channels_last"
            else torch.contiguous_format
        )

    plan = prefix_plan(41, 5, 4, 16)
    x = torch.randn(1, 3, 41, 8, 8)
    pieces = [x[:, :, :1], x[:, :, 1:17], x[:, :, 17:33], x[:, :, 33:]]
    expected = torch.cat([encode(piece) for piece in pieces], 2)
    calls.clear()
    gate = EncoderPrefixGate(encode, plan)
    actual = torch.cat([gate(piece) for piece in pieces], 2)
    assert calls == [1, 16]
    assert actual.shape == expected.shape
    assert torch.equal(actual[:, :, :5], expected[:, :, :5])
    assert torch.count_nonzero(actual[:, :, 5:]) == 0
    assert gate.pixel_offset == 41
    assert gate.executed_frames == 17 and gate.skipped_frames == 24


def test_unexpected_chunk_crossing_the_prefix_fails():
    from robotics_bench.optimizations.cosmos_prefix import (
        EncoderPrefixGate,
        prefix_plan,
    )

    gate = EncoderPrefixGate(lambda x: x, prefix_plan(41, 5, 4, 16))
    with pytest.raises(ValueError, match="boundary"):
        gate(torch.zeros(1, 3, 20, 8, 8))


def toy_policy(suite="robocasa"):
    latents, conditions = (9, 4) if suite == "libero" else (11, 5)

    class Encoder(torch.nn.Module):
        def forward(self, x):
            return (x[:, :2, ::4] + 2).contiguous()

    core_type = type("WanVAE_", (), {})
    core_type.__module__ = "cosmos_policy.tokenizers.wan2pt1"
    core = core_type()
    core.training = False
    core.temporal_window = 16
    core.encoder = Encoder().eval()
    core.conv1 = torch.nn.Conv3d(2, 2, 1).eval()

    class Tokenizer:
        keep_encoder_cache = False
        temporal_compression_factor = 4
        model = SimpleNamespace(
            model=core, is_parallel=False, video_std=torch.ones(latents)
        )

        def get_pixel_num_frames(self, n):
            return 1 + 4 * (n - 1)

        def encode(self, x):
            pieces = [x[:, :, :1]] + [
                x[:, :, start : start + 16] for start in range(1, x.shape[2], 16)
            ]
            return core.conv1(torch.cat([core.encoder(piece) for piece in pieces], 2))

    return SimpleNamespace(
        tokenizer=Tokenizer(),
        net=SimpleNamespace(training=False, is_context_parallel_enabled=False),
        config=SimpleNamespace(
            state_t=latents,
            min_num_conditional_frames=conditions,
            max_num_conditional_frames=conditions,
            use_flowunipc_scheduler=False,
            denoise_replace_gt_frames=True,
        ),
        denoise=lambda x, sigma, condition: x,
    )


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("suite", ["libero", "robocasa"])
def test_each_denoise_mask_is_checked_and_failed_request_is_cleared(inplace, suite):
    from robotics_bench.optimizations.cosmos_prefix import optimize_conditioning_prefix

    model = toy_policy(suite)
    frames = model.tokenizer.get_pixel_num_frames(model.config.state_t)
    x = torch.randn(1, 3, frames, 8, 8)
    native = model.tokenizer.encode(x)
    with torch.inference_mode(), optimize_conditioning_prefix(model) as report:
        with pytest.raises(ValueError, match="omitted"):
            with model._robotics_action_only_context(
                suite=suite,
                action_horizon=16 if suite == "libero" else 32,
                image_keys=("primary_image", "wrist_image")
                if suite == "libero"
                else ("primary_image", "secondary_image", "wrist_image"),
                use_proprio=True,
            ):
                model.tokenizer.encode(x)
                mask = torch.zeros(1, 1, model.config.state_t, 8, 8)
                mask[:, :, : model.config.min_num_conditional_frames] = 1
                condition = SimpleNamespace(condition_video_input_mask_B_C_T_H_W=mask)
                model.denoise(x, torch.ones(1), condition)
                if not inplace:
                    mask = mask.clone()
                    condition = SimpleNamespace(
                        condition_video_input_mask_B_C_T_H_W=mask
                    )
                mask[:, :, 6] = 1
                model.denoise(x, torch.ones(1), condition)
        assert report["requests"] == 0
        assert torch.equal(model.tokenizer.encode(x), native)
    assert not hasattr(model, "_robotics_action_only_context")


@pytest.mark.parametrize("suite", ["libero", "robocasa"])
def test_shared_prefix_matches_case_boundary_and_preserves_condition(suite):
    from robotics_bench.optimizations.cosmos_prefix import optimize_conditioning_prefix

    model = toy_policy(suite)
    frames = model.tokenizer.get_pixel_num_frames(model.config.state_t)
    x = torch.randn(1, 3, frames, 8, 8)
    native = model.tokenizer.encode(x)
    boundary = dict(
        suite=suite,
        action_horizon=16 if suite == "libero" else 32,
        image_keys=("primary_image", "wrist_image")
        if suite == "libero"
        else ("primary_image", "secondary_image", "wrist_image"),
        use_proprio=True,
    )
    with torch.inference_mode(), optimize_conditioning_prefix(model) as report:
        for wrong in (
            {"suite": "robocasa" if suite == "libero" else "libero"},
            {"action_horizon": boundary["action_horizon"] + 1},
            {"image_keys": tuple(reversed(boundary["image_keys"]))},
        ):
            with pytest.raises(ValueError, match="action boundary"):
                with model._robotics_action_only_context(**(boundary | wrong)):
                    raise AssertionError("An incompatible case was admitted")
        with model._robotics_action_only_context(**boundary):
            actual = model.tokenizer.encode(x)
            condition_frames = model.config.min_num_conditional_frames
            assert actual.shape == native.shape
            assert torch.equal(
                actual[:, :, :condition_frames], native[:, :, :condition_frames]
            )
            mask = torch.zeros(1, 1, model.config.state_t, 8, 8)
            mask[:, :, :condition_frames] = 1
            model.denoise(
                x,
                torch.ones(1),
                SimpleNamespace(condition_video_input_mask_B_C_T_H_W=mask),
            )
        assert report["suite"] == suite
        assert report["requests"] == 1
        assert report["encoded_pixel_frames"] == 17
        assert report["skipped_pixel_frames"] == frames - 17
