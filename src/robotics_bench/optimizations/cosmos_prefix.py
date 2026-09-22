"""Skip unconsumed causal VAE suffix work at the action-only engine boundary.

The full raw sequence, final latent projection, diffusion shape and noise remain
unchanged. Auxiliary clean-latent suffixes are not preserved; this adapter must
not be used by planning/visualization callers that consume those private values.
"""

from contextlib import contextmanager
from .patching import Patches


# Bind model geometry to the engine contract; camera layouts are not interchangeable.
_ACTION_BOUNDARIES = {
    (9, 4): ("libero", 16, ("primary_image", "wrist_image")),
    (11, 5): (
        "robocasa",
        32,
        ("primary_image", "secondary_image", "wrist_image"),
    ),
}


def prefix_plan(input_frames, condition_frames, factor, window):
    values = (input_frames, condition_frames, factor, window)
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError("Prefix dimensions must be positive integers")
    if (input_frames - 1) % factor or window % factor:
        raise ValueError("Prefix geometry must match native causal compression")
    latents = 1 + (input_frames - 1) // factor
    if condition_frames > latents:
        raise ValueError("Condition prefix exceeds the latent sequence")
    needed = (condition_frames - 1) * factor
    encoded = min(input_frames, 1 + ((needed + window - 1) // window) * window)
    return {
        "input_frames": input_frames,
        "latent_frames": latents,
        "condition_frames": condition_frames,
        "encoded_frames": encoded,
        "factor": factor,
        "window": window,
    }


class EncoderPrefixGate:
    """One native encode call, keeping its loop and final projection intact."""

    def __init__(self, original, plan):
        self.original, self.plan = original, plan
        self.pixel_offset = self.executed_frames = self.skipped_frames = 0
        self.prototype = self.input_signature = None

    def __call__(self, x, *args, **kwargs):
        import torch

        frames = x.shape[2]
        start, end = self.pixel_offset, self.pixel_offset + frames
        if frames < 1 or end > self.plan["input_frames"]:
            raise ValueError("Unexpected encoder chunk length")
        if start < self.plan["encoded_frames"] < end:
            raise ValueError("Encoder chunk crosses the admitted prefix boundary")
        signature = (x.shape[0], x.shape[1], *x.shape[3:], x.dtype, x.device)
        if self.input_signature is not None and signature != self.input_signature:
            raise ValueError("Encoder geometry changed within one request")
        self.input_signature = signature
        if start < self.plan["encoded_frames"]:
            output = self.original(x, *args, **kwargs)
            self.prototype = (
                output.shape[0],
                output.shape[1],
                *output.shape[3:],
                output.dtype,
                output.device,
                output.is_contiguous(memory_format=torch.channels_last_3d),
            )
            self.executed_frames += frames
        else:
            if self.prototype is None or frames % self.plan["factor"]:
                raise ValueError("Cannot infer a native suffix output shape")
            batch, channels, height, width, dtype, device, channels_last = (
                self.prototype
            )
            output = torch.empty(
                (batch, channels, frames // self.plan["factor"], height, width),
                dtype=dtype,
                device=device,
                memory_format=torch.channels_last_3d
                if channels_last
                else torch.contiguous_format,
            ).zero_()
            self.skipped_frames += frames
        self.pixel_offset = end
        return output


@contextmanager
def optimize_conditioning_prefix(model):
    import torch

    tokenizer = model.tokenizer
    wrapper = tokenizer.model
    core = wrapper.model
    config = model.config
    boundary = _ACTION_BOUNDARIES.get(
        (config.state_t, config.min_num_conditional_frames)
    )
    if (
        type(core).__module__ != "cosmos_policy.tokenizers.wan2pt1"
        or type(core).__name__ != "WanVAE_"
        or model.net.training
        or core.training
        or wrapper.is_parallel
        or tokenizer.keep_encoder_cache
        or getattr(model.net, "is_context_parallel_enabled", False)
        or boundary is None
        or config.min_num_conditional_frames != config.max_num_conditional_frames
        or config.use_flowunipc_scheduler
        or not config.denoise_replace_gt_frames
    ):
        raise ValueError(
            "Condition-prefix reuse requires a supported action-only LIBERO or RoboCasa Wan configuration"
        )
    admitted_suite, expected_horizon, expected_images = boundary
    projection = core.conv1
    if (
        tuple(projection.kernel_size) != (1, 1, 1)
        or tuple(projection.stride) != (1, 1, 1)
        or tuple(projection.padding) != (0, 0, 0)
        or tuple(projection.dilation) != (1, 1, 1)
    ):
        raise ValueError("The final VAE projection must not mix temporal positions")
    std = wrapper.video_std
    if not bool((torch.isfinite(std) & (std != 0)).all()):
        raise ValueError(
            "VAE normalization requires finite nonzero standard deviations"
        )
    plan = prefix_plan(
        tokenizer.get_pixel_num_frames(config.state_t),
        config.min_num_conditional_frames,
        tokenizer.temporal_compression_factor,
        core.temporal_window,
    )
    report = {
        "suite": admitted_suite,
        "plan": plan,
        "requests": 0,
        "encode_calls": 0,
        "encoded_pixel_frames": 0,
        "skipped_pixel_frames": 0,
        "mask_checks": 0,
        "equivalence_boundary": "engine actions only; auxiliary clean-latent suffix is not preserved",
        "noise_shape_unchanged": True,
    }
    request = gate = None
    patches = Patches()
    original_encode, original_encoder = tokenizer.encode, core.encoder.forward
    original_denoise = model.denoise

    @contextmanager
    def action_request(*, suite, action_horizon, image_keys, use_proprio):
        nonlocal request
        if request is not None:
            raise RuntimeError("Condition-prefix reuse requires serialized requests")
        if (
            suite != admitted_suite
            or action_horizon != expected_horizon
            or not use_proprio
            or tuple(image_keys) != expected_images
        ):
            raise ValueError(
                f"Condition-prefix reuse requires the {admitted_suite} action boundary "
                f"with H={expected_horizon} and cameras={expected_images}"
            )
        request = {"encodes": 0, "mask_checked": False}
        try:
            yield
            if not request["encodes"] or not request["mask_checked"]:
                raise RuntimeError(
                    "Action request did not validate its encoded conditioning mask"
                )
            report["requests"] += 1
        finally:
            request = None

    def encode(module, state):
        nonlocal gate
        if request is None:
            return original_encode(state)
        if (
            gate is not None
            or torch.is_grad_enabled()
            or state.ndim != 5
            or state.shape[0] != 1
            or state.shape[1] != 3
            or state.shape[2] != plan["input_frames"]
        ):
            raise ValueError("Unexpected action-only VAE encode workload")
        gate = EncoderPrefixGate(original_encoder, plan)
        try:
            # Preserve the policy-specific RNG reset, clear_cache calls, full
            # conv1 geometry, latent scale and video mean/std normalization.
            value = original_encode(state)
            if (
                gate.pixel_offset != state.shape[2]
                or value.shape[2] != plan["latent_frames"]
            ):
                raise ValueError(
                    "Native encode did not preserve the full sequence shape"
                )
            report["encode_calls"] += 1
            report["encoded_pixel_frames"] += gate.executed_frames
            report["skipped_pixel_frames"] += gate.skipped_frames
            request["encodes"] += 1
            request["mask_checked"] = False
            return value
        finally:
            gate = None

    def encoder(module, x, *args, **kwargs):
        return (
            original_encoder(x, *args, **kwargs)
            if gate is None
            else gate(x, *args, **kwargs)
        )

    def denoise(module, x, sigma, condition):
        if request is not None:
            mask = condition.condition_video_input_mask_B_C_T_H_W
            if mask.ndim != 5 or mask.shape[2] != plan["latent_frames"]:
                raise ValueError("Unexpected conditioning mask shape")
            valid = (
                torch.isfinite(mask).all()
                & (mask[:, :, plan["condition_frames"] :] == 0).all()
            )
            if not bool(valid):
                raise ValueError(
                    "An omitted VAE suffix is used by the conditioning mask"
                )
            report["mask_checks"] += 1
            request["mask_checked"] = True
        return original_denoise(x, sigma, condition)

    try:
        patches.set(model, "_robotics_action_only_context", action_request)
        patches.bind(tokenizer, "encode", encode)
        patches.bind(core.encoder, "forward", encoder)
        patches.bind(model, "denoise", denoise)
        yield report
    finally:
        patches.restore()
