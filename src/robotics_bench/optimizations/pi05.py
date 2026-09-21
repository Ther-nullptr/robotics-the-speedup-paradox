"""Reversible switches for the repository-owned PI0.5 inference graph."""

from contextlib import contextmanager

from .patching import Patches


@contextmanager
def optimize_pi05(model, config):
    """Apply independent switches to one model; restore every patch on exit.

    Inference is serialized per model instance. Masks live for one observation;
    no generated suffix KV or observation embedding survives a request.
    """
    import torch
    from robotics_bench.models.pi05 import modeling_pi05 as native
    from robotics_bench.models.pi05 import modeling_gemma as gemma

    if not isinstance(model, native.PI05Pytorch):
        raise TypeError("Optimization requires the repository-owned PI05Pytorch")
    supported = {
        "flow_loop",
        "mask_cache",
        "rope",
        "gated_residual",
        "gelu_mul",
        "cuda_graph",
        "norm",
        "condition_cache",
        "empty_image_cache",
        "projection_fusion",
        "shared_quant",
        "activation_quant_fusion",
        "integer_grouped",
        "integer_pack_reuse",
        "integer_qkv",
        "integer_gate_up",
        "integer_group_views",
    }
    if "cuda_graph" in config.enabled and "flow_loop" not in config.enabled:
        raise ValueError(
            "cuda_graph requires flow_loop to remove device-dependent host control"
        )
    if "activation_quant_fusion" in config.enabled and not config.precision.startswith(
        "int"
    ):
        raise ValueError("activation_quant_fusion requires INT4 or INT8")
    if "empty_image_cache" in config.enabled and "cuda_graph" not in config.enabled:
        raise ValueError("empty_image_cache currently requires cuda_graph")
    unsupported = set(config.enabled) - supported
    if unsupported:
        raise ValueError(f"Unimplemented PI0.5 switches: {sorted(unsupported)}")
    if model.training:
        raise ValueError("Optimizations require eval mode")
    from contextlib import ExitStack

    stack = ExitStack()
    p = Patches()
    report = {
        "switches": list(config.enabled),
        "precision": config.precision,
        "coverage": {},
    }
    try:
        if "projection_fusion" in config.enabled:
            from robotics_kernels.projections import ProjectionGroup, ProjectionSlice

            for name, module in list(model.named_modules()):
                names = None
                if isinstance(module, gemma.GemmaAttention):
                    names = ("q_proj", "k_proj", "v_proj")
                if isinstance(module, gemma.GemmaMLP):
                    names = ("gate_proj", "up_proj")
                if names is not None:
                    group = ProjectionGroup([getattr(module, n) for n in names])
                    p.set(module, "_robotics_projection_group", group)
                    for i, n in enumerate(names):
                        p.set(module, n, ProjectionSlice(group, i))
                    report["coverage"][name + ".projection"] = "bf16_concatenated_gemm"
        if config.precision != "bf16":
            from .quantization import quantize_linears

            report["quantization"] = stack.enter_context(
                quantize_linears(
                    model,
                    config.precision,
                    scopes=tuple(getattr(config, "scopes", ("text",))),
                    shared_quant="shared_quant" in config.enabled,
                    grouped="integer_grouped" in config.enabled,
                    pack_reuse="integer_pack_reuse" in config.enabled,
                    group_qkv="integer_qkv" in config.enabled,
                    group_gate_up="integer_gate_up" in config.enabled,
                    group_views="integer_group_views" in config.enabled,
                    tactic=getattr(config, "tactic", 0),
                )
            )
        if {
            "rope",
            "gated_residual",
            "gelu_mul",
            "norm",
            "activation_quant_fusion",
        } & set(config.enabled):
            from robotics_kernels import fused

            for name, mod in list(model.named_modules()):
                if "rope" in config.enabled and isinstance(mod, gemma.GemmaAttention):
                    p.globals(mod, "forward", {"apply_rotary_pos_emb": fused.rope})
                    report["coverage"][name + ".rope"] = "triton_preserve_rounding"
                if "gated_residual" in config.enabled and isinstance(
                    mod, gemma.GemmaDecoderLayer
                ):
                    p.globals(mod, "forward", {"_gated_residual": fused.gated_residual})
                    report["coverage"][name + ".residual"] = "triton_preserve_rounding"
                if "norm" in config.enabled and isinstance(mod, gemma.GemmaRMSNorm):

                    def norm_forward(module, x, cond=None):
                        inv = torch.rsqrt(
                            torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
                            + module.eps
                        )
                        if cond is None or module.dense is None:
                            return fused.norm_affine(
                                x, inv, module.weight.float()
                            ), None
                        modulation = module.dense(cond).unsqueeze(1)
                        scale, shift, gate = modulation.chunk(3, dim=-1)
                        return fused.norm_affine(x, inv, scale, shift), gate.to(x.dtype)

                    p.bind(mod, "forward", norm_forward)
                    report["coverage"][name + ".norm"] = (
                        "torch_reference_reduction_triton_affine"
                    )
                if isinstance(mod, gemma.GemmaMLP) and (
                    "gelu_mul" in config.enabled
                    or (
                        "activation_quant_fusion" in config.enabled
                        and hasattr(mod.down_proj, "forward_gelu")
                    )
                ):
                    fused.prepare_gelu(
                        mod.gate_proj.weight.device, mod.gate_proj.weight.dtype
                    )

                    def mlp_forward(module, x):
                        if "activation_quant_fusion" in config.enabled and hasattr(
                            module.down_proj, "forward_gelu"
                        ):
                            gate, up = module.gate_proj(x), module.up_proj(x)
                            if "integer_group_views" not in config.enabled:
                                gate, up = gate.contiguous(), up.contiguous()
                            return module.down_proj.forward_gelu(gate, up)
                        return module.down_proj(
                            fused.gelu_mul(
                                module.gate_proj(x).contiguous(),
                                module.up_proj(x).contiguous(),
                            )
                        )

                    p.bind(mod, "forward", mlp_forward)
                    report["coverage"][name + ".activation"] = "triton_torch_gelu_lut"
        original_sample = model.sample_actions
        mask_state = {}
        if "mask_cache" in config.enabled:

            def denoise(module, prefix_pad_masks, past_key_values, x_t, timestep):
                suffix_embs, suffix_pad_masks, suffix_att_masks, cond = (
                    module.embed_suffix(x_t, timestep)
                )
                if not mask_state:
                    b, s = suffix_pad_masks.shape
                    prefix = prefix_pad_masks[:, None, :].expand(
                        b, s, prefix_pad_masks.shape[1]
                    )
                    suffix = native.make_att_2d_masks(
                        suffix_pad_masks, suffix_att_masks
                    )
                    mask_state["mask"] = module._prepare_attention_masks_4d(
                        torch.cat([prefix, suffix], dim=2)
                    )
                    mask_state["positions"] = (
                        torch.sum(prefix_pad_masks, dim=-1)[:, None]
                        + torch.cumsum(suffix_pad_masks, dim=1)
                        - 1
                    )
                module.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
                outputs, _ = module.paligemma_with_expert.forward(
                    attention_mask=mask_state["mask"],
                    position_ids=mask_state["positions"],
                    past_key_values=past_key_values,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, cond],
                )
                suffix_out = outputs[1][:, -module.config.chunk_size :].to(
                    torch.float32
                )
                return module.action_out_proj(suffix_out)

            p.bind(model, "denoise_step", denoise)
        if "flow_loop" in config.enabled or "mask_cache" in config.enabled:

            @torch.no_grad()
            def sample(
                module, images, img_masks, tokens, masks, noise=None, num_steps=None
            ):
                mask_state.clear()
                try:
                    if "flow_loop" not in config.enabled:
                        return original_sample(
                            images,
                            img_masks,
                            tokens,
                            masks,
                            noise=noise,
                            num_steps=num_steps,
                        )
                    steps = (
                        module.config.num_inference_steps
                        if num_steps is None
                        else num_steps
                    )
                    if type(steps) is not int or not 1 <= steps <= 1000:
                        raise ValueError("flow_loop supports 1..1000 integer steps")
                    b = tokens.shape[0]
                    device = tokens.device
                    if noise is None:
                        noise = module.sample_noise(
                            (b, module.config.chunk_size, module.config.max_action_dim),
                            device,
                        )
                    embeddings, pad, att = module.embed_prefix(
                        images, img_masks, tokens, masks
                    )
                    prefix_mask = module._prepare_attention_masks_4d(
                        native.make_att_2d_masks(pad, att)
                    )
                    positions = torch.cumsum(pad, dim=1) - 1
                    module.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
                    _, cache = module.paligemma_with_expert.forward(
                        attention_mask=prefix_mask,
                        position_ids=positions,
                        past_key_values=None,
                        inputs_embeds=[embeddings, None],
                        use_cache=True,
                    )
                    dt = torch.full(
                        (), -1.0 / steps, dtype=torch.float32, device=device
                    )
                    time = torch.ones((), dtype=torch.float32, device=device)
                    x = noise
                    for step in range(steps):
                        if "condition_cache" in config.enabled:
                            module._robotics_flow_step = step
                            module._robotics_flow_steps = steps
                        velocity = module.denoise_step(pad, cache, x, time.expand(b))
                        x = x + dt * velocity
                        time += dt
                    return x
                finally:
                    mask_state.clear()

            p.bind(model, "sample_actions", sample)
        if "condition_cache" in config.enabled:
            if "flow_loop" not in config.enabled:
                raise ValueError("condition_cache requires flow_loop")
            condition_values = {}
            p.set(model, "_robotics_flow_step", 0)
            p.set(model, "_robotics_flow_steps", 0)
            original_suffix = model.embed_suffix

            def cached_suffix(module, actions, timestep):
                key = (
                    module._robotics_flow_steps,
                    module._robotics_flow_step,
                    actions.shape[0],
                    actions.device,
                    actions.dtype,
                )
                if key not in condition_values:
                    result = original_suffix(actions, timestep)
                    condition_values[key] = result[3]
                    return result
                hidden = module.action_in_proj(actions)
                padding = torch.ones(
                    hidden.shape[:2], dtype=torch.bool, device=hidden.device
                )
                first = torch.ones(
                    (hidden.shape[0], 1), dtype=hidden.dtype, device=hidden.device
                )
                rest = torch.zeros(
                    (hidden.shape[0], module.config.chunk_size - 1),
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                return (
                    hidden,
                    padding,
                    torch.cat((first, rest), dim=1),
                    condition_values[key],
                )

            p.bind(model, "embed_suffix", cached_suffix)
        if "cuda_graph" in config.enabled:
            # Native torch.tensor(Python-list, device='cuda') performs a host
            # transfer that cannot be captured. Build identical masks on device.
            image_cache = {}
            image_cache_signature = None

            def prefix(module, images, img_masks, tokens, masks):
                import math

                nonlocal image_cache_signature
                padded = (
                    tuple(getattr(module, "_robotics_padded_cameras", ()))
                    if "empty_image_cache" in config.enabled
                    else ()
                )
                signature = (
                    tuple((tuple(x.shape), x.dtype, x.device) for x in images),
                    padded,
                )
                if signature != image_cache_signature:
                    image_cache.clear()
                    image_cache_signature = signature
                embeddings = []
                padding = []
                for index, (image, valid) in enumerate(
                    zip(images, img_masks, strict=True)
                ):
                    if index in padded and index in image_cache:
                        encoded = image_cache[index]
                    else:
                        encoded = module.paligemma_with_expert.embed_image(image)
                        if index in padded:
                            image_cache[index] = encoded

                    embeddings.append(encoded)
                    padding.append(
                        valid[:, None].expand(encoded.shape[0], encoded.shape[1])
                    )
                language = module.paligemma_with_expert.embed_language_tokens(tokens)
                embeddings.append(language * math.sqrt(language.shape[-1]))
                padding.append(masks)
                hidden = torch.cat(embeddings, dim=1)
                pad = torch.cat(padding, dim=1)
                att = torch.zeros(pad.shape, dtype=torch.bool, device=pad.device)
                return hidden, pad, att

            def suffix(module, actions, timestep):
                cache_key = (
                    (
                        module._robotics_flow_steps,
                        module._robotics_flow_step,
                        actions.shape[0],
                        actions.device,
                        actions.dtype,
                    )
                    if "condition_cache" in config.enabled
                    else None
                )
                if cache_key is not None and cache_key in condition_values:
                    hidden = module.action_in_proj(actions)
                    condition = condition_values[cache_key]
                else:
                    time_embedding = native.create_sinusoidal_pos_embedding(
                        timestep,
                        module.action_in_proj.out_features,
                        min_period=module.config.min_period,
                        max_period=module.config.max_period,
                        device=timestep.device,
                    ).to(timestep.dtype)
                    hidden = module.action_in_proj(actions)
                    condition = torch.nn.functional.silu(
                        module.time_mlp_out(
                            torch.nn.functional.silu(module.time_mlp_in(time_embedding))
                        )
                    )
                    if cache_key is not None:
                        condition_values[cache_key] = condition
                pad = torch.ones(
                    hidden.shape[:2], dtype=torch.bool, device=hidden.device
                )
                first = torch.ones(
                    (hidden.shape[0], 1), dtype=hidden.dtype, device=hidden.device
                )
                rest = torch.zeros(
                    (hidden.shape[0], module.config.chunk_size - 1),
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                return hidden, pad, torch.cat((first, rest), dim=1), condition

            p.bind(model, "embed_prefix", prefix)
            p.bind(model, "embed_suffix", suffix)
            from robotics_kernels.graph import CudaGraphCall

            eager = model.sample_actions
            image_count = None
            active_steps = None
            active_padded = None
            graph_call = None

            @torch.no_grad()
            def graphed(
                module, images, img_masks, tokens, masks, noise=None, num_steps=None
            ):
                nonlocal image_count, active_steps, active_padded, graph_call
                steps = (
                    module.config.num_inference_steps
                    if num_steps is None
                    else num_steps
                )
                if noise is None:
                    noise = module.sample_noise(
                        (
                            tokens.shape[0],
                            module.config.chunk_size,
                            module.config.max_action_dim,
                        ),
                        tokens.device,
                    )
                count = len(images)
                padded = tuple(getattr(module, "_robotics_padded_cameras", ()))
                if (
                    graph_call is None
                    or count != image_count
                    or steps != active_steps
                    or padded != active_padded
                ):
                    image_count, active_steps, active_padded = count, steps, padded

                    def execute(*values):
                        return eager(
                            list(values[:count]),
                            list(values[count : 2 * count]),
                            values[-3],
                            values[-2],
                            noise=values[-1],
                            num_steps=steps,
                        )

                    graph_call = CudaGraphCall(
                        execute, retained_tensors=lambda: tuple(image_cache.values())
                    )
                result = graph_call(*images, *img_masks, tokens, masks, noise)
                report["graph"] = {
                    "captures": graph_call.captures,
                    "replays": graph_call.replays,
                }
                return result

            p.bind(model, "sample_actions", graphed)
        yield report
    finally:
        stack.close()
        p.restore()
