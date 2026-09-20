"""Bind loaded weights to the repository-owned Cosmos execution classes.

Model construction and dependency configuration are still supplied by the native
loader. Switching Python classes preserves all parameters and buffers and avoids
constructing a second multi-gigabyte model. Only known reviewed class families
are migrated; common Torch/TransformerEngine/VAE library operators remain intact.
"""

from importlib import import_module


MODULES = {
    "cosmos_policy._src.predict2.networks.minimal_v4_dit": "minimal_v4_dit",
    "cosmos_policy._src.predict2.networks.minimal_v1_lvg_dit": "minimal_v1_lvg_dit",
    "cosmos_policy.models.policy_video2world_model": "policy_video2world_model",
    "cosmos_policy.models.policy_text2world_model": "policy_text2world_model",
    "cosmos_policy.modules.cosmos_sampler": "cosmos_sampler",
    "cosmos_policy.modules.hybrid_edm_sde": "hybrid_edm_sde",
}


def bind_owned_runtime(model):
    modules = {
        name: import_module("." + local, __package__) for name, local in MODULES.items()
    }
    objects = list(model.named_modules())
    for name in ("sampler", "sde"):
        item = getattr(model, name, None)
        if item is not None and all(obj is not item for _, obj in objects):
            objects.append((name, item))
    changed = []
    try:
        for name, obj in objects:
            previous = type(obj)
            if previous.__module__ not in modules:
                continue
            owned = getattr(modules[previous.__module__], previous.__name__, None)
            if not isinstance(owned, type):
                raise RuntimeError(f"Unsupported Cosmos execution class: {previous}")
            obj.__class__ = owned
            changed.append((name, obj, previous))
        if not any(name == "net" for name, _, _ in changed):
            raise RuntimeError("Expected a supported Cosmos net for local migration")
    except BaseException:
        for _, obj, previous in reversed(changed):
            obj.__class__ = previous
        raise
    report = [
        {
            "module": name,
            "previous": old.__module__ + "." + old.__name__,
            "current": type(obj).__module__ + "." + type(obj).__name__,
        }
        for name, obj, old in changed
    ]
    return report
