"""Audit native Cosmos checkpoint loading without importing Torch.

Cosmos exports only net/net_ema in state_dict and its non-strict loader can
silently discard incompatible weights. Both the input and child load results
must therefore be checked. This guard is for the serial, single-model loader.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import re


def _module_type(module):
    return type(module).__module__ + "." + type(module).__name__


def _extra_state_compatibility(model):
    """Map only verified native metadata exceptions to state_dict key names."""
    modules = list(model.named_modules())
    wrapper_type = (
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.CheckpointWrapper"
    )
    wrapped_prefixes = sorted(
        (
            name + "._checkpoint_wrapped_module"
            for name, module in modules
            if _module_type(module) == wrapper_type
        ),
        key=len,
        reverse=True,
    )

    def canonical_key(key):
        # Torch's real CheckpointWrapper strips this component in state_dict,
        # but includes it in named_modules and load incompatibility reports.
        for prefix in wrapped_prefixes:
            if key.startswith(prefix + "."):
                key = prefix.rsplit(".", 1)[0] + key[len(prefix) :]
        return key

    details = {}
    operator_types = {}
    for raw_name, module in modules:
        name = canonical_key(raw_name)
        class_name = _module_type(module)
        reason = None
        if name and any(
            cls.__module__.startswith("transformer_engine.")
            for cls in type(module).__mro__
        ):
            reason = "TransformerEngine non-parameter FP8 extra state"
        if re.fullmatch(r"net\.blocks\.\d+\.cross_attn\.attn_op", name):
            operator_types[name] = class_name
            if (
                class_name
                == "cosmos_policy._src.predict2.networks.a2a_cp.MinimalA2AAttnOp"
                and not list(module.named_parameters())
                and not list(module.named_buffers())
                and not module.state_dict()
            ):
                reason = (
                    "Legacy TransformerEngine attention metadata on the verified "
                    "stateless MinimalA2AAttnOp (no parameters, buffers, or state)"
                )
        if reason is not None:
            details[name + "._extra_state"] = {
                "module_type": class_name,
                "module_path": raw_name,
                "reason": reason,
            }
    return details, operator_types, canonical_key


@contextmanager
def checkpoint_load_guard(loader, audit_path: Path):
    """Intercept the loader's model construction and fail on incomplete loads."""
    report = {
        "status": "loading",
        "load_calls": 0,
        "submodule_load_calls": {},
        "expected_keys": [],
        "provided_keys": [],
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "allowed_extra_state_keys": [],
        "allowed_extra_state_details": {},
        "attention_operator_types": {},
        "errors": [],
    }
    restores = []
    original_instantiate = loader.instantiate

    def fail(message):
        report["errors"].append(message)
        raise RuntimeError("Cosmos checkpoint audit: " + message)

    def replace(instance, name, replacement):
        previous = vars(instance).get(name)
        existed = name in vars(instance)
        restores.append((instance, name, previous, existed))
        setattr(instance, name, replacement)

    def audited_instantiate(*args, **kwargs):
        model = original_instantiate(*args, **kwargs)
        if report.get("model_class"):
            fail("multiple model instances are unsupported")
        report["model_class"] = type(model).__module__ + "." + type(model).__name__
        details, operator_types, canonical_key = _extra_state_compatibility(model)
        allowed = set(details)
        report["allowed_extra_state_keys"] = sorted(allowed)
        report["allowed_extra_state_details"] = details
        report["attention_operator_types"] = operator_types
        original_load = model.load_state_dict

        def wrap_child(name, module):
            original_child_load = module.load_state_dict
            report["submodule_load_calls"][name] = 0

            def child_load(*load_args, **load_kwargs):
                report["submodule_load_calls"][name] += 1
                try:
                    result = original_child_load(*load_args, **load_kwargs)
                except Exception as exc:
                    report["errors"].append(f"{name}.load_state_dict failed: {exc}")
                    raise
                if not hasattr(result, "missing_keys") or not hasattr(
                    result, "unexpected_keys"
                ):
                    fail(f"{name}.load_state_dict returned no verifiable result")
                for field in ("missing_keys", "unexpected_keys"):
                    keys = [
                        canonical_key(name + "." + key)
                        for key in getattr(result, field)
                    ]
                    report[field] = sorted(set(report[field]) | set(keys))
                    invalid = sorted(set(keys) - allowed)
                    if invalid:
                        fail(f"{name} {field}: {invalid}")
                return result

            replace(module, "load_state_dict", child_load)

        for name in ("net", "net_ema"):
            module = getattr(model, name, None)
            if module is not None:
                wrap_child(name, module)

        def audited_load(state, *load_args, **load_kwargs):
            report["load_calls"] += 1
            if report["load_calls"] != 1:
                fail("expected exactly one model load_state_dict call")
            # Called after on_train_start, against the actual instantiated model.
            expected = model.state_dict()
            report["expected_keys"] = sorted(expected)
            report["provided_keys"] = sorted(state)
            report["missing_keys"] = sorted(set(expected) - set(state))
            report["unexpected_keys"] = sorted(set(state) - set(expected))
            for field in ("missing_keys", "unexpected_keys"):
                invalid = sorted(set(report[field]) - allowed)
                if invalid:
                    fail(f"{field}: {invalid}")
            for key in sorted(set(expected) & set(state) - allowed):
                expected_shape = getattr(expected[key], "shape", None)
                supplied_shape = getattr(state[key], "shape", None)
                if (
                    expected_shape is None
                    or supplied_shape is None
                    or tuple(expected_shape) != tuple(supplied_shape)
                ):
                    report["shape_mismatches"].append(
                        {
                            "key": key,
                            "expected": list(expected_shape)
                            if expected_shape is not None
                            else None,
                            "provided": list(supplied_shape)
                            if supplied_shape is not None
                            else None,
                        }
                    )
            if report["shape_mismatches"]:
                fail(f"shape mismatches: {report['shape_mismatches']}")
            try:
                result = original_load(state, *load_args, **load_kwargs)
            except Exception as exc:
                report["errors"].append(f"model.load_state_dict failed: {exc}")
                raise
            for name, count in report["submodule_load_calls"].items():
                if any(key.startswith(name + ".") for key in expected) and count != 1:
                    fail(f"{name} checkpoint load was skipped or repeated")
            if not report["submodule_load_calls"].get("net"):
                fail("net checkpoint load was skipped")
            return result

        replace(model, "load_state_dict", audited_load)
        return model

    loader.instantiate = audited_instantiate
    try:
        yield report
        if report["load_calls"] != 1:
            fail("loader did not call model.load_state_dict exactly once")
        if report["errors"]:
            raise RuntimeError(
                "Cosmos checkpoint audit: " + "; ".join(report["errors"])
            )
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        loader.instantiate = original_instantiate
        for instance, name, previous, existed in reversed(restores):
            if existed:
                setattr(instance, name, previous)
            else:
                delattr(instance, name)
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(report, indent=2) + "\n")
