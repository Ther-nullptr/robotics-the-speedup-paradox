"""Reject incomplete checkpoint loads before a policy can reach the rollout.

This module uses Python's method protocol only; importing it does not require
torch. The guard is intended for serialized model loading in an isolated worker,
because it temporarily patches methods on the supplied policy class.
"""

from contextlib import contextmanager
import inspect
import json
from pathlib import Path


def _parameter_dtype_numel(parameters):
    """Read metadata only; omit totals if any parameter lacks that protocol."""
    counts = {}
    for parameter in parameters:
        dtype = getattr(parameter, "dtype", None)
        numel = getattr(parameter, "numel", None)
        if dtype is None or not callable(numel):
            return None
        name = str(dtype)
        counts[name] = counts.get(name, 0) + numel()
    return dict(sorted(counts.items()))


def _summarize(policy_cls, instance, calls):
    """Account for successful loads into the instance actually being returned."""
    calls = [call for call in calls if call["instance"] is instance]
    provided = set()
    missing = set()
    unexpected = set()
    loaded = set()
    errors = []
    for call in calls:
        provided.update(call["provided_keys"])
        missing.update(call["missing_keys"])
        unexpected.update(call["unexpected_keys"])
        if call["error"] is not None:
            errors.append(call["error"])
        else:
            loaded.update(
                set(call["provided_keys"])
                - set(call["missing_keys"])
                - set(call["unexpected_keys"])
            )

    aliases = {}
    unique_parameters = {}
    if instance is not None:
        for name, parameter in instance.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
            unique_parameters[id(parameter)] = parameter
    parameter_names = {name for group in aliases.values() for name in group}
    missing_parameters = []
    covered_tied_aliases = {}
    for group in aliases.values():
        covered = sorted(set(group) & loaded)
        if not covered:
            missing_parameters.extend(group)
        else:
            for name in set(group) & missing - loaded:
                covered_tied_aliases[name] = covered

    # Missing keys outside named_parameters include persistent buffers and any
    # unknown names reported by the loader. They cannot borrow tied coverage.
    unresolved_missing = sorted(missing - parameter_names - loaded)
    report = {
        "class": f"{policy_cls.__module__}.{policy_cls.__qualname__}",
        "provided_keys": sorted(provided),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "missing_parameters": sorted(missing_parameters),
        "covered_tied_aliases": dict(sorted(covered_tied_aliases.items())),
        "unresolved_missing_keys": unresolved_missing,
        "load_errors": errors,
        "load_call_count": len(calls),
    }
    if instance is not None:
        dtype_numel = _parameter_dtype_numel(unique_parameters.values())
        if dtype_numel is not None:
            report["parameter_dtype_numel"] = dtype_numel
    failures = []
    if not calls:
        failures.append(
            "checkpoint loader returned without load_state_dict on the returned instance"
        )
    if errors:
        failures.append("load_state_dict failed: " + "; ".join(errors))
    if missing_parameters:
        failures.append("missing parameters: " + ", ".join(sorted(missing_parameters)))
    if unresolved_missing:
        failures.append("missing keys: " + ", ".join(unresolved_missing))
    if unexpected:
        failures.append("unexpected keys: " + ", ".join(sorted(unexpected)))
    return report, failures


def _write_audit(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


@contextmanager
def checkpoint_load_guard(policy_cls, audit_path: Path):
    """Audit ``policy_cls.from_pretrained`` and reject incomplete loaded models.

    ``from_pretrained`` must be a classmethod. Calls made by that loader to
    ``load_state_dict`` are forced to ``strict=False`` so missing tied aliases
    can be checked by parameter identity. Shape/load errors remain failures even
    when the original loader catches them. Every returned parameter must have a
    successfully supplied key, either its own or an alias of the same object.
    Missing buffers/unknown keys and unexpected keys are rejected.

    The JSON audit describes the latest attempted ``from_pretrained`` call.
    When all parameters expose ``dtype`` and ``numel()``,
    ``parameter_dtype_numel`` sums their elements by dtype, counting each shared
    parameter object once. It excludes buffers and does not read tensor values.
    Loading multiple state dictionaries into that returned instance is accounted
    for together. A load into another instance cannot satisfy the guard. This is
    a completeness check, not a numerical/checkpoint provenance validation.

    Both original descriptors are restored on every context exit. Inherited
    methods remain inherited, rather than becoming bound instance attributes.
    Do not overlap this context with concurrent loading on the patched class.

    Yields a list containing only instances returned after successful auditing.
    Callers may inspect those instances after evaluation; failed loads are never
    appended. Existing callers that do not bind the yielded value are unchanged.
    """
    audit_path = Path(audit_path)
    original_pretrained = inspect.getattr_static(policy_cls, "from_pretrained")
    original_load = inspect.getattr_static(policy_cls, "load_state_dict")
    if not isinstance(original_pretrained, classmethod):
        raise TypeError("policy_cls.from_pretrained must be a classmethod")

    absent = object()
    local_descriptors = {
        name: policy_cls.__dict__.get(name, absent)
        for name in ("from_pretrained", "load_state_dict")
    }
    calls = []
    loaded_models = []

    def guarded_load(instance, state_dict, strict=True, *args, **kwargs):
        call = {
            "instance": instance,
            "provided_keys": list(state_dict),
            "missing_keys": [],
            "unexpected_keys": [],
            "error": None,
        }
        calls.append(call)
        try:
            bound_load = original_load.__get__(instance, type(instance))
            result = bound_load(state_dict, False, *args, **kwargs)
            call["missing_keys"] = list(result.missing_keys)
            call["unexpected_keys"] = list(result.unexpected_keys)
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        return result

    def guarded_pretrained(cls, *args, **kwargs):
        start = len(calls)
        try:
            instance = original_pretrained.__get__(None, cls)(*args, **kwargs)
        except Exception as exc:
            attempt = calls[start:]
            instance = attempt[-1]["instance"] if attempt else None
            report, _ = _summarize(cls, instance, attempt)
            report["status"] = "failed"
            report["loader_error"] = f"{type(exc).__name__}: {exc}"
            _write_audit(audit_path, report)
            raise

        report, failures = _summarize(cls, instance, calls[start:])
        report["status"] = "failed" if failures else "passed"
        _write_audit(audit_path, report)
        if failures:
            raise RuntimeError("Checkpoint load rejected: " + "; ".join(failures))
        loaded_models.append(instance)
        return instance

    try:
        policy_cls.from_pretrained = classmethod(guarded_pretrained)
        policy_cls.load_state_dict = guarded_load
        yield loaded_models
    finally:
        for name, descriptor in local_descriptors.items():
            if descriptor is absent:
                delattr(policy_cls, name)
            else:
                setattr(policy_cls, name, descriptor)
