"""Checkpoint load safety checks without importing torch or model libraries."""

from collections import namedtuple
import importlib.util
import inspect
import json
import math
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1] / "benchmarks/static/pi05_libero/load_guard.py"
)
LoadResult = namedtuple("LoadResult", "missing_keys unexpected_keys")


def api():
    assert SOURCE.is_file(), "checkpoint load guard is not implemented"
    spec = importlib.util.spec_from_file_location("tested_load_guard", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeParameter:
    def __init__(self, shape=(2,)):
        self.shape = shape


class FakePolicy:
    """Model a loader that may swallow a load error and return its instance."""

    def __init__(self, *, tied=False, buffer=False):
        weight = FakeParameter()
        self.parameters = {"weight": weight}
        if tied:
            self.parameters["alias"] = weight
        self.buffers = {"running_stat": FakeParameter()} if buffer else {}
        self.strict_received = None

    def named_parameters(self, *, remove_duplicate=True):
        assert remove_duplicate is False
        return self.parameters.items()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.strict_received = strict
        self.assign_received = assign
        expected = self.parameters | self.buffers
        for name in state_dict.keys() & expected.keys():
            if state_dict[name].shape != expected[name].shape:
                raise ValueError("size mismatch for " + name)
        result = LoadResult(
            sorted(expected.keys() - state_dict.keys()),
            sorted(state_dict.keys() - expected.keys()),
        )
        if strict and (result.missing_keys or result.unexpected_keys):
            raise RuntimeError("strict loading failed")
        return result

    @classmethod
    def from_pretrained(
        cls,
        state_dict,
        *,
        tied=False,
        buffer=False,
        swallow=False,
        skip_load=False,
        assign=False,
    ):
        policy = cls(tied=tied, buffer=buffer)
        if not skip_load:
            try:
                policy.load_state_dict(state_dict, True, assign=assign)
            except Exception:
                if not swallow:
                    raise
        return policy


def read_audit(path):
    return json.loads(path.read_text())


def test_complete_load_forces_non_strict_and_forwards_assign(tmp_path):
    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(FakePolicy, path):
        policy = FakePolicy.from_pretrained({"weight": FakeParameter()}, assign=True)
    assert policy.strict_received is False
    assert policy.assign_received is True
    report = read_audit(path)
    assert report["class"].endswith("FakePolicy")
    assert report["provided_keys"] == ["weight"]
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["missing_parameters"] == []
    assert report["covered_tied_aliases"] == {}
    assert report["status"] == "passed"
    assert "parameter_dtype_numel" not in report


@pytest.mark.parametrize("provided,missing", [("weight", "alias"), ("alias", "weight")])
def test_loaded_tied_alias_covers_same_parameter_object(tmp_path, provided, missing):
    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(FakePolicy, path):
        FakePolicy.from_pretrained({provided: FakeParameter()}, tied=True)
    report = read_audit(path)
    assert report["missing_keys"] == [missing]
    assert report["missing_parameters"] == []
    assert report["covered_tied_aliases"] == {missing: [provided]}


def test_equal_shape_but_distinct_parameters_are_not_tied(tmp_path):
    class SeparatePolicy(FakePolicy):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.parameters["other"] = FakeParameter()

    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(SeparatePolicy, path):
        with pytest.raises(RuntimeError, match="missing parameters"):
            SeparatePolicy.from_pretrained({"weight": FakeParameter()}, swallow=True)
    assert read_audit(path)["missing_parameters"] == ["other"]


def test_uncovered_parameter_rejected_even_if_loader_omits_missing_report(tmp_path):
    class LyingPolicy(FakePolicy):
        def load_state_dict(self, state_dict, strict=True, assign=False):
            return LoadResult([], [])

    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(LyingPolicy, path):
        with pytest.raises(RuntimeError, match="missing parameters"):
            LyingPolicy.from_pretrained({})
    assert read_audit(path)["missing_parameters"] == ["weight"]


@pytest.mark.parametrize("missing", ["running_stat", "unknown_loader_key"])
def test_missing_buffer_or_unknown_key_is_rejected(tmp_path, missing):
    class MissingKeyPolicy(FakePolicy):
        def load_state_dict(self, state_dict, strict=True, assign=False):
            return LoadResult([missing], [])

    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(MissingKeyPolicy, path):
        with pytest.raises(RuntimeError, match="missing keys"):
            MissingKeyPolicy.from_pretrained({"weight": FakeParameter()})
    assert read_audit(path)["missing_keys"] == [missing]
    assert read_audit(path)["status"] == "failed"


def test_actual_missing_buffer_rejected(tmp_path):
    with api().checkpoint_load_guard(FakePolicy, tmp_path / "audit.json"):
        with pytest.raises(RuntimeError, match="missing keys"):
            FakePolicy.from_pretrained({"weight": FakeParameter()}, buffer=True)


def test_unexpected_key_rejected_after_loader_returns(tmp_path):
    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(FakePolicy, path):
        with pytest.raises(RuntimeError, match="unexpected keys"):
            FakePolicy.from_pretrained(
                {"weight": FakeParameter(), "extra": FakeParameter()}, swallow=True
            )
    assert read_audit(path)["unexpected_keys"] == ["extra"]


def test_swallowed_shape_error_is_rejected_by_outer_guard(tmp_path):
    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(FakePolicy, path):
        with pytest.raises(RuntimeError, match="load_state_dict failed"):
            FakePolicy.from_pretrained({"weight": FakeParameter((3,))}, swallow=True)
    report = read_audit(path)
    assert report["status"] == "failed"
    assert "size mismatch" in report["load_errors"][0]


def test_unswallowed_shape_error_keeps_original_exception_and_audit(tmp_path):
    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(FakePolicy, path):
        with pytest.raises(ValueError, match="size mismatch"):
            FakePolicy.from_pretrained({"weight": FakeParameter((3,))})
    assert read_audit(path)["status"] == "failed"


def test_no_load_call_rejected_even_after_previous_success(tmp_path):
    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(FakePolicy, path):
        FakePolicy.from_pretrained({"weight": FakeParameter()})
        with pytest.raises(RuntimeError, match="without.*load_state_dict"):
            FakePolicy.from_pretrained({}, skip_load=True)
    assert read_audit(path)["status"] == "failed"
    assert read_audit(path)["provided_keys"] == []


def test_loading_a_different_instance_does_not_cover_returned_model(tmp_path):
    class WrongInstancePolicy(FakePolicy):
        @classmethod
        def from_pretrained(cls, state_dict):
            cls().load_state_dict(state_dict)
            return cls()

    with api().checkpoint_load_guard(WrongInstancePolicy, tmp_path / "audit.json"):
        with pytest.raises(RuntimeError, match="without.*load_state_dict"):
            WrongInstancePolicy.from_pretrained({"weight": FakeParameter()})


def test_local_method_descriptors_restored_after_failure(tmp_path):
    before = {
        name: inspect.getattr_static(FakePolicy, name)
        for name in ("from_pretrained", "load_state_dict")
    }
    with pytest.raises(RuntimeError):
        with api().checkpoint_load_guard(FakePolicy, tmp_path / "audit.json"):
            FakePolicy.from_pretrained({})
    for name, descriptor in before.items():
        assert inspect.getattr_static(FakePolicy, name) is descriptor
    assert isinstance(FakePolicy.__dict__["from_pretrained"], classmethod)


def test_inherited_descriptors_are_removed_on_exit_and_subclass_binding_kept(tmp_path):
    class ChildPolicy(FakePolicy):
        pass

    inherited = inspect.getattr_static(ChildPolicy, "from_pretrained")
    with api().checkpoint_load_guard(ChildPolicy, tmp_path / "nested" / "audit.json"):
        instance = ChildPolicy.from_pretrained({"weight": FakeParameter()})
        assert type(instance) is ChildPolicy
        assert "from_pretrained" in ChildPolicy.__dict__
    assert "from_pretrained" not in ChildPolicy.__dict__
    assert "load_state_dict" not in ChildPolicy.__dict__
    assert inspect.getattr_static(ChildPolicy, "from_pretrained") is inherited


def test_restores_descriptors_when_context_body_raises(tmp_path):
    before = dict(FakePolicy.__dict__)
    with pytest.raises(LookupError):
        with api().checkpoint_load_guard(FakePolicy, tmp_path / "audit.json"):
            raise LookupError("caller failed")
    assert FakePolicy.__dict__["from_pretrained"] is before["from_pretrained"]
    assert FakePolicy.__dict__["load_state_dict"] is before["load_state_dict"]


def test_provided_key_reported_missing_does_not_cover_parameter(tmp_path):
    class RejectedKeyPolicy(FakePolicy):
        def load_state_dict(self, state_dict, strict=True, assign=False):
            return LoadResult(["weight"], [])

    with api().checkpoint_load_guard(RejectedKeyPolicy, tmp_path / "audit.json"):
        with pytest.raises(RuntimeError, match="missing parameters"):
            RejectedKeyPolicy.from_pretrained({"weight": FakeParameter()})


def test_multiple_successful_loads_cover_one_returned_instance(tmp_path):
    class PartitionedPolicy(FakePolicy):
        @classmethod
        def from_pretrained(cls, state_dict):
            instance = cls(buffer=True)
            instance.load_state_dict({"weight": state_dict["weight"]})
            instance.load_state_dict({"running_stat": state_dict["running_stat"]})
            return instance

    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(PartitionedPolicy, path):
        PartitionedPolicy.from_pretrained(
            {"weight": FakeParameter(), "running_stat": FakeParameter()}
        )
    report = read_audit(path)
    assert report["status"] == "passed"
    assert report["provided_keys"] == ["running_stat", "weight"]
    assert report["unresolved_missing_keys"] == []
    assert report["load_call_count"] == 2


def test_success_after_swallowed_load_error_does_not_erase_error(tmp_path):
    class RetryingPolicy(FakePolicy):
        @classmethod
        def from_pretrained(cls, state_dict):
            instance = cls()
            try:
                instance.load_state_dict({"weight": FakeParameter((3,))})
            except ValueError:
                pass
            instance.load_state_dict(state_dict)
            return instance

    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(RetryingPolicy, path):
        with pytest.raises(RuntimeError, match="load_state_dict failed"):
            RetryingPolicy.from_pretrained({"weight": FakeParameter()})
    assert read_audit(path)["load_call_count"] == 2
    assert len(read_audit(path)["load_errors"]) == 1


def test_dtype_numel_counts_parameter_objects_once_per_dtype(tmp_path):
    class SizedParameter(FakeParameter):
        def __init__(self, dtype, shape=(2, 3)):
            super().__init__(shape)
            self.dtype = dtype

        def numel(self):
            return math.prod(self.shape)

    class TypedPolicy(FakePolicy):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            weight = SizedParameter("torch.bfloat16")
            self.parameters = {
                "weight": weight,
                "tied_alias": weight,
                "other": SizedParameter("torch.bfloat16"),
                "bias": SizedParameter("torch.float32", (3,)),
            }

    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(TypedPolicy, path):
        TypedPolicy.from_pretrained(
            {
                "weight": FakeParameter((2, 3)),
                "other": FakeParameter((2, 3)),
                "bias": FakeParameter((3,)),
            }
        )
    assert read_audit(path)["parameter_dtype_numel"] == {
        "torch.bfloat16": 12,
        "torch.float32": 3,
    }


def test_incomplete_dtype_metadata_does_not_report_partial_totals(tmp_path):
    class PartialMetadataPolicy(FakePolicy):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.parameters["weight"].dtype = "torch.float32"

    path = tmp_path / "audit.json"
    with api().checkpoint_load_guard(PartialMetadataPolicy, path):
        PartialMetadataPolicy.from_pretrained({"weight": FakeParameter()})
    assert "parameter_dtype_numel" not in read_audit(path)


def test_context_exposes_only_models_that_passed_audit(tmp_path):
    with api().checkpoint_load_guard(FakePolicy, tmp_path / "audit.json") as models:
        assert models == []
        first = FakePolicy.from_pretrained({"weight": FakeParameter()})
        assert models == [first]
        with pytest.raises(RuntimeError, match="missing parameters"):
            FakePolicy.from_pretrained({}, swallow=True)
        assert models == [first]
        second = FakePolicy.from_pretrained({"weight": FakeParameter()})
        assert models == [first, second]
    assert models == [first, second]


def test_swallowed_load_failure_does_not_expose_rejected_model(tmp_path):
    with api().checkpoint_load_guard(FakePolicy, tmp_path / "audit.json") as models:
        with pytest.raises(RuntimeError, match="load_state_dict failed"):
            FakePolicy.from_pretrained({"weight": FakeParameter((3,))}, swallow=True)
        assert models == []
