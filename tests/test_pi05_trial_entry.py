"""CPU-only checks for the external-checkout LIBERO trial entry."""

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ENTRY = Path(__file__).resolve().parents[1] / "benchmarks/static/pi05_libero/run.py"


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "sim"
    (source / "vlash").mkdir(parents=True)
    (source / "vlash/eval_libero.py").write_text(
        "raise AssertionError('dry-run imported simulator')\n"
        "class EvalConfig:\n"
        "    runtime_stack: str = 'lerobot'\n"
        "    async_delay: int = 0\n"
        "    delay_state_with_observation: bool = True\n"
        "    action_quant: int = 1\n"
        "    quant_ladder: str = 'none'\n"
        "def main(): pass\n"
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"type": "pi05", "chunk_size": 50})
    )
    (checkpoint / "model.safetensors").write_bytes(b"fixture, not real weights")
    for name in ["policy_preprocessor.json", "policy_postprocessor.json"]:
        (checkpoint / name).write_text('{"steps": []}')
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.model").write_bytes(b"fixture")
    libero_config = tmp_path / "libero-config"
    libero_config.mkdir()
    (libero_config / "config.yaml").write_text("{}\n")
    options = tmp_path / "case.json"
    options.write_text("{}")
    return {
        "sim_source": source,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
        "libero_config_dir": libero_config,
        "output_dir": tmp_path / "output",
        "config": options,
    }


def invoke(inputs=None, *extra):
    args = [sys.executable, str(ENTRY)]
    for name, value in (inputs or {}).items():
        args.extend(["--" + name.replace("_", "-"), str(value)])
    return subprocess.run(
        args + list(extra), text=True, capture_output=True, timeout=30
    )


def test_help_does_not_require_model_dependencies():
    result = invoke(None, "--help")
    assert result.returncode == 0, result.stderr
    assert "--sim-source" in result.stdout


def test_dry_run_does_not_import_sources_or_create_outputs(inputs):
    result = invoke(inputs, "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schedule"] == "sync"
    assert "--eval.runtime_stack=lerobot" in plan["upstream_arguments"]
    assert "--eval.action_quant=1" in plan["upstream_arguments"]
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "options",
    [
        {"episodes": 0},
        {"seed": True},
        {"batch_size": 2, "episodes": 1},
        {"n_action_steps": 51},
        {"unexpected_option": True},
        {"action_quant": 2},
        {"delay_state_with_observation": False},
    ],
)
def test_rejects_invalid_or_out_of_scope_options(inputs, options):
    inputs["config"].write_text(json.dumps(options))
    result = invoke(inputs, "--dry-run")
    assert result.returncode != 0
    assert not inputs["output_dir"].exists()


def test_historical_delay_is_not_reported_as_real_async(inputs):
    inputs["config"].write_text(json.dumps({"async_delay": 2}))
    result = invoke(inputs, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schedule"] == "history_observation"


def test_quantization_requires_explicit_dependencies(inputs):
    inputs["config"].write_text(json.dumps({"quant_ladder": "fp16_to_w8a8"}))
    result = invoke(inputs, "--dry-run")
    assert result.returncode != 0
    assert "quant" in result.stderr.lower()


def test_missing_processor_state_fails_preflight(inputs):
    (inputs["checkpoint"] / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"state_file": "missing.safetensors"}]})
    )
    result = invoke(inputs, "--dry-run")
    assert result.returncode != 0
    assert "missing.safetensors" in result.stderr


def test_identity_changes_when_weights_change(inputs):
    first = invoke(inputs, "--dry-run")
    assert first.returncode == 0, first.stderr
    (inputs["checkpoint"] / "model.safetensors").write_bytes(
        b"different fixture weights"
    )
    second = invoke(inputs, "--dry-run")
    assert second.returncode == 0, second.stderr
    assert (
        json.loads(first.stdout)["case_fingerprint"]
        != json.loads(second.stdout)["case_fingerprint"]
    )


def test_existing_output_is_not_reused(inputs):
    inputs["output_dir"].mkdir()
    marker = inputs["output_dir"] / "eval_results.json"
    marker.write_text("preserve me")
    result = invoke(inputs, "--gpu", "3")
    assert result.returncode != 0
    assert marker.read_text() == "preserve me"


def test_importing_entry_as_spawn_child_does_not_run_it(inputs):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; runpy.run_path(__import__('sys').argv[1], run_name='__mp_main__')",
            str(ENTRY),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout


@pytest.mark.parametrize(
    "profiles",
    [
        [{"name": "one", "w4_short_names": ["TXT.B00.mlp.down"]}],
        [{"name": "one", "w8_short_names": "TXT.B00.mlp.down"}],
        [{"name": "one", "w8_short_names": []}],
        [{"name": "one", "w8_short_names": [3]}],
        [{"name": "one", "w8_short_names": ["TXT.B00.mlp.down"]}] * 2,
    ],
)
def test_quant_profile_must_match_the_requested_ladder(inputs, profiles):
    root = inputs["config"].parent
    quant_source = root / "quant"
    (quant_source / "vlash/quantization").mkdir(parents=True)
    (quant_source / "vlash/quantization/runtime_apply.py").write_text("# fixture\n")
    kernel_source = root / "kernel"
    (kernel_source / "eval").mkdir(parents=True)
    (kernel_source / "eval/quant_linear.py").write_text("# fixture\n")
    profile = root / "quant.json"
    profile.write_text(json.dumps({"profiles": profiles}))
    inputs.update(
        quant_source=quant_source, kernel_source=kernel_source, quant_profile=profile
    )
    inputs["config"].write_text(
        json.dumps({"quant_ladder": "fp16_to_w8a8", "quant_selected_profile": "one"})
    )
    result = invoke(inputs, "--dry-run")
    assert result.returncode != 0
    assert "profile" in result.stderr.lower()


def test_quantized_run_cannot_accept_zero_matching_wrappers():
    spec = importlib.util.spec_from_file_location("pi05_trial_entry_audit", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Model:
        def named_modules(self):
            return [("linear", object())]

    assert hasattr(module, "audit_quantized_modules"), (
        "Runtime quantization audit is missing"
    )
    with pytest.raises(RuntimeError, match="w8a8"):
        module.audit_quantized_modules([Model()], "fp16_to_w8a8")


def test_quantization_audit_records_wrapper_paths():
    spec = importlib.util.spec_from_file_location("pi05_trial_entry_positive", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Quant:
        _vlash_qserve_scheme = "w4a4"

    class Model:
        def named_modules(self):
            return [("linear", Quant())]

    assert hasattr(module, "audit_quantized_modules"), (
        "Runtime quantization audit is missing"
    )
    assert module.audit_quantized_modules([Model()], "fp16_to_w4a4") == {
        "linear": "w4a4"
    }
