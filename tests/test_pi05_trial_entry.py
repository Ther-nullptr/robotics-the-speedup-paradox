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
    assert plan["identity"]["case"]["video_episodes_per_task"] == 0


def test_video_limit_can_be_set_directly_without_a_json_preset(inputs):
    result = invoke(
        inputs, "--video-episodes-per-task", "2", "--video-fps", "20", "--dry-run"
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["identity"]["case"]["video_episodes_per_task"] == 2
    assert plan["video"]["enabled"] is True
    assert plan["video"]["max_episodes_per_task"] == 2
    assert plan["video"]["fps"] == 20
    assert plan["video"]["directory"] == str(inputs["output_dir"] / "videos")
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_video_limit_rejects_invalid_values(inputs, value):
    inputs["config"].write_text(json.dumps({"video_episodes_per_task": value}))
    result = invoke(inputs, "--dry-run")
    assert result.returncode != 0
    assert "video_episodes_per_task" in result.stderr


def test_cli_can_disable_video_from_json(inputs):
    inputs["config"].write_text(json.dumps({"video_episodes_per_task": 3}))
    result = invoke(inputs, "--video-episodes-per-task", "0", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["video"]["enabled"] is False


def test_record_video_selects_every_evaluated_episode(inputs):
    result = invoke(inputs, "--record-video", "--episodes", "500", "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["video"]["record_all_episodes"] is True
    assert plan["video"]["max_episodes_per_task"] == 500
    assert plan["identity"]["case"]["video_episodes_per_task"] == "all"


def test_no_record_video_overrides_an_all_video_preset(inputs):
    inputs["config"].write_text('{"video_episodes_per_task":"all"}')
    result = invoke(inputs, "--no-record-video", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["video"]["enabled"] is False


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_video_fps_rejects_invalid_values(inputs, value):
    inputs["config"].write_text(json.dumps({"video_fps": value}))
    result = invoke(inputs, "--dry-run")
    assert result.returncode != 0
    assert "video_fps" in result.stderr


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


def test_legacy_delay_selects_paper_async(inputs):
    inputs["config"].write_text(json.dumps({"async_delay": 2}))
    result = invoke(inputs, "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schedule"] == "paper_async"
    assert plan["identity"]["case"]["overlap_actions"] == 2
    assert "async_delay" not in plan["identity"]["case"]
    assert "--eval.async_delay=2" in plan["upstream_arguments"]


@pytest.mark.parametrize("overlap", [0, 2, 5])
def test_paper_async_maps_n_prime_to_primitive_history_steps(inputs, overlap):
    inputs["config"].write_text(
        json.dumps({"schedule": "paper_async", "overlap_actions": overlap})
    )
    result = invoke(inputs, "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    contract = plan["paper_async"]
    assert plan["schedule"] == "paper_async"
    assert contract["n_prime"] == overlap
    assert contract["n_actions"] == 5
    assert contract["state_policy"] == "same_snapshot"
    assert contract["history_fallback"] == "current_until_available"
    assert contract["timing"]["status"] == "requires_inference_profile"
    assert contract["timing"]["cycle_time_ms"] is None
    assert contract["timing"]["action_time_ms"] == pytest.approx(1000 / 30)
    assert f"--eval.async_delay={overlap}" in plan["upstream_arguments"]
    assert "--eval.action_quant=1" in plan["upstream_arguments"]
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "options",
    [
        {"schedule": "sync", "overlap_actions": 2},
        {"schedule": "sync", "async_delay": 2},
        {"overlap_actions": 2, "async_delay": 3},
        {"overlap_actions": 2, "async_delay": True},
        {"overlap_actions": True, "async_delay": 1},
        {"overlap_actions": 6},  # Bound by executed n=5, even with predicted H=50.
        {"async_delay": 6},
        {"overlap_actions": True},
        {"schedule": "async"},
        {"paper_inference_time_ms": 100},
        {"paper_action_time_ms": 0},
    ],
)
def test_invalid_paper_async_options_fail_before_execution(inputs, options):
    inputs["config"].write_text(json.dumps(options))
    result = invoke(inputs, "--dry-run")
    assert result.returncode != 0
    assert not inputs["output_dir"].exists()


def test_overlap_alias_and_canonical_config_have_same_identity(inputs):
    plans = []
    for options in ({"overlap_actions": 2}, {"async_delay": 2}):
        inputs["config"].write_text(json.dumps(options))
        result = invoke(inputs, "--dry-run")
        assert result.returncode == 0, result.stderr
        plans.append(json.loads(result.stdout))
    assert plans[0]["case_fingerprint"] == plans[1]["case_fingerprint"]


def test_paper_cycle_estimate_requires_explicit_latency_provenance(inputs):
    inputs["config"].write_text(
        json.dumps(
            {
                "schedule": "paper_async",
                "overlap_actions": 2,
                "paper_action_time_ms": 20,
                "paper_inference_time_ms": 100,
                "paper_inference_time_source": "synthetic unit-test fixture",
            }
        )
    )
    result = invoke(inputs, "--dry-run")
    assert result.returncode == 0, result.stderr
    timing = json.loads(result.stdout)["paper_async"]["timing"]
    assert timing["clock_domain"] == "paper_model"
    assert timing["cycle_time_ms"] == 160
    assert timing["residual_inference_time_ms"] == 60
    assert timing["inference_time_source"] == "synthetic unit-test fixture"


def test_cli_runs_paper_async_without_a_case_file(inputs):
    inputs.pop("config")
    result = invoke(
        inputs,
        "--schedule",
        "paper_async",
        "--overlap-actions",
        "2",
        "--episodes",
        "3",
        "--seed",
        "7",
        "--suite",
        "libero_goal",
        "--n-action-steps",
        "4",
        "--num-inference-steps",
        "6",
        "--quant",
        "none",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schedule"] == "paper_async"
    assert plan["identity"]["case"]["episodes"] == 3
    assert plan["identity"]["case"]["seed"] == 7
    assert "--eval.async_delay=2" in plan["upstream_arguments"]
    assert "--env.task=libero_goal" in plan["upstream_arguments"]
    assert "--policy.n_action_steps=4" in plan["upstream_arguments"]
    assert "--policy.num_inference_steps=6" in plan["upstream_arguments"]
    assert not inputs["output_dir"].exists()


def test_cli_overrides_json_and_legacy_delay(inputs):
    inputs["config"].write_text(json.dumps({"episodes": 4, "async_delay": 2}))
    result = invoke(inputs, "--episodes", "2", "--overlap-actions", "3", "--dry-run")
    assert result.returncode == 0, result.stderr
    options = json.loads(result.stdout)["identity"]["case"]
    assert options["episodes"] == 2
    assert options["overlap_actions"] == 3
    assert "async_delay" not in options


def test_cli_can_explicitly_turn_off_json_compile_and_quantization(inputs):
    inputs["config"].write_text(
        json.dumps(
            {
                "compile_model": True,
                "quant_ladder": "fp16_to_w8a8",
                "quant_selected_profile": "old-profile",
            }
        )
    )
    result = invoke(inputs, "--no-compile-model", "--quant", "none", "--dry-run")
    assert result.returncode == 0, result.stderr
    options = json.loads(result.stdout)["identity"]["case"]
    assert options["compile_model"] is False
    assert options["quant_ladder"] == "none"
    assert options["quant_selected_profile"] is None


@pytest.mark.parametrize(
    "option,value",
    [
        ("--overlap-actions", "6"),
        ("--episodes", "0"),
        ("--batch-size", "2"),
        ("--quant", "w8a8"),
        ("--paper-inference-time-ms", "100"),
    ],
)
def test_cli_invalid_experiments_fail_preflight(inputs, option, value):
    result = invoke(inputs, option, value, "--dry-run")
    assert result.returncode != 0
    assert not inputs["output_dir"].exists()


def test_environment_paths_work_and_explicit_path_wins(inputs, monkeypatch):
    cli_inputs = {"output_dir": inputs["output_dir"]}
    for name in ("sim_source", "checkpoint", "tokenizer", "libero_config_dir"):
        monkeypatch.setenv("ROBOTICS_" + name.upper(), str(inputs[name]))
    monkeypatch.setenv(
        "ROBOTICS_CHECKPOINT", str(inputs["checkpoint"].parent / "missing")
    )
    result = invoke(cli_inputs, "--checkpoint", str(inputs["checkpoint"]), "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["checkpoint"] == str(inputs["checkpoint"])
    assert plan["sources"]["sim"]["path"] == str(inputs["sim_source"])
    assert not inputs["output_dir"].exists()


def test_removed_quant_preset_is_rejected_by_cli(inputs):
    result = invoke(inputs, "--quant", "w8a8-single-layer", "--dry-run")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert not inputs["output_dir"].exists()


def test_cli_rejects_sync_with_positive_overlap(inputs):
    result = invoke(inputs, "--schedule", "sync", "--overlap-actions", "2", "--dry-run")
    assert result.returncode != 0
    assert "sync requires overlap_actions=0" in result.stderr


def test_cli_can_restore_sync_after_legacy_async_json(inputs):
    inputs["config"].write_text(
        json.dumps({"schedule": "paper_async", "async_delay": 2})
    )
    result = invoke(inputs, "--schedule", "sync", "--overlap-actions", "0", "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schedule"] == "sync"
    assert "--eval.async_delay=0" in plan["upstream_arguments"]


def test_quant_none_rejects_an_ambiguous_profile_override(inputs):
    result = invoke(
        inputs,
        "--quant",
        "none",
        "--quant-selected-profile",
        "custom",
        "--dry-run",
    )
    assert result.returncode != 0
    assert "profile" in result.stderr.lower()


def test_cli_profile_values_feed_the_shared_paper_model(inputs):
    result = invoke(
        inputs,
        "--overlap-actions",
        "2",
        "--paper-action-time-ms",
        "20",
        "--paper-inference-time-ms",
        "100",
        "--paper-inference-time-source",
        "synthetic CLI fixture",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schedule"] == "paper_async"
    assert plan["paper_async"]["timing"]["cycle_time_ms"] == 160


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
