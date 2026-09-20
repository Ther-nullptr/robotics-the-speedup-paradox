"""CPU-only CLI checks; never import the optional quantization/GPU stack."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


CLI = (
    Path(__file__).resolve().parents[1] / "benchmarks/static/pi05_libero/quant_smoke.py"
)


def sources(tmp_path):
    quant = tmp_path / "quant source"
    backend = quant / "vlash/quantization/qserve_backend.py"
    backend.parent.mkdir(parents=True)
    backend.write_text("raise RuntimeError('backend must not load during dry-run')\n")
    kernel = tmp_path / "kernel source"
    (kernel / "mini_qserve_gemm").mkdir(parents=True)
    (kernel / "mini_qserve_gemm/__init__.py").write_text("")
    (kernel / "eval").mkdir()
    (kernel / "eval/quant_linear.py").write_text("")
    return quant, kernel


def run_cli(quant, kernel, output, *extra, env=None):
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--quant-source",
            str(quant),
            "--kernel-source",
            str(kernel),
            "--scheme",
            "w8a8",
            "--output-dir",
            str(output),
            "--gpu",
            "3",
            *extra,
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_module_import_has_no_optional_dependencies():
    assert CLI.is_file(), "quant_smoke.py is not implemented"
    script = """
import runpy
import sys
runpy.run_path(sys.argv[1], run_name='quant_smoke_import_test')
assert 'torch' not in sys.modules
assert not any(name == 'vlash' or name.startswith('vlash.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(CLI)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("scheme", ["w8a8", "w4a4"])
def test_dry_run_reports_plan_without_imports_or_output_files(tmp_path, scheme):
    quant, kernel = sources(tmp_path)
    output = tmp_path / "result"
    result = run_cli(quant, kernel, output, "--scheme", scheme, "--dry-run")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "dry_run"
    assert report["scope"] == "synthetic_linear_functional_smoke"
    assert report["scheme"] == scheme
    assert report["quality_gate"] == "not_established"
    assert report["speed_gate"] == "not_measured"
    assert report["plan"]["m_values"] == [1, 64, 129]
    assert report["plan"]["n"] == report["plan"]["k"] == 1024
    assert report["requested_gpu"] == "3"
    assert report["quant_source"] == str(quant.resolve())
    assert not output.exists()


@pytest.mark.parametrize("missing", ["backend", "kernel"])
def test_dry_run_rejects_missing_source_without_loading_backend(tmp_path, missing):
    quant, kernel = sources(tmp_path)
    if missing == "backend":
        (quant / "vlash/quantization/qserve_backend.py").unlink()
    else:
        (kernel / "eval/quant_linear.py").unlink()
    output = tmp_path / "result"
    result = run_cli(quant, kernel, output, "--dry-run")
    assert result.returncode != 0
    assert "qserve_backend.py" in result.stderr or "kernel-source" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("gpu", ["-1", "0,1", "all", "", "GPU-invalid"])
def test_gpu_must_select_one_index_or_uuid(tmp_path, gpu):
    quant, kernel = sources(tmp_path)
    result = run_cli(quant, kernel, tmp_path / "result", "--gpu", gpu, "--dry-run")
    assert result.returncode != 0
    assert "--gpu" in result.stderr


def test_gpu_uuid_and_profiler_are_declared_in_plan(tmp_path):
    quant, kernel = sources(tmp_path)
    gpu = "GPU-01234567-89ab-cdef-0123-456789abcdef"
    result = run_cli(
        quant, kernel, tmp_path / "result", "--gpu", gpu, "--profile", "--dry-run"
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["requested_gpu"] == gpu
    assert report["plan"]["profile_cuda_kernels"] is True


def test_existing_output_is_not_overwritten(tmp_path):
    quant, kernel = sources(tmp_path)
    output = tmp_path / "result"
    output.mkdir()
    original = output / "report.json"
    original.write_text('{"status":"original"}\n')
    result = run_cli(quant, kernel, output)
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert original.read_text() == '{"status":"original"}\n'


def test_runtime_failure_writes_json_after_setting_device_and_source(tmp_path):
    quant, kernel = sources(tmp_path)
    fake_modules = tmp_path / "optional modules"
    fake_modules.mkdir()
    (fake_modules / "torch.py").write_text(
        "import os\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'\n"
        f"assert os.environ['VLASH_QSERVE_ROOT'] == {str(kernel.resolve())!r}\n"
        "raise RuntimeError('optional torch import failed for CPU test')\n"
    )
    env = dict(os.environ, PYTHONPATH=str(fake_modules))
    output = tmp_path / "result"
    result = run_cli(quant, kernel, output, env=env)
    assert result.returncode == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert "optional torch import failed for CPU test" in report["error"]
    assert report["rows"] == []
    assert report["quality_gate"] == "not_established"


def test_actual_run_with_invalid_path_records_failure(tmp_path):
    quant, kernel = sources(tmp_path)
    (quant / "vlash/quantization/qserve_backend.py").unlink()
    output = tmp_path / "result"
    result = run_cli(quant, kernel, output)
    assert result.returncode == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert "qserve_backend.py" in report["error"]


def test_required_inputs_have_no_machine_specific_defaults():
    result = subprocess.run(
        [sys.executable, str(CLI), "--dry-run"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    for flag in (
        "--quant-source",
        "--kernel-source",
        "--scheme",
        "--output-dir",
        "--gpu",
    ):
        assert flag in result.stderr
