"""Exercise the thin Bash launcher with a fake Python, without model imports."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


CASE_DIR = Path(__file__).resolve().parents[1] / "benchmarks/static/pi05_libero"
SHELL = CASE_DIR / "run.sh"


@pytest.fixture
def recorder(tmp_path):
    binary_dir = tmp_path / "fake python environment"
    binary_dir.mkdir()
    executable = binary_dir / "python"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "record = {'argv': sys.argv[1:], 'cwd': os.getcwd(),\n"
        "          'checkpoint': os.environ.get('ROBOTICS_CHECKPOINT'),\n"
        "          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}\n"
        "Path(os.environ['ARGV_RECORD']).write_text(json.dumps(record))\n"
        "print('fake stdout')\n"
        "print('fake stderr', file=sys.stderr)\n"
        "sys.exit(int(os.environ.get('FAKE_EXIT_CODE', '0')))\n"
    )
    executable.chmod(0o755)
    outside = tmp_path / "outside working directory"
    outside.mkdir()
    output = tmp_path / "arguments.json"
    env = {
        **os.environ,
        "ROBOTICS_PI05_PYTHON": str(executable),
        "ARGV_RECORD": str(output),
        "ROBOTICS_CHECKPOINT": "explicit environment checkpoint",
    }
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return executable, outside, output, env


def invoke(recorder, *arguments, launcher=SHELL):
    _, outside, output, env = recorder
    located = launcher if launcher.is_absolute() else outside / launcher
    assert located.is_file(), "run.sh is not implemented"
    result = subprocess.run(
        ["bash", str(launcher), *arguments],
        cwd=outside,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, json.loads(output.read_text())


def test_outside_directory_and_spaced_arguments_are_preserved(recorder):
    arguments = ["--checkpoint", "model directory with spaces", "--dry-run", ""]
    result, recorded = invoke(recorder, *arguments)
    assert result.returncode == 0
    assert recorded["argv"] == [str(CASE_DIR / "run.py"), *arguments]
    assert recorded["cwd"] == str(recorder[1])
    assert recorded["cuda_visible_devices"] is None
    assert result.stdout == "fake stdout\n"
    assert result.stderr == "fake stderr\n"


def test_shell_literals_are_forwarded_without_execution(recorder, tmp_path):
    marker = tmp_path / "must-not-be-created"
    arguments = [
        "--label",
        f"$(touch '{marker}')",
        f"`touch '{marker}'`",
        f"; touch '{marker}'",
        "$HOME",
        "*.json",
        "first line\nsecond line",
    ]
    result, recorded = invoke(recorder, *arguments)
    assert result.returncode == 0
    assert recorded["argv"][1:] == arguments
    assert not marker.exists()


@pytest.mark.parametrize("exit_code", [0, 7, 130])
def test_python_exit_code_is_propagated(recorder, exit_code):
    recorder[3]["FAKE_EXIT_CODE"] = str(exit_code)
    result, _ = invoke(recorder)
    assert result.returncode == exit_code


@pytest.mark.parametrize("interpreter_value", [None, ""])
def test_default_python_is_resolved_from_path(recorder, interpreter_value):
    executable, _, _, env = recorder
    if interpreter_value is None:
        env.pop("ROBOTICS_PI05_PYTHON")
    else:
        env["ROBOTICS_PI05_PYTHON"] = interpreter_value
    env["PATH"] = str(executable.parent) + os.pathsep + env["PATH"]
    result, recorded = invoke(recorder, "--help")
    assert result.returncode == 0
    assert recorded["argv"] == [str(CASE_DIR / "run.py"), "--help"]


def test_local_environment_file_is_not_automatically_sourced(recorder):
    local = recorder[1] / ".local"
    local.mkdir()
    (local / "pi05-libero.env").write_text(
        "export ROBOTICS_CHECKPOINT='unexpected auto-source'\n"
    )
    result, recorded = invoke(recorder)
    assert result.returncode == 0
    assert recorded["checkpoint"] == "explicit environment checkpoint"


def test_launcher_directory_can_contain_spaces(recorder, tmp_path):
    assert SHELL.is_file(), "run.sh is not implemented"
    copied_dir = tmp_path / "checkout with spaces" / "case entry"
    copied_dir.mkdir(parents=True)
    copied_launcher = copied_dir / "run.sh"
    shutil.copy2(SHELL, copied_launcher)
    result, recorded = invoke(recorder, "--help", launcher=copied_launcher)
    assert result.returncode == 0
    assert recorded["argv"] == [str(copied_dir / "run.py"), "--help"]
    assert recorded["cwd"] == str(recorder[1])


def test_relative_launcher_path_resolves_adjacent_python_script(recorder):
    relative = os.path.relpath(SHELL, recorder[1])
    result, recorded = invoke(recorder, "--dry-run", launcher=Path(relative))
    assert result.returncode == 0
    assert Path(recorded["argv"][0]).resolve() == CASE_DIR / "run.py"


def test_shell_and_environment_example_have_valid_bash_syntax():
    for path in (SHELL, CASE_DIR / "paths.env.example"):
        assert path.is_file(), f"{path.name} is not implemented"
        result = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0, result.stderr


def test_environment_example_declares_explicit_placeholder_paths():
    source = CASE_DIR / "paths.env.example"
    assert source.is_file(), "paths.env.example is not implemented"
    text = source.read_text()
    for variable in (
        "ROBOTICS_PI05_PYTHON",
        "ROBOTICS_SIM_SOURCE",
        "ROBOTICS_CHECKPOINT",
        "ROBOTICS_TOKENIZER",
        "ROBOTICS_LIBERO_CONFIG_DIR",
        "ROBOTICS_QUANT_SOURCE",
        "ROBOTICS_KERNEL_SOURCE",
    ):
        assert f'{variable}="PATH/TO/' in text
    assert "source .local/pi05-libero.env" in text
    assert "CUDA_VISIBLE_DEVICES=" not in text
