"""Resource preparation remains usable without model or simulator imports."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/prepare_resources.py"


@pytest.fixture
def api():
    spec = importlib.util.spec_from_file_location("prepare_resources", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_needs_no_dependencies_network_or_existing_paths(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            "--case",
            "cosmos_libero",
            "--root",
            str(tmp_path / "absent"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert {r["name"] for r in plan["resources"]} == {"policy", "vae", "libero-assets"}
    assert not (tmp_path / "absent").exists()
    assert "model-480p-16fps.pt" not in result.stdout


def test_optional_datasets_and_root_override(api, tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTICS_RESOURCE_ROOT", str(tmp_path))
    for case, repo in [
        ("pi05_libero", "lerobot/libero"),
        ("cosmos_libero", "nvidia/LIBERO-Cosmos-Policy"),
    ]:
        args = api.build_parser().parse_args(["--case", case, "--with-dataset"])
        plan = api.build_plan(args)
        assert plan["root"] == str(tmp_path)
        dataset = next(r for r in plan["resources"] if r["name"] == "dataset")
        assert dataset["repo_id"] == repo
        assert dataset["repo_type"] == "dataset"


def fake_policy(path):
    path.mkdir(parents=True)
    for name in ["Cosmos-Policy-LIBERO-Predict2-2B.pt", "libero_t5_embeddings.pkl"]:
        (path / name).write_bytes(b"example-test-resource")
    (path / "libero_dataset_statistics.json").write_text("{}")


def test_local_policy_validation_rejects_missing_and_lfs_pointer(api, tmp_path):
    args = api.build_parser().parse_args(["--case", "cosmos_libero"])
    resource = api.build_plan(args)["resources"][0]
    with pytest.raises(ValueError, match="missing|Missing"):
        api.validate_local(resource, tmp_path)
    fake_policy(tmp_path / "policy")
    (tmp_path / "policy/Cosmos-Policy-LIBERO-Predict2-2B.pt").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:000\n"
    )
    with pytest.raises(ValueError, match="LFS"):
        api.validate_local(resource, tmp_path / "policy")


def test_download_pins_revision_and_rejects_truncated_file(api, tmp_path):
    resource = {
        "name": "example",
        "repo_id": "org/model",
        "repo_type": "model",
        "revision": "main",
        "patterns": ["config.json"],
        "required": ["config.json"],
    }
    calls = []
    info = SimpleNamespace(
        sha="a" * 40, siblings=[SimpleNamespace(rfilename="config.json", size=2)]
    )
    cache = tmp_path / "snapshot"
    cache.mkdir()
    (cache / "config.json").write_text("{")

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(cache)

    hub = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(repo_info=lambda **kw: info),
        snapshot_download=snapshot,
    )
    with pytest.raises(ValueError, match="size|Size"):
        api.download_resource(resource, tmp_path, hub)
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["allow_patterns"] == ["config.json"]
    (cache / "config.json").write_text("{}")
    result = api.download_resource(resource, tmp_path, hub)
    assert result["resolved_revision"] == "a" * 40
    assert result["file_count"] == 1


def test_missing_remote_required_file_fails_before_download(api, tmp_path):
    args = api.build_parser().parse_args(["--case", "cosmos_libero"])
    resource = api.build_plan(args)["resources"][0]
    info = SimpleNamespace(sha="a" * 40, siblings=[])
    hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(repo_info=lambda **kw: info))
    with pytest.raises(ValueError, match="missing|Missing"):
        api.download_resource(resource, tmp_path, hub)


def test_env_is_shell_safe_and_does_not_replace_different_file(api, tmp_path):
    target = tmp_path / "paths.env"
    value = str(tmp_path / "space ' $(touch MUST_NOT_EXIST) `false`")
    api.write_env(target, {"ROBOTICS_CHECKPOINT": value})
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s" "$ROBOTICS_CHECKPOINT"',
            "bash",
            str(target),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == value
    assert not (tmp_path / "MUST_NOT_EXIST").exists()
    api.write_env(target, {"ROBOTICS_CHECKPOINT": value})
    with pytest.raises(FileExistsError):
        api.write_env(target, {"ROBOTICS_CHECKPOINT": "different"})


def test_failed_preparation_does_not_publish_env(api, tmp_path, monkeypatch):
    args = api.build_parser().parse_args(
        ["--case", "cosmos_libero", "--root", str(tmp_path)]
    )
    plan = api.build_plan(args)

    def fail(*a):
        raise RuntimeError("download interrupted")

    monkeypatch.setattr(api, "download_resource", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        api.prepare(plan, hub=object())
    assert not Path(plan["env_file"]).exists()


def test_libero_configuration_requires_paired_task_files(api, tmp_path):
    root = tmp_path / "libero"
    for suite in api.LIBERO_SUITES:
        for directory, suffix in [
            ("bddl_files", ".bddl"),
            ("init_files", ".pruned_init"),
        ]:
            parent = root / directory / suite
            parent.mkdir(parents=True)
            for index in range(10):
                (parent / f"task_{index}{suffix}").write_text("fixture")
    config = api.libero_paths(root, tmp_path / "dataset", tmp_path / "assets")
    assert config["init_states"] == str(root / "init_files")
    (root / "init_files/libero_object/task_0.pruned_init").unlink()
    with pytest.raises(ValueError, match="initial|task"):
        api.libero_paths(root, tmp_path / "dataset", tmp_path / "assets")


def test_python_symlink_preserves_virtual_environment(api, tmp_path):
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    args = api.build_parser().parse_args(
        ["--case", "cosmos_libero", "--python", str(executable)]
    )
    assert api.build_plan(args)["python"] == str(executable)


def test_complete_local_preparation_is_repeatable_and_offline(
    api, tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    fake_policy(tmp_path / "policy")
    vae = tmp_path / "vae/tokenizer"
    vae.mkdir(parents=True)
    (vae / "tokenizer.pth").write_bytes(b"test vae")
    assets = tmp_path / ".cache/libero/assets"
    for name in api.ASSET_DIRECTORIES:
        (assets / name).mkdir(parents=True)
        (assets / name / "test.xml").write_text("<test/>")
    args = api.build_parser().parse_args(
        [
            "--case",
            "cosmos_libero",
            "--root",
            str(tmp_path / "resources"),
            "--reuse",
            f"policy={tmp_path / 'policy'}",
            "--reuse",
            f"vae={vae.parent}",
            "--reuse",
            f"libero-assets={assets}",
        ]
    )
    plan = api.build_plan(args)
    report = api.prepare(plan, hub=object())
    assert api.prepare(plan, hub=object()) == report
    assert all(r["origin"] == "local" for r in report["resources"])
    assert "ROBOTICS_COSMOS_VAE_CHECKPOINT" in Path(plan["env_file"]).read_text()
