"""Dense-campaign balanced-prefix regressions using tiny synthetic raw ledgers."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dense = load(
    "dense_analysis_test",
    ROOT / "benchmarks/dynamic/dynamicvla_dom/analyze_dense_study.py",
)
fixtures = load("base_analysis_fixtures", ROOT / "tests/test_dynamicvla_analysis.py")


def make_campaign(tmp_path, count=3):
    blocks = []
    for index in range(count):
        folder = tmp_path / f"block-{index:02d}"
        folder.mkdir()
        seed = 4 + index * 2
        cells = []
        for task in ("pick", "place"):
            for delay in (0, 100):
                cell = fixtures.make_run(folder, task, delay)
                cell["seed"] = seed
                run = folder / cell["run_dir"]
                manifest = json.loads((run / "case-manifest.json").read_text())
                manifest["options"]["seed"] = seed
                manifest["options"].update(model_gpu="0", sim_gpu="1", rotation="euler")
                manifest["resources"] = {
                    "checkpoint": "/fixtures/checkpoint",
                    "env_cfg": f"/fixtures/{task}.json",
                }
                fixtures.write(run / "case-manifest.json", manifest)
                episodes = [
                    json.loads(line)
                    for line in (run / "episodes.jsonl").read_text().splitlines()
                ]
                for row in episodes:
                    row["env_seed"] = seed + row["episode_id"]
                fixtures.jsonl(run / "episodes.jsonl", episodes)
                cells.append(cell)
        fixtures.study(folder, cells)
        blocks.append(
            {
                "id": f"block-{index:02d}",
                "seed": seed,
                "episodes": 2,
                "study": f"block-{index:02d}/study.json",
            }
        )
    campaign = {
        "format": dense.FORMAT,
        "id": "dense-test",
        "expected_tasks": ["pick", "place"],
        "delays_ms": [0, 100],
        "num_steps": 10,
        "episodes_per_task_condition": count * 2,
        "block_episodes": 2,
        "seed": 4,
        "blocks": blocks,
    }
    campaign["identity"] = {
        "resources": {"checkpoint": "/fixtures/checkpoint"},
        "tasks": {f"/fixtures/{task}.json": task for task in ("pick", "place")},
        "source_sha256": {"src/robotics_bench/server.py": "source"},
        "checkpoint_sha256": "weights",
        "checkpoint_config_sha256": "config",
        "model_gpu": "0",
        "sim_gpu": "1",
        "rotation": "euler",
        "num_steps": 10,
    }
    path = tmp_path / "campaign.json"
    fixtures.write(path, campaign)
    return path


def make_partial(tmp_path, index):
    path = tmp_path / f"block-{index:02d}/pick-100/case-manifest.json"
    value = json.loads(path.read_text())
    value["status"] = "running"
    fixtures.write(path, value)


def test_only_balanced_prefix_aggregates_and_report_is_explicit(tmp_path):
    path = make_campaign(tmp_path)
    make_partial(tmp_path, 1)
    result = dense.analyze(path)
    assert result["status"] == "partial" and result["interim"]
    assert result["completed_blocks"] == 2
    assert result["included_blocks"] == 1
    assert (result["included_episodes"], result["expected_episodes"]) == (8, 24)
    assert result["completed_episodes"] == 22
    assert [r["included_in_aggregate"] for r in result["block_progress"]] == [
        True,
        False,
        False,
    ]
    assert all(
        r["metrics"]["episodes"] == 2 and r["interim"]
        for r in result["task_conditions"]
    )
    assert all(r["metrics"]["episodes"] == 4 for r in result["pooled_conditions"])
    assert result["pooled_conditions"][1]["paired_vs_zero"]["episodes"] == 4
    output = tmp_path / "analysis"
    dense.save(result, output)
    assert (
        (output / "report.md")
        .read_text()
        .startswith("**INTERIM BALANCED PREFIX: 1/3 blocks; 8/24 episodes included.")
    )


def test_first_partial_block_does_not_pool_later_complete_blocks(tmp_path):
    path = make_campaign(tmp_path)
    make_partial(tmp_path, 0)
    result = dense.analyze(path)
    assert result["included_blocks"] == 0 and result["completed_blocks"] == 2
    assert all(
        row["metrics"] is None
        for row in result["task_conditions"] + result["pooled_conditions"]
    )


def test_complete_campaign_pools_disjoint_seeds(tmp_path):
    result = dense.analyze(make_campaign(tmp_path))
    assert result["status"] == "completed" and not result["interim"]
    assert result["included_episodes"] == result["expected_episodes"] == 24
    assert all(r["metrics"]["episodes"] == 6 for r in result["task_conditions"])
    assert all(r["metrics"]["episodes"] == 12 for r in result["pooled_conditions"])
    assert result["pooled_conditions"][1]["paired_vs_zero"]["episodes"] == 12


def test_overlapping_seeds_rejected(tmp_path):
    path = make_campaign(tmp_path)
    value = json.loads(path.read_text())
    value["blocks"][1]["seed"] = 5
    fixtures.write(path, value)
    with pytest.raises(ValueError, match="overlapping seeds"):
        dense.analyze(path)


def test_duplicate_condition_grid_rejected(tmp_path):
    path = make_campaign(tmp_path)
    block = tmp_path / "block-00/study.json"
    value = json.loads(block.read_text())
    value["cells"].append(value["cells"][0])
    fixtures.write(block, value)
    with pytest.raises(ValueError, match="Duplicate"):
        dense.analyze(path)


@pytest.mark.parametrize("kind", ["source", "task"])
def test_drift_in_excluded_completed_cells_still_rejected(tmp_path, kind):
    path = make_campaign(tmp_path)
    make_partial(tmp_path, 1)
    manifest_path = tmp_path / "block-02/pick-0/case-manifest.json"
    value = json.loads(manifest_path.read_text())
    if kind == "source":
        value["source_sha256"]["server.py"] = "changed"
    else:
        value["identity"]["task_config_sha256"] = "changed"
    fixtures.write(manifest_path, value)
    with pytest.raises(ValueError, match="identity"):
        dense.analyze(path)


def test_missing_future_descriptor_remains_planned(tmp_path):
    path = make_campaign(tmp_path)
    (tmp_path / "block-02/study.json").unlink()
    result = dense.analyze(path)
    assert result["included_blocks"] == 2
    assert result["block_progress"][-1]["status"] == "planned"


@pytest.mark.parametrize(
    "key",
    [
        "checkpoint_sha256",
        "checkpoint_config_sha256",
        "source_sha256",
        "tasks",
        "model_gpu",
        "sim_gpu",
        "rotation",
        "resources",
    ],
)
def test_consistent_runs_must_match_declared_campaign_identity(tmp_path, key):
    path = make_campaign(tmp_path, count=1)
    campaign = json.loads(path.read_text())
    identity = campaign["identity"]
    if key == "source_sha256":
        identity[key]["src/robotics_bench/server.py"] = "different"
    elif key == "tasks":
        identity[key]["/fixtures/pick.json"] = "different"
    elif key == "resources":
        identity[key]["checkpoint"] = "/different/checkpoint"
    else:
        identity[key] = "different"
    fixtures.write(path, campaign)
    with pytest.raises(ValueError, match="declared campaign identity"):
        dense.analyze(path)


def test_campaign_identity_is_required(tmp_path):
    path = make_campaign(tmp_path, count=1)
    campaign = json.loads(path.read_text())
    del campaign["identity"]
    fixtures.write(path, campaign)
    with pytest.raises(ValueError, match="declared identity"):
        dense.analyze(path)
