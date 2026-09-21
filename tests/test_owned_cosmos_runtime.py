"""Runtime migration is bounded and records what code is now owned."""

from pathlib import Path
import json

ROOT = Path(__file__).parents[1] / "src/robotics_bench/models/cosmos"


def test_cosmos_owned_runtime_has_its_own_policy_and_dit_graph():
    assert (ROOT / "runtime.py").exists(), "Cosmos runtime has not been migrated"
    assert (
        "from .minimal_v4_dit import MiniTrainDIT"
        in (ROOT / "minimal_v1_lvg_dit.py").read_text()
    )
    assert (
        "from .policy_text2world_model import"
        in (ROOT / "policy_video2world_model.py").read_text()
    )
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    assert len(provenance["files"]) >= 5
    assert all(len(f["original_sha256"]) == 64 for f in provenance["files"])
