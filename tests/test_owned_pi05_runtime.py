"""The execution graph must resolve to the repository-owned implementation."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1] / "src/robotics_bench/models/pi05"


def test_owned_runtime_selects_local_backbones():
    assert (ROOT / "modeling_pi05.py").exists(), "PI0.5 runtime has not been migrated"
    main = ast.parse((ROOT / "modeling_pi05.py").read_text())
    imports = [node for node in ast.walk(main) if isinstance(node, ast.ImportFrom)]
    assert any(n.level == 1 and n.module == "modeling_paligemma" for n in imports)
    pali = (ROOT / "modeling_paligemma.py").read_text()
    assert "AutoModel.from_config" not in pali
    assert "SiglipVisionModel(config.vision_config)" in pali
    assert "GemmaModel(config.text_config)" in pali


def test_source_provenance_is_recorded_without_local_paths():
    path = ROOT / "PROVENANCE.json"
    assert path.exists(), "Missing vendored runtime provenance"
    import json

    data = json.loads(path.read_text())
    assert len(data["files"]) == 4
    for item in data["files"]:
        assert len(item["original_sha256"]) == 64
        assert not item["source"].startswith("/home/")
