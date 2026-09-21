"""Preserved FP backends must not depend on the sibling VLM checkout."""

from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]


def test_fp_sources_are_independent_and_complete():
    package = ROOT / "src/robotics_kernels/blackwell"
    assert package.is_dir(), "Blackwell FP sources have not been preserved"
    for stem in ("fp4", "fp8", "fp4_mixed", "fp4_mx", "mxfp8"):
        assert (package / "csrc" / f"{stem}_cutlass_linear.cu").is_file()
    for path in package.glob("*.py"):
        text = path.read_text()
        ast.parse(text)
        assert "from streaming_vlm" not in text
        assert "import streaming_vlm" not in text
        assert "/home/" not in text
    for path in (package / "csrc").glob("*.cu"):
        assert "TORCH_LIBRARY(streaming_vlm" not in path.read_text()
