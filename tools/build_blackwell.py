#!/usr/bin/env python3
"""Build an explicitly selected, repository-owned Blackwell extension."""

import argparse
import os
from pathlib import Path

SOURCES = {
    "fp4": ("fp4_cutlass_linear.cu", "robotics_cutlass_fp4"),
    "fp8": ("fp8_cutlass_linear.cu", "robotics_cutlass_fp8"),
    "fp4_mixed": ("fp4_mixed_cutlass_linear.cu", "robotics_cutlass_fp4_mixed"),
    "fp4_mx": ("fp4_mx_cutlass_linear.cu", "robotics_cutlass_fp4_mx"),
    "mxfp8": ("mxfp8_cutlass_linear.cu", "robotics_mxfp8"),
    "mxfp8_swiglu": ("mxfp8_swiglu.cu", "robotics_mxfp8_swiglu"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=SOURCES, required=True)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument(
        "--arch",
        required=True,
        help="Explicit CUDA target, e.g. 10.0a or 11.0a; requires a compatible toolkit",
    )
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, namespace = SOURCES[args.backend]
    includes = [args.cutlass_root / "include", args.cutlass_root / "tools/util/include"]
    for path in includes:
        if not path.is_dir():
            parser.error(f"Missing CUTLASS directory: {path}")
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch
    os.environ.setdefault("MAX_JOBS", "2")
    from torch.utils.cpp_extension import load

    output = args.build_dir or root / "build" / namespace
    output.mkdir(parents=True, exist_ok=True)
    library = load(
        name=namespace + "_ext",
        sources=[str(root / "src/robotics_kernels/blackwell/csrc" / source)],
        build_directory=str(output),
        extra_include_paths=[str(p) for p in includes],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        is_python_module=False,
        verbose=args.verbose,
    )
    print(library)


if __name__ == "__main__":
    main()
