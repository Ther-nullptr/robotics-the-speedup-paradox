"""Check the FP32 noise boundary when moving native controls into a larger JIT."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("gpu must be nonnegative")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    import jax
    import jax.numpy as jnp
    import numpy as np

    output = args.output_dir.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (output / "diagnostic-source.py").write_bytes(source)
    fused = jax.jit(
        lambda base, step: jax.random.normal(jax.random.fold_in(base, step), (6,)) * 0.1
    )
    separated = jax.jit(
        lambda base, step: (
            jax.lax.optimization_barrier(
                jax.random.normal(jax.random.fold_in(base, step), (6,))
            )
            * 0.1
        )
    )
    rows = []
    for seed in range(4):
        base = jax.random.fold_in(jax.random.key(seed), 1)
        for step in (0, 1, 2, 18, 255):
            reference = np.asarray(
                jax.random.normal(jax.random.fold_in(base, step), (6,)) * 0.1
            )
            a, b = (
                np.asarray(fused(base, jnp.int32(step))),
                np.asarray(separated(base, jnp.int32(step))),
            )
            rows.append(
                {
                    "seed": seed,
                    "control_index": step,
                    "fused_exact": reference.tobytes() == a.tobytes(),
                    "separated_exact": reference.tobytes() == b.tobytes(),
                    "max_fused_error": float(np.max(np.abs(reference - a))),
                    "reference": reference.tolist(),
                    "fused": a.tolist(),
                    "separated": b.tolist(),
                }
            )
    report = {
        "scope": "Native action-noise shape6, FP32, std0.1; no physics, policy or performance measurement.",
        "python": sys.version,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy")
        },
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "rows": rows,
        "fused_mismatches": sum(not row["fused_exact"] for row in rows),
        "separated_mismatches": sum(not row["separated_exact"] for row in rows),
    }
    (output / "noise-rounding.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Fused mismatches: {report['fused_mismatches']}/{len(rows)}; separated mismatches: {report['separated_mismatches']}/{len(rows)}"
    )
    if report["separated_mismatches"]:
        raise SystemExit("Noise boundary check failed")


if __name__ == "__main__":
    main()
