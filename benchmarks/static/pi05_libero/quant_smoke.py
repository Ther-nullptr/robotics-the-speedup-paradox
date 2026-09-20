#!/usr/bin/env python3
"""Probe an external quantized linear backend; no task-quality or speed claim.

Each CLI invocation runs one scheme in a fresh process. The optional CUDA stack
is loaded only after argument/path checks and device selection. ``--dry-run``
uses the standard library alone and does not create output artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback


def gpu_selector(value: str) -> str:
    uuid = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
    if not re.fullmatch(r"[0-9]+|" + uuid, value):
        raise argparse.ArgumentTypeError(
            "select one non-negative GPU index or full GPU UUID"
        )
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-source",
        required=True,
        type=Path,
        help="source root containing vlash/quantization/qserve_backend.py",
    )
    parser.add_argument(
        "--kernel-source",
        required=True,
        type=Path,
        help="mini_qserve_gemm root containing its package and eval/quant_linear.py",
    )
    parser.add_argument("--scheme", required=True, choices=("w8a8", "w4a4"))
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new, dedicated directory; existing paths are never reused",
    )
    parser.add_argument(
        "--gpu",
        required=True,
        type=gpu_selector,
        help="physical GPU index or full GPU UUID, mapped to cuda:0",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate paths and print the plan"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="capture actual CUDA kernel names once per shape; not a latency benchmark",
    )
    return parser.parse_args(argv)


def validate_sources(args):
    backend = args.quant_source / "vlash/quantization/qserve_backend.py"
    if not backend.is_file():
        raise ValueError(
            f"--quant-source must contain vlash/quantization/qserve_backend.py: {backend}"
        )
    for relative in ("mini_qserve_gemm/__init__.py", "eval/quant_linear.py"):
        if not (args.kernel_source / relative).is_file():
            raise ValueError(
                f"--kernel-source must contain {relative}: {args.kernel_source}"
            )
    return backend


def run_probe(args, report):
    """The only entry point that imports or executes the optional GPU backend."""
    if "torch" in sys.modules or "vlash" in sys.modules:
        raise RuntimeError(
            "run this probe in a fresh process before importing torch or vlash"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLASH_QSERVE_ROOT"] = str(args.kernel_source)
    sys.path.insert(0, str(args.quant_source))

    import torch
    import vlash.quantization.qserve_backend as backend

    expected_backend = (
        args.quant_source / "vlash/quantization/qserve_backend.py"
    ).resolve()
    if Path(backend.__file__).resolve() != expected_backend:
        raise RuntimeError(f"imported a different quant backend: {backend.__file__}")
    report["torch_version"] = str(torch.__version__)
    report["cuda_version"] = torch.version.cuda
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for the selected GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible GPU")
    report["device"] = {
        "logical": "cuda:0",
        "name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
    }
    torch.manual_seed(42)
    dtype = torch.bfloat16
    layer = torch.nn.Linear(1024, 1024, bias=False, device="cuda:0", dtype=dtype).eval()
    quant = backend.make_qserve_quant_linear(
        layer, args.scheme, activation_dtype=dtype, output_dtype=dtype
    ).eval()
    report["modules"] = [
        {"path": name, "type": f"{type(module).__module__}.{type(module).__name__}"}
        for name, module in quant.named_modules()
    ]
    with torch.inference_mode():
        for rows in (1, 64, 129):
            inputs = torch.randn(rows, 1024, device="cuda:0", dtype=dtype)
            reference = layer(inputs).float()
            raw_output = quant(inputs)
            output = raw_output.float()
            torch.cuda.synchronize()
            row = {
                "m": rows,
                "n": 1024,
                "k": 1024,
                "output_dtype": str(raw_output.dtype),
                "output_shape": list(raw_output.shape),
                "comparison_dtype": str(output.dtype),
                "finite": bool(torch.isfinite(output).all()),
                "reference_finite": bool(torch.isfinite(reference).all()),
            }
            report["rows"].append(row)
            if output.shape != reference.shape:
                raise RuntimeError(
                    f"quantized output shape differs from reference for M={rows}"
                )
            if not row["finite"] or not row["reference_finite"]:
                raise RuntimeError(f"non-finite output or reference for M={rows}")
            error = output - reference
            reference_norm = reference.norm().item()
            metrics = {
                "rmse": error.square().mean().sqrt().item(),
                "max_abs_error": error.abs().max().item(),
                "relative_l2": error.norm().item() / reference_norm
                if reference_norm
                else None,
            }
            if not all(
                value is None or math.isfinite(value) for value in metrics.values()
            ):
                raise RuntimeError(f"non-finite error metric for M={rows}")
            row.update(metrics)
            if not reference_norm:
                row["relative_l2_missing_reason"] = "zero_reference_norm"
            if args.profile:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    quant(inputs)
                    torch.cuda.synchronize()
                row["cuda_kernel_names"] = sorted(
                    {
                        event.name
                        for event in profile.events()
                        if "cuda" in str(event.device_type).lower()
                    }
                )
                if not row["cuda_kernel_names"]:
                    raise RuntimeError(
                        f"profiler did not capture CUDA kernel names for M={rows}"
                    )
    if args.profile:
        report["cuda_kernel_names"] = sorted(
            {name for row in report["rows"] for name in row["cuda_kernel_names"]}
        )
    report["status"] = "functional_pass"


def main(argv=None):
    args = parse_args(argv)
    args.quant_source = args.quant_source.expanduser().resolve()
    args.kernel_source = args.kernel_source.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    report = {
        "schema": "robotics.pi05.quant-smoke.v1",
        "scope": "synthetic_linear_functional_smoke",
        "scheme": args.scheme,
        "quant_source": str(args.quant_source),
        "kernel_source": str(args.kernel_source),
        "requested_gpu": args.gpu,
        "quality_gate": "not_established",
        "speed_gate": "not_measured",
        "plan": {
            "m_values": [1, 64, 129],
            "n": 1024,
            "k": 1024,
            "input_dtype": "torch.bfloat16",
            "requested_output_dtype": "torch.bfloat16",
            "bias": False,
            "seed": 42,
            "profile_cuda_kernels": args.profile,
        },
        "rows": [],
    }
    if args.output_dir.exists():
        print(
            f"output directory already exists; use a new path: {args.output_dir}",
            file=sys.stderr,
        )
        return 1
    if args.dry_run:
        try:
            validate_sources(args)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        report["status"] = "dry_run"
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"cannot reserve output directory: {exc}", file=sys.stderr)
        return 1
    try:
        backend_path = validate_sources(args)
        report["backend_sha256"] = hashlib.sha256(backend_path.read_bytes()).hexdigest()
        run_probe(args, report)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "report.json").write_text(serialized)
    print(serialized, end="", flush=True)
    return 0 if report["status"] == "functional_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
