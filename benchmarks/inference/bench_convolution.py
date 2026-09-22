"""Replay profiled Cosmos convolutions with explicit CUTLASS tactics.

The trace supplies actual shape, stride, padding, dtype and input layout.
Synthetic values are used only for isolated kernel timing and numerical checks;
these results are not model or task-quality measurements. Prepared device time
excludes activation-layout conversion. Complete device and eager synchronized
wall times include it; one-time weight-pack wall time is reported separately.
"""

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def extract_cases(trace):
    from robotics_kernels.ampere_ada.convolution import convolution_plan

    cases = {}
    for event in trace.get("traceEvents", []):
        if event.get("name") != "aten::convolution":
            continue
        args = event.get("args", {})
        shapes = args.get("Input Dims", [])
        strides = args.get("Input Strides", [])
        values = args.get("Concrete Inputs", [])
        if len(shapes) < 3 or len(strides) < 1 or len(values) < 9:
            raise ValueError(
                "Convolution trace must include shapes, strides and concrete inputs"
            )
        row = {
            "input_shape": shapes[0],
            "weight_shape": shapes[1],
            "input_strides": strides[0],
            "bias": bool(shapes[2]),
            "stride": ast.literal_eval(values[3]),
            "padding": ast.literal_eval(values[4]),
            "dilation": ast.literal_eval(values[5]),
            "transposed": ast.literal_eval(values[6]),
            "groups": int(values[8]),
            "dtype": args.get("Input type", [None])[0],
        }
        key = json.dumps(row, sort_keys=True)
        if key in cases:
            cases[key]["calls"] += 1
            continue
        row["calls"] = 1
        row["supported"] = False
        try:
            if row["transposed"] or row["dtype"] != "c10::BFloat16":
                raise ValueError("Only nontransposed BF16 convolution is supported")
            convolution_plan(
                row["weight_shape"],
                stride=row["stride"],
                padding=row["padding"],
                dilation=row["dilation"],
                groups=row["groups"],
            )
            row["supported"] = True
        except ValueError as exc:
            row["reason"] = str(exc)
        shape, weight = row["input_shape"], row["weight_shape"]
        output = [
            (d + 2 * p - dilation * (kernel - 1) - 1) // stride + 1
            for d, p, dilation, kernel, stride in zip(
                shape[2:], row["padding"], row["dilation"], weight[2:], row["stride"]
            )
        ]
        row["multiply_adds"] = shape[0] * math.prod(output) * math.prod(weight)
        cases[key] = row
    result = sorted(
        cases.values(), key=lambda row: -(row["multiply_adds"] * row["calls"])
    )
    for index, row in enumerate(result):
        row["id"] = f"conv-{index:03d}"
    return result


def measure_events(function, repeats, inner, warmup_ms):
    import torch

    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            output = function()
    graph.replay()
    torch.cuda.synchronize()
    # Short captures can otherwise measure a changing clock/power state.
    # Sustained warmup does not imply that clocks are locked.
    deadline = time.perf_counter() + warmup_ms / 1000
    while time.perf_counter() < deadline:
        graph.replay()
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    del output, graph
    return samples


def measure_wall(function, repeats, warmup_ms):
    import torch

    deadline = time.perf_counter() + warmup_ms / 1000
    while time.perf_counter() < deadline:
        function()
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        output = function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    del output
    return samples


def main(argv=None):
    from robotics_kernels.ampere_ada.convolution import TACTICS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--inner", type=int, default=4)
    parser.add_argument(
        "--warmup-ms",
        type=float,
        default=500,
        help="Sustained warmup per measured scope; clocks remain unlocked",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Largest recorded aggregate MAC workloads; 0 means all",
    )
    parser.add_argument(
        "--dimensions", type=int, nargs="+", choices=(2, 3), default=[2, 3]
    )
    parser.add_argument(
        "--case-ids", nargs="+", help="Optional explicit IDs from an earlier manifest"
    )
    parser.add_argument(
        "--tactics", type=int, nargs="+", choices=tuple(TACTICS), default=list(TACTICS)
    )
    parser.add_argument("--cudnn-benchmark", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 3 or args.inner < 1 or args.limit < 0 or args.warmup_ms < 0:
        raise ValueError("Require repeats>=3, inner>=1, limit>=0 and warmup-ms>=0")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    trace = args.trace.resolve()
    cases = extract_cases(json.loads(trace.read_text()))
    selected = [
        row
        for row in cases
        if row["supported"]
        and len(row["input_shape"]) - 2 in args.dimensions
        and (args.case_ids is None or row["id"] in args.case_ids)
    ]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No supported convolution cases were selected")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from robotics_kernels.ampere_ada.convolution import (
        PackedConvolution,
        build_inputs,
        load_convolution_extension,
    )

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    props = torch.cuda.get_device_properties(0)

    def gpu_state():
        command = [
            "nvidia-smi",
            "-i",
            args.gpu,
            "--query-gpu=uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu",
            "--format=csv,noheader",
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        return (
            result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        )

    metadata = {
        "status": "running",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "device": props.name,
        "capability": [props.major, props.minor],
        "gpu_selector": args.gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "repeats": args.repeats,
        "graph_inner": args.inner,
        "warmup_ms": args.warmup_ms,
        "clocks_locked": False,
        "gpu_state_before": gpu_state(),
        "trace": str(trace),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "inputs": "synthetic values in recorded tensor shapes and input strides",
        "boundaries": {
            "prepared_conv": "CUDA events per captured convolution; activation layout and weight packing excluded",
            "complete_conv": "CUDA events per captured call including required input/output layout copies",
            "eager_wall": "CPU call through synchronized CUDA completion including dispatch and layout copies",
            "weight_pack_ms": "one-time synchronized CPU wall time; outside steady-state samples",
        },
        "tactics": TACTICS,
        "build_inputs": build_inputs(),
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "src/robotics_kernels/ampere_ada/convolution.py",
                ROOT / "src/robotics_kernels/ampere_ada/csrc/convolution_fprop.cu",
            ]
        },
        "cases": cases,
        "selected_case_ids": [row["id"] for row in selected],
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    try:
        started = time.perf_counter()
        load_convolution_extension()
        metadata["build_seconds"] = time.perf_counter() - started
        with torch.inference_mode():
            for index, case in enumerate(selected):
                dimensions = len(case["input_shape"]) - 2
                cls = torch.nn.Conv2d if dimensions == 2 else torch.nn.Conv3d
                module = cls(
                    case["weight_shape"][1],
                    case["weight_shape"][0],
                    tuple(case["weight_shape"][2:]),
                    stride=tuple(case["stride"]),
                    padding=tuple(case["padding"]),
                    bias=case["bias"],
                    device="cuda",
                    dtype=torch.bfloat16,
                ).eval()
                x = torch.empty_strided(
                    case["input_shape"],
                    case["input_strides"],
                    device="cuda",
                    dtype=torch.bfloat16,
                ).normal_()
                reference = module(x)
                native_events = measure_events(
                    lambda: module(x), args.repeats, args.inner, args.warmup_ms
                )
                native_wall = measure_wall(
                    lambda: module(x), args.repeats, args.warmup_ms
                )
                baseline = {
                    "case_id": case["id"],
                    "backend": "native_cudnn",
                    "scope": "complete_conv",
                    "samples_ms": native_events,
                    "median_ms": statistics.median(native_events),
                    "eager_wall_ms": native_wall,
                    "median_eager_wall_ms": statistics.median(native_wall),
                }
                records.append(baseline)
                order = args.tactics if index % 2 == 0 else list(reversed(args.tactics))
                for tactic in order:
                    torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    packed = PackedConvolution.from_module(module, tactic=tactic)
                    torch.cuda.synchronize()
                    weight_pack_ms = (time.perf_counter_ns() - started) / 1e6
                    actual = packed(x)
                    diff = actual.float() - reference.float()
                    numerical = {
                        "max_abs": diff.abs().max().item(),
                        "rms": diff.square().mean().sqrt().item(),
                        "relative_l2": (
                            diff.norm() / reference.float().norm().clamp_min(1e-30)
                        ).item(),
                        "allclose_rtol_002_atol_002": torch.allclose(
                            actual, reference, rtol=0.02, atol=0.02
                        ),
                    }
                    if (
                        not torch.isfinite(actual).all()
                        or not numerical["allclose_rtol_002_atol_002"]
                    ):
                        records.append(
                            {
                                "case_id": case["id"],
                                "tactic": tactic,
                                "backend": packed.backend,
                                "numerical": numerical,
                                "status": "numerical_check_failed",
                            }
                        )
                        continue
                    packed_input = packed.pack_input(x)
                    for scope, function in (
                        ("prepared_conv", lambda: packed.forward_packed(packed_input)),
                        ("complete_conv", lambda: packed(x)),
                    ):
                        events = measure_events(
                            function, args.repeats, args.inner, args.warmup_ms
                        )
                        row = {
                            "case_id": case["id"],
                            "backend": packed.backend,
                            "tactic": tactic,
                            "scope": scope,
                            "samples_ms": events,
                            "median_ms": statistics.median(events),
                            "weight_pack_ms": weight_pack_ms,
                            "numerical": numerical,
                        }
                        if scope == "complete_conv":
                            wall = measure_wall(function, args.repeats, args.warmup_ms)
                            row.update(
                                eager_wall_ms=wall,
                                median_eager_wall_ms=statistics.median(wall),
                                speedup_vs_native_device=baseline["median_ms"]
                                / row["median_ms"],
                                speedup_vs_native_wall=baseline["median_eager_wall_ms"]
                                / statistics.median(wall),
                            )
                        records.append(row)
                print(
                    json.dumps(
                        {
                            "case_id": case["id"],
                            "input_shape": case["input_shape"],
                            "completed": True,
                        }
                    ),
                    flush=True,
                )
                (output / "measurements.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["gpu_state_after"] = gpu_state()
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
        with (output / "summary.csv").open("w", newline="") as stream:
            fields = [
                "case_id",
                "backend",
                "tactic",
                "scope",
                "median_ms",
                "median_eager_wall_ms",
                "weight_pack_ms",
                "speedup_vs_native_device",
                "speedup_vs_native_wall",
                "status",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
