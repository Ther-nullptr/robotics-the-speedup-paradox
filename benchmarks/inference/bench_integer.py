"""Compare real-shape BF16, prepared integer GEMM and complete integer Linear.

GPU event timing of repeated CUDA Graph nodes; not policy or task latency.
Weights/activation preparation outside a prepared-GEMM scope are explicitly
excluded. Complete-Linear scope includes online activation preparation.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# M,N,K from recorded model calls; grouped rows concatenate compatible outputs.
SHAPES = {
    "pi05_text": [
        (968, 2048, 2048),
        (968, 256, 2048),
        (968, 16384, 2048),
        (968, 2048, 16384),
        (968, 2560, 2048),
        (968, 32768, 2048),
    ],
    "cosmos_dit": [
        (1764, 2048, 2048),
        (512, 2048, 1024),
        (1764, 8192, 2048),
        (1764, 2048, 8192),
        (1764, 6144, 2048),
        (512, 4096, 1024),
    ],
}


def measure(function, repeats, inner):
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
    samples = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    del graph, output
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*SHAPES, "both"), default="both")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--inner", type=int, default=8)
    parser.add_argument("--bits", type=int, choices=(4, 8), nargs="+", default=[8, 4])
    parser.add_argument(
        "--tactics", type=int, choices=range(8), nargs="+", default=list(range(8))
    )
    parser.add_argument("--pack-reuse", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 3 or args.inner < 1:
        raise ValueError("Require at least three samples and one graph iteration")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from robotics_kernels.ampere_ada.integer import IntegerLinear
    from robotics_kernels.ampere_ada.tactics import describe

    torch.manual_seed(42)
    props = torch.cuda.get_device_properties(0)
    records = []
    metadata = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "device": props.name,
        "capability": [props.major, props.minor],
        "gpu_selector": args.gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "boundary": "GPU event time per graph node; online Linear preparation included only in complete_linear",
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src/robotics_kernels/ampere_ada").rglob("*")
            if p.suffix in (".py", ".cu")
        },
        "repeats": args.repeats,
        "graph_inner": args.inner,
        "pack_reuse": args.pack_reuse,
    }
    cases = SHAPES if args.case == "both" else {args.case: SHAPES[args.case]}
    try:
        for case, shapes in cases.items():
            for m, n, k in shapes:
                linear = torch.nn.Linear(
                    k, n, bias=False, device="cuda", dtype=torch.bfloat16
                ).eval()
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                with torch.inference_mode():
                    baseline = measure(
                        lambda linear=linear, x=x: linear(x), args.repeats, args.inner
                    )
                    records.append(
                        {
                            "case": case,
                            "shape": [m, n, k],
                            "precision": "bf16",
                            "scope": "complete_linear",
                            "samples_ms": baseline,
                            "median_ms": statistics.median(baseline),
                        }
                    )
                    for bits in args.bits:
                        layer = IntegerLinear.from_linear(
                            linear, bits=bits, pack_reuse=args.pack_reuse
                        )
                        packed = layer.pack_input(x)
                        reference = layer.forward_packed(packed)
                        # Alternate tactic order by precision to reduce ordering bias.
                        order = (
                            args.tactics if bits == 8 else list(reversed(args.tactics))
                        )
                        for tactic in order:
                            layer.tactic = tactic
                            actual = layer.forward_packed(packed)
                            if not torch.equal(actual, reference):
                                raise ValueError(
                                    f"Quantized reference mismatch for bits={bits}, tactic={tactic}, shape={(m, n, k)}"
                                )
                            for scope, function in (
                                (
                                    "prepared_gemm",
                                    lambda layer=layer, packed=packed: (
                                        layer.forward_packed(packed)
                                    ),
                                ),
                                ("complete_linear", lambda layer=layer, x=x: layer(x)),
                            ):
                                samples = measure(function, args.repeats, args.inner)
                                row = {
                                    "case": case,
                                    "shape": [m, n, k],
                                    "precision": f"int{bits}",
                                    "tactic": tactic,
                                    "kernel": describe(tactic, bits),
                                    "scope": scope,
                                    "samples_ms": samples,
                                    "median_ms": statistics.median(samples),
                                    "exact_quantized_reference": True,
                                }
                                row["speedup_vs_bf16"] = (
                                    statistics.median(baseline) / row["median_ms"]
                                )
                                records.append(row)
                        del layer, packed, reference, actual
                    print(
                        json.dumps(
                            {"case": case, "shape": [m, n, k], "completed": True}
                        ),
                        flush=True,
                    )
                    del linear, x
                (output / "measurements.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
        summary = []
        for case, shapes in cases.items():
            for shape in shapes:
                for bits in args.bits:
                    rows = [
                        r
                        for r in records
                        if r["case"] == case
                        and r["shape"] == list(shape)
                        and r["precision"] == f"int{bits}"
                        and r["scope"] == "complete_linear"
                    ]
                    best = min(rows, key=lambda r: r["median_ms"])
                    summary.append(
                        {
                            "case": case,
                            "M": shape[0],
                            "N": shape[1],
                            "K": shape[2],
                            "bits": bits,
                            "tactic": best["tactic"],
                            "median_ms": best["median_ms"],
                            "speedup_vs_bf16": best["speedup_vs_bf16"],
                        }
                    )
        with (output / "summary.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
