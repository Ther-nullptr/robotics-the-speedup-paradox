"""Normalize one reviewed measurement cohort for profile-visualizer."""

from datetime import datetime, timezone

from robotics_bench.optimizations.config import PRECISION_SWITCHES


def build_history(records, *, protocol, cohort, source, revision, current):
    if (
        not records
        or records[0].get("precision", "bf16") != "bf16"
        or records[0]["switches"]
    ):
        raise ValueError("The first record must be the unoptimized BF16 anchor")
    if current not in {r["id"] for r in records}:
        raise ValueError("Select an existing current run explicitly")
    anchor = records[0]["id"]
    protocol = {**protocol, "original_state_id": anchor}
    switches = sorted({s for r in records for s in r["switches"]})
    data = {
        "schema_version": 1,
        "title": protocol["label"] + " optimization history",
        "evidence_kind": "measured",
        "current_run_id": current,
        "protocols": [protocol],
        "optimizations": [
            {
                "id": s,
                "label": s.replace("_", " "),
                "scope": "precision" if s in PRECISION_SWITCHES else "shared",
            }
            for s in switches
        ],
        "states": [],
        "runs": [],
        "comparisons": [],
        "notes": [
            "Measured policy-service latency; kernel microbenchmarks and paper-model control periods are separate.",
            "Exactness covers the recorded validation inputs, not an unmeasured task suite. Quantized task quality remains pending unless explicitly assessed.",
            "Ratios use explicit baseline/candidate samples from this cohort; different rounds are never multiplied.",
        ],
    }
    for index, r in enumerate(records):
        precision = r.get("precision", "bf16")
        rid = r["id"]
        exact = r["exact"]
        classification = (
            "reference"
            if index == 0
            else (
                "lossy"
                if precision != "bf16"
                else ("exact" if exact else "numerical_exception")
            )
        )
        shared = [s for s in r["switches"] if s not in PRECISION_SWITCHES]
        shared_config = {
            "enabled_shared": sorted(shared),
            "workload": protocol["workload"],
        }
        state = {
            "id": rid,
            "label": r.get("label", rid),
            "revision": revision,
            "precision": precision,
            "classification": classification,
            "status": r.get(
                "status",
                "accepted"
                if index == 0
                else ("experimental" if precision != "bf16" or exact else "rejected"),
            ),
            "description": r.get(
                "description",
                "See source record for full settings and validation scope.",
            ),
            "shared_config": shared_config,
            "active_optimizations": r["switches"],
        }
        if index:
            state["parent_id"] = anchor
        data["states"].append(state)
        quality = {
            "status": "unmeasured"
            if precision != "bf16"
            else ("pass" if exact else "fail"),
            "reference": anchor,
            "assessment": r.get(
                "assessment",
                "Fixed-input action comparison only; full-suite quality is not assessed",
            ),
            "sources": [source],
            "metrics": [
                {
                    "name": "Max action difference",
                    "value": r["max_abs"],
                    "unit": "action",
                }
            ],
        }
        data["runs"].append(
            {
                "id": rid,
                "state_id": rid,
                "protocol_id": protocol["id"],
                "cohort_id": cohort,
                "recorded_at": r.get(
                    "recorded_at", datetime.now(timezone.utc).isoformat()
                ),
                "samples": r["samples_ms"],
                "sources": [source],
                "quality": quality,
            }
        )
        if index:
            data["comparisons"].append(
                {
                    "id": "cumulative-" + rid,
                    "label": "Versus unoptimized anchor",
                    "kind": "cumulative",
                    "baseline_run_id": anchor,
                    "candidate_run_id": rid,
                }
            )
        if precision != "bf16":
            matching = [
                s
                for s in data["states"][:-1]
                if s["precision"] == "bf16" and s["shared_config"] == shared_config
            ]
            if matching:
                data["comparisons"].append(
                    {
                        "id": "precision-" + rid,
                        "label": "Versus matched optimized BF16",
                        "kind": "matched_precision",
                        "baseline_run_id": matching[-1]["id"],
                        "candidate_run_id": rid,
                    }
                )
    return data
