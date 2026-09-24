"""Maintain a CPU-only study registry from explicit run and timing bindings.

Metrics are re-derived from completed raw ledgers. Studies remain separate;
reused runs are labeled rather than pooled into a new combined benchmark.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from statistics import median


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARY = load_file(
    Path(__file__).with_name("summarize_experiment.py"), "_record_summary"
)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def percentile(values, fraction):
    values = sorted(values)
    location = (len(values) - 1) * fraction
    low = math.floor(location)
    high = math.ceil(location)
    return values[low] + (values[high] - values[low]) * (location - low)


def chunks(run, rows, options, case_id):
    if not all("inference_calls" in row for row in rows):
        return None
    for row in rows:
        count = row["inference_calls"]
        if type(count) is not int or count < 1:
            raise ValueError("Invalid recorded chunk count")
    if case_id.startswith("cosmos_"):
        requests = defaultdict(list)
        for line in (run / "requests.jsonl").read_text().splitlines():
            if line.strip():
                event = json.loads(line)
                requests[
                    event["task"], event["init_state_id"], event["env_seed"]
                ].append(event)
        expected_keys = set()
        n = options["n_action_steps"]
        for row in rows:
            key = row["task"], row["init_state_id"], row["env_seed"]
            expected_keys.add(key)
            events = requests[key]
            if (
                len(events) != row["inference_calls"]
                or len(events) != (row["primitive_steps"] + n - 1) // n
            ):
                raise ValueError(f"Recorded chunk count disagrees with requests: {key}")
            for i, event in enumerate(events):
                offset = options.get("overlap_actions", 0) if i else 0
                if (
                    event["inference_index"],
                    event["control_step"],
                    event["observation_step"],
                    event["history_offset_steps"],
                ) != (i + 1, i * n, i * n - offset, offset):
                    raise ValueError(f"Recorded chunk history disagrees: {key}")
        if set(requests) != expected_keys:
            raise ValueError("Unexpected chunk request identities")
    successful = [row for row in rows if row["success"]]
    return (
        sum(row["inference_calls"] for row in successful) / len(successful)
        if successful
        else None
    )


def timing(binding, base, options, manifest, tier, run):
    result = dict(
        timing_status="not_requested",
        policy_median_ms=None,
        policy_p95_ms=None,
        timed_calls=None,
        timing_source=None,
        timing_gpu=None,
        timing_gpu_uuid=None,
    )
    if not binding:
        return result
    folder = resolve(binding["run_dir"], base)
    result.update(timing_status="planned", timing_source=str(folder))
    if not (folder / "manifest.json").exists():
        return result
    metadata = read(folder / "manifest.json")
    result["timing_status"] = metadata["status"]
    if metadata["status"] != "completed":
        return result
    model = metadata["model"]
    if metadata.get("case") != manifest.get("case_id"):
        raise ValueError("Timing case differs from quality run")
    initial_index = binding.get("input_episode", 0)
    if type(initial_index) is not int or initial_index < 0:
        raise ValueError("Timing input episode must be nonnegative")
    reference = Path(manifest.get("reference_run") or run)
    initial = reference / "initializations" / f"{initial_index:06d}"
    if (initial / "episode.json").exists() and metadata.get("task") is not None:
        if read(initial / "episode.json").get("description") != metadata["task"]:
            raise ValueError("Timing task instruction differs from quality reference")
    if (initial / "rendered-observation.npz").exists():
        if metadata.get("input_sha256", [])[:1] != [
            digest(initial / "rendered-observation.npz")
        ]:
            raise ValueError("Timing input differs from the declared reference episode")
    if model["steps"] != options["num_inference_steps"]:
        raise ValueError("Timing sampling steps differ from quality run")
    if model["horizon"] != options["action_horizon"]:
        raise ValueError("Timing action horizon differs from quality run")
    if model.get("runtime") != options.get("model_runtime"):
        raise ValueError("Timing model runtime differs from quality run")
    if model.get("checkpoint") != manifest.get("resources", {}).get("checkpoint"):
        raise ValueError("Timing checkpoint differs from quality run")
    entries = [
        row
        for row in read(folder / "measurements.json")
        if row["id"] == binding["variant_id"]
    ]
    if len(entries) != 1:
        raise ValueError("Timing variant must identify exactly one measurement")
    row = entries[0]
    if row.get("precision") in ("int8", "int4") or row.get("quant_tier") is not None:
        optimizations = options.get("optimizations", {})
        if set(row.get("scopes", [])) != set(optimizations.get("scopes", [])):
            raise ValueError("Timing quantization scopes differ from quality run")
        if row.get("tactic") != optimizations.get("tactic"):
            raise ValueError("Timing integer tactic differs from quality run")
    if tier is not None and row.get("quant_tier") != tier:
        raise ValueError("Timing tier differs from the declared quality tier")
    if set(row["switches"]) != set(
        options.get("optimizations", {}).get("switches", [])
    ):
        raise ValueError("Timing optimization switches differ from quality run")
    if tier is None and row.get("precision") != options.get("optimizations", {}).get(
        "precision"
    ):
        raise ValueError("Timing precision differs from quality run")
    values = row["samples_ms"]
    if not values or any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values
    ):
        raise ValueError("Invalid policy timing samples")
    observed = median(values)
    if not math.isclose(observed, row["median_ms"], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("Timing median disagrees with raw samples")
    result.update(
        policy_median_ms=observed,
        policy_p95_ms=percentile(values, 0.95),
        timed_calls=len(values),
        timing_gpu=metadata.get("gpu"),
        timing_gpu_uuid=metadata.get("gpu_uuid"),
        timing_manifest_sha256=digest(folder / "manifest.json"),
        timing_measurements_sha256=digest(folder / "measurements.json"),
    )
    return result


def collect(path):
    path = Path(path).expanduser().resolve()
    study = read(path)
    if study.get("format") != "robotics-study-v1" or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", study.get("id", "")
    ):
        raise ValueError("Invalid study format or ID")
    cells = study["cells"]
    if not cells or len({c["id"] for c in cells}) != len(cells):
        raise ValueError("Study cell IDs must be nonempty and unique")
    sources = [resolve(cell["run_dir"], path.parent) for cell in cells]
    if len(set(sources)) != len(sources):
        raise ValueError("The same run cannot appear twice inside one study")
    records = []
    for cell, run in zip(cells, sources):
        row = dict(
            study_id=study["id"],
            case_id=study["case_id"],
            cell_id=cell["id"],
            task=cell["task"],
            role=cell.get("role", "candidate"),
            tier=cell.get("tier"),
            status="planned",
            source_run=str(run),
            quality_reused_from=cell.get("quality_reused_from"),
            code_commit=cell.get("code_commit", study.get("code_commit")),
            episodes=None,
            successes=None,
            success_rate=None,
            mean_chunks_success=None,
            mean_control_steps_success=None,
            mean_control_steps_failure_budget=None,
            policy_median_ms=None,
            policy_p95_ms=None,
            timing_status="planned" if cell.get("timing") else "not_requested",
        )
        manifest_file = run / "case-manifest.json"
        if not manifest_file.exists():
            records.append(row)
            continue
        manifest = read(manifest_file)
        options = manifest.get("options", manifest.get("identity", {}).get("case", {}))
        if manifest.get("case_id", study["case_id"]) != study["case_id"]:
            raise ValueError("Case identity differs from study")
        row.update(
            status=manifest["status"],
            precision=options.get("optimizations", {}).get("precision"),
            schedule=options.get("schedule"),
            overlap_actions=options.get("overlap_actions"),
            n_action_steps=options.get("n_action_steps"),
            sampling_steps=options.get("num_inference_steps"),
            sampling_seed=options.get("seed"),
            layout_id=options.get("layout_id"),
            style_id=options.get("style_id"),
            gpu=manifest.get("gpu"),
            case_fingerprint=manifest.get("case_fingerprint"),
            source_manifest_sha256=digest(manifest_file),
            initialization_retries=options.get("initialization_retries", 0),
            reference_observations=manifest.get("reference_observations"),
        )
        if row["status"] != "completed":
            records.append(row)
            continue
        ledger, episodes = SUMMARY.read_ledger(run)
        if (
            SUMMARY.completion_status(ledger, episodes, False) != "complete"
            or len(episodes) != cell["expected_episodes"]
        ):
            raise ValueError(f"Incomplete completed-cell coverage: {cell['id']}")
        if any(episode["task"] != cell["task"] for episode in episodes):
            raise ValueError("Episode task differs from study cell")
        coverage = read(run / "coverage.json")
        stats = SUMMARY.summarize_episodes(
            episodes, max_steps=coverage.get("max_primitive_steps")
        )["overall"]
        row.update(
            episodes=stats["episodes"],
            successes=stats["successes"],
            success_rate=stats["success_rate"],
            mean_chunks_success=chunks(run, episodes, options, study["case_id"]),
            mean_control_steps_success=stats["successful_episodes"][
                "mean_control_steps"
            ],
            mean_control_steps_failure_budget=stats["failure_penalized"][
                "mean_control_steps"
            ],
            ledger_sha256=digest(ledger),
            snapshot_replayed_episodes=sum(
                e.get("initial_observation_source") == "reference_snapshot"
                for e in episodes
            ),
        )
        quant = (
            manifest.get("engine", {}).get("optimizations", {}).get("quantization", {})
        )
        if isinstance(quant, dict):
            modules = quant.get("modules", {})
            row["w4_sites"] = sum(
                "int4_" in value.get("backend", "") for value in modules.values()
            )
            row["w8_sites"] = sum(
                "int8_" in value.get("backend", "") for value in modules.values()
            )
            row["backend_map_sha256"] = hashlib.sha256(
                json.dumps(modules, sort_keys=True).encode()
            ).hexdigest()
        if cell.get("tier") is not None and study["case_id"].startswith("cosmos_"):
            from robotics_bench.optimizations.progressive import (
                cosmos_candidate_sites,
                cosmos_precision_map,
            )

            expected = cosmos_precision_map(cosmos_candidate_sites(), cell["tier"])
            actual = {
                name.replace("._checkpoint_wrapped_module", ""): value
                for name, value in quant.get("modules", {}).items()
            }
            if set(actual) != set(expected) or quant.get("skipped"):
                raise ValueError("Tier candidate coverage is incomplete")
            if any(
                actual[name]["backend"] != f"cutlass_sm80_int{bits}_s32_bf16"
                for name, bits in expected.items()
            ):
                raise ValueError("Actual precision map differs from declared tier")
        row.update(
            timing(
                cell.get("timing"),
                path.parent,
                options,
                manifest,
                cell.get("tier"),
                run,
            )
        )
        records.append(row)
    quality_complete = all(row["status"] == "completed" for row in records)
    timing_complete = all(
        row["timing_status"] in ("not_requested", "completed") for row in records
    )
    return dict(
        id=study["id"],
        case_id=study["case_id"],
        description=study.get("description", ""),
        descriptor=str(path),
        descriptor_sha256=digest(path),
        complete=quality_complete and timing_complete,
        quality_complete=quality_complete,
        timing_complete=timing_complete,
        records=records,
    )


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def register(study_path, registry):
    registry = Path(registry).expanduser().resolve()
    registry.mkdir(parents=True, exist_ok=True)
    with (registry / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = registry / "index.json"
        index = (
            read(index_path)
            if index_path.exists()
            else {"format": "robotics-study-index-v1", "studies": {}}
        )
        study = collect(study_path)
        existing = index["studies"].get(study["id"])
        if existing and existing["descriptor"] != study["descriptor"]:
            raise ValueError("Study ID is already bound to another descriptor")
        index["studies"][study["id"]] = {"descriptor": study["descriptor"]}
        studies = [collect(entry["descriptor"]) for entry in index["studies"].values()]
        rows = [row for item in studies for row in item["records"]]
        sources = defaultdict(list)
        for row in rows:
            sources[row["source_run"]].append(row)
        for shared in sources.values():
            originals = [row for row in shared if not row["quality_reused_from"]]
            if len(originals) != 1:
                raise ValueError(
                    "Shared run requires exactly one original study and explicit reuse"
                )
            original = originals[0]["study_id"]
            if any(
                row["quality_reused_from"] not in (None, original) for row in shared
            ):
                raise ValueError(
                    "Reuse source must identify the original registered study"
                )
        index["updated_at"] = datetime.now(timezone.utc).isoformat()
        for item in studies:
            index["studies"][item["id"]].update(
                {
                    k: item[k]
                    for k in (
                        "case_id",
                        "description",
                        "descriptor_sha256",
                        "complete",
                        "quality_complete",
                        "timing_complete",
                    )
                }
            )
        write_json(
            registry / "records.json",
            {
                "format": "robotics-study-records-v1",
                "updated_at": index["updated_at"],
                "aggregation": "none; study boundaries and reused sources are retained",
                "records": rows,
            },
        )
        fields = [
            "study_id",
            "case_id",
            "cell_id",
            "task",
            "role",
            "tier",
            "precision",
            "schedule",
            "overlap_actions",
            "w4_sites",
            "w8_sites",
            "status",
            "episodes",
            "successes",
            "success_rate",
            "mean_chunks_success",
            "mean_control_steps_success",
            "mean_control_steps_failure_budget",
            "timing_status",
            "policy_median_ms",
            "policy_p95_ms",
            "timed_calls",
            "gpu",
            "timing_gpu",
            "timing_gpu_uuid",
            "sampling_steps",
            "n_action_steps",
            "sampling_seed",
            "layout_id",
            "style_id",
            "snapshot_replayed_episodes",
            "code_commit",
            "quality_reused_from",
            "source_run",
            "case_fingerprint",
            "timing_source",
        ]
        temporary = registry / "records.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(registry / "records.csv")
        lines = [
            "# Robotics study records",
            "",
            "Success rate uses all completed episodes. Mean chunks uses successful episodes with recorded request counts. Policy times exclude simulation. Studies and reused endpoints are not pooled automatically.",
            "",
            "| Study | Case | Quality cells | Timing cells | Complete |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for item in studies:
            values = item["records"]
            quality = sum(row["status"] == "completed" for row in values)
            requested = [
                row for row in values if row["timing_status"] != "not_requested"
            ]
            timed = sum(row["timing_status"] == "completed" for row in requested)
            lines.append(
                f"| {item['id']} | {item['case_id']} | {quality}/{len(values)} | {timed}/{len(requested)} | {item['complete']} |"
            )
        (registry / "README.md").write_text("\n".join(lines) + "\n")
        write_json(index_path, index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        index = register(args.study, args.registry)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(
        f"Recorded {len(index['studies'])} studies: {args.registry.resolve() / 'README.md'}"
    )


if __name__ == "__main__":
    main()
