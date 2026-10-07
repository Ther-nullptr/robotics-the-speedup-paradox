"""Aggregate LingBot evaluator episode reports without GPU dependencies."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


INPUT_FORMAT = "lingbot-evaluator-score-input-v1"
OUTPUT_FORMAT = "lingbot-evaluator-score-summary-v1"
STABILITY_WEIGHTS = {
    "normalized_jerk": 0.35,
    "high_frequency_ratio": 0.30,
    "sign_flip_rate": 0.20,
    "normalized_tail_std": 0.15,
}


def _number(value, name, minimum=0.0):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def _complete_mean(values):
    values = list(values)
    if not values or any(value is None for value in values):
        return None
    return math.fsum(values) / len(values)


def _config_label(value, name="config"):
    if not isinstance(value, str) or not value or not value.replace("-", "_").isalnum():
        raise ValueError(f"{name} must contain only letters, numbers, _ or -")
    return value


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def _step_map(records, name):
    if not isinstance(records, list):
        raise ValueError(f"{name} must be a list")
    result = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{name}[{index}] must be an object")
        step = _integer(record.get("step_idx", index), f"{name}[{index}].step_idx")
        if step in result:
            raise ValueError(f"{name} contains duplicate step_idx {step}")
        result[step] = record
    return result


def _sum_fields(record, fields, name):
    return math.fsum(
        _number(record[field], f"{name}.{field}") if field in record else 0.0
        for field in fields
    )


def _stability(metrics, name):
    if not isinstance(metrics, dict):
        raise ValueError(f"{name}.action_stability must be an object")
    jerk = _number(
        metrics.get("action_normalized_jerk"),
        f"{name}.action_normalized_jerk",
    )
    high_frequency = _number(
        metrics.get("action_high_freq_ratio"),
        f"{name}.action_high_freq_ratio",
    )
    sign_flip = _number(
        metrics.get("action_sign_flip_rate"),
        f"{name}.action_sign_flip_rate",
    )
    normalized_tail = _number(
        metrics.get("action_normalized_tail_std"),
        f"{name}.action_normalized_tail_std",
    )
    for field, value in (
        ("action_high_freq_ratio", high_frequency),
        ("action_sign_flip_rate", sign_flip),
    ):
        if value > 1:
            raise ValueError(f"{name}.{field} must be <= 1")
    instability = 100 * (
        STABILITY_WEIGHTS["normalized_jerk"] * jerk / (jerk + 1)
        + STABILITY_WEIGHTS["high_frequency_ratio"] * high_frequency
        + STABILITY_WEIGHTS["sign_flip_rate"] * sign_flip
        + STABILITY_WEIGHTS["normalized_tail_std"]
        * normalized_tail
        / (normalized_tail + 1)
    )
    instability = min(100.0, max(0.0, instability))
    score = 100.0 - instability
    for field, expected in (
        ("action_instability_score", instability),
        ("action_stability_score", score),
    ):
        if field in metrics and not math.isclose(
            _number(metrics[field], f"{name}.{field}"),
            expected,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError(f"{name}.{field} disagrees with its components")
    return {
        "normalized_jerk": jerk,
        "sign_flip_rate": sign_flip,
        "high_frequency_ratio": high_frequency,
        "normalized_tail_std": normalized_tail,
        "stability_score": score,
    }


def score_episode(report, config, source_sha256):
    """Validate one evaluator report and expose its episode-level metrics."""
    if not isinstance(report, dict):
        raise ValueError("episode report must be an object")
    task = report.get("task_name")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task_name must be a nonempty string")
    episode_index = _integer(report.get("episode_index"), "episode_index")
    seed = _integer(report.get("seed"), "seed")
    success = report.get("success")
    if type(success) is not bool:
        raise ValueError("success must be a boolean")
    name = f"{config}/{task}/episode={episode_index}/seed={seed}"

    denoise = report.get("denoise")
    cache = report.get("cache")
    if not isinstance(denoise, dict) or not isinstance(cache, dict):
        raise ValueError(f"{name} requires denoise and cache objects")
    for section_name, section in (("denoise", denoise), ("cache", cache)):
        if "task_name" in section and section["task_name"] != task:
            raise ValueError(f"{name} {section_name}.task_name disagrees")
        if "success" in section and section["success"] is not success:
            raise ValueError(f"{name} {section_name}.success disagrees")

    chunks = _integer(denoise.get("num_action_infers"), f"{name}.num_action_infers", 1)
    actions = _step_map(denoise.get("action_infer_reports"), f"{name}.action")
    kv_updates = _step_map(cache.get("kv_cache_step_reports"), f"{name}.kv")
    if len(actions) != chunks or set(actions) != set(kv_updates):
        raise ValueError(f"{name} action/KV step sets must match num_action_infers")

    chunk_rows = []
    control_availability = set()
    for step in sorted(actions):
        action, kv = actions[step], kv_updates[step]
        action_round_trip = _number(
            action.get("round_trip_s"), f"{name}.action[{step}].round_trip_s"
        )
        kv_round_trip = _number(
            kv.get("round_trip_s"), f"{name}.kv[{step}].round_trip_s"
        )
        control_available = "execute_actions_equivalent_s" in action
        control_availability.add(control_available)
        control = (
            _number(
                action["execute_actions_equivalent_s"],
                f"{name}.action[{step}].execute_actions_equivalent_s",
            )
            if control_available
            else None
        )
        endpoint_compute = action_round_trip + kv_round_trip
        server_compute = _sum_fields(
            action,
            ("encode_obs_s", "video_denoise_s", "action_denoise_s"),
            f"{name}.action[{step}]",
        ) + _sum_fields(
            kv,
            ("encode_obs_s", "kv_cache_latent_s", "kv_cache_action_s"),
            f"{name}.kv[{step}]",
        )
        chunk_rows.append(
            {
                "endpoint_compute_s": endpoint_compute,
                "endpoint_plus_control_s": (
                    endpoint_compute + control if control is not None else None
                ),
                "server_compute_s": server_compute,
            }
        )
    if len(control_availability) > 1:
        raise ValueError(
            f"{name} must record execute_actions_equivalent_s for every chunk or none"
        )

    stability = _stability(report.get("action_stability"), name)
    task_endpoint_compute = math.fsum(row["endpoint_compute_s"] for row in chunk_rows)
    task_endpoint_plus_control = (
        math.fsum(row["endpoint_plus_control_s"] for row in chunk_rows)
        if control_availability == {True}
        else None
    )
    episode = {
        "config": config,
        "task": task,
        "episode_index": episode_index,
        "seed": seed,
        "success": success,
        "action_chunks": chunks,
        "endpoint_compute_per_chunk_s": _mean(
            row["endpoint_compute_s"] for row in chunk_rows
        ),
        "endpoint_plus_control_per_chunk_s": _complete_mean(
            row["endpoint_plus_control_s"] for row in chunk_rows
        ),
        "server_compute_per_chunk_s": _mean(
            row["server_compute_s"] for row in chunk_rows
        ),
        "task_endpoint_compute_s": task_endpoint_compute,
        "task_endpoint_plus_control_s": task_endpoint_plus_control,
        **stability,
        "source_sha256": source_sha256,
    }
    episode["_chunk_rows"] = chunk_rows
    return episode


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def load_group(config, path):
    config = _config_label(config)
    path = Path(path)
    if path.is_dir():
        paths = sorted(path.rglob("*_analysis.json"))
    elif path.is_file():
        paths = [path]
    else:
        raise ValueError(f"input does not exist: {path}")
    if not paths:
        raise ValueError(f"no *_analysis.json reports found for {config}")
    reports = []
    for report_path in paths:
        raw = report_path.read_bytes()
        report = _read_json(report_path)
        if not isinstance(report, dict):
            raise ValueError(f"{report_path} must contain one episode object")
        reports.append((report, hashlib.sha256(raw).hexdigest()))
    return reports


def load_manifest(path):
    data = _read_json(Path(path))
    if not isinstance(data, dict) or data.get("format") != INPUT_FORMAT:
        raise ValueError(f"input manifest format must be {INPUT_FORMAT}")
    if type(data.get("synthetic")) is not bool:
        raise ValueError("input manifest synthetic must be a boolean")
    groups = data.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("input manifest groups must be a nonempty list")
    result = []
    seen_configs = set()
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{group_index}] must be an object")
        config = _config_label(group.get("config"), f"groups[{group_index}].config")
        if config in seen_configs:
            raise ValueError(f"duplicate group config: {config}")
        seen_configs.add(config)
        reports = group.get("reports")
        if not isinstance(reports, list) or not reports:
            raise ValueError(f"groups[{group_index}].reports must be nonempty")
        result.append(
            (
                config,
                [
                    (
                        report,
                        hashlib.sha256(
                            json.dumps(
                                report,
                                ensure_ascii=False,
                                allow_nan=False,
                                sort_keys=True,
                            ).encode()
                        ).hexdigest(),
                    )
                    for report in reports
                ],
            )
        )
    baseline = _config_label(data.get("baseline_config"), "baseline_config")
    return result, baseline, data["synthetic"]


def _group_summary(config, task, episodes):
    successes = [episode for episode in episodes if episode["success"]]
    success_chunks = [
        chunk for episode in successes for chunk in episode["_chunk_rows"]
    ]
    return {
        "config": config,
        "task": task,
        "trials": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes),
        "mean_success_chunks": _mean(e["action_chunks"] for e in successes),
        "mean_success_endpoint_compute_per_chunk_s": _mean(
            row["endpoint_compute_s"] for row in success_chunks
        ),
        "mean_success_endpoint_plus_control_per_chunk_s": _complete_mean(
            row["endpoint_plus_control_s"] for row in success_chunks
        ),
        "mean_success_server_compute_per_chunk_s": _mean(
            row["server_compute_s"] for row in success_chunks
        ),
        "mean_success_task_endpoint_compute_s": _mean(
            e["task_endpoint_compute_s"] for e in successes
        ),
        "mean_success_task_endpoint_plus_control_s": _complete_mean(
            e["task_endpoint_plus_control_s"] for e in successes
        ),
        "mean_success_stability_score": _mean(e["stability_score"] for e in successes),
        "mean_success_normalized_jerk": _mean(e["normalized_jerk"] for e in successes),
        "mean_success_sign_flip_rate": _mean(e["sign_flip_rate"] for e in successes),
        "mean_success_high_frequency_ratio": _mean(
            e["high_frequency_ratio"] for e in successes
        ),
        "mean_success_normalized_tail_std": _mean(
            e["normalized_tail_std"] for e in successes
        ),
    }


def summarize(episodes, baseline_config):
    seen = set()
    grouped = {}
    config_order = []
    for episode in episodes:
        key = (
            episode["config"],
            episode["task"],
            episode["episode_index"],
            episode["seed"],
        )
        if key in seen:
            raise ValueError(f"duplicate episode: {key}")
        seen.add(key)
        group_key = (episode["config"], episode["task"])
        grouped.setdefault(group_key, []).append(episode)
        if episode["config"] not in config_order:
            config_order.append(episode["config"])
    if baseline_config not in config_order:
        raise ValueError(f"baseline config not found: {baseline_config}")

    rows = [
        _group_summary(config, task, grouped[(config, task)])
        for config in config_order
        for task in sorted(t for c, t in grouped if c == config)
    ]
    baselines = {row["task"]: row for row in rows if row["config"] == baseline_config}
    metric_names = (
        "endpoint_compute_per_chunk_s",
        "endpoint_plus_control_per_chunk_s",
        "server_compute_per_chunk_s",
        "task_endpoint_compute_s",
        "task_endpoint_plus_control_s",
    )
    for row in rows:
        base = baselines.get(row["task"])
        if base is None:
            raise ValueError(f"missing {baseline_config} baseline for {row['task']}")
        row["success_rate_delta_pp"] = 100 * (
            row["success_rate"] - base["success_rate"]
        )
        for name in metric_names:
            field = f"mean_success_{name}"
            row[f"speedup_{name.removesuffix('_s')}"] = _ratio(base[field], row[field])
        missing = []
        if not row["successes"] or not base["successes"]:
            missing.append("candidate or baseline has no successful episodes")
        if (
            row["successes"]
            and base["successes"]
            and (
                row["mean_success_task_endpoint_plus_control_s"] is None
                or base["mean_success_task_endpoint_plus_control_s"] is None
            )
        ):
            missing.append("execute_actions_equivalent_s is unavailable")
        row["missing_reasons"] = missing
    return rows


def markdown(rows, baseline_config):
    lines = [
        "# LingBot evaluator episode score summary",
        "",
        f"Baseline config: `{baseline_config}`.",
        "",
        "Latency, chunk-count and stability means are conditioned on successful episodes. Per-chunk means are chunk-weighted; task and stability means are episode-weighted.",
        "Endpoint task compute is the sum of action/KV round trips, not task wall time. Endpoint plus control is reported only when every successful chunk supplies an explicit control equivalent.",
        "",
        "| Task | Config | Trials | Successes | SR (%) | Chunks | Endpoint/chunk (s) | Endpoint task (s) | Chunk speedup | Endpoint-task speedup | Stability |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(value, digits=3):
        return "N/A" if value is None else f"{value:.{digits}f}"

    for row in rows:
        lines.append(
            f"| {row['task']} | {row['config']} | {row['trials']} | "
            f"{row['successes']} | {100 * row['success_rate']:.1f} | "
            f"{fmt(row['mean_success_chunks'])} | "
            f"{fmt(row['mean_success_endpoint_compute_per_chunk_s'])} | "
            f"{fmt(row['mean_success_task_endpoint_compute_s'])} | "
            f"{fmt(row['speedup_endpoint_compute_per_chunk'])} | "
            f"{fmt(row['speedup_task_endpoint_compute'])} | "
            f"{fmt(row['mean_success_stability_score'], 2)} |"
        )
    return "\n".join(lines) + "\n"


def _public_episode(episode):
    return {key: value for key, value in episode.items() if not key.startswith("_")}


def write_outputs(output, episodes, rows, baseline_config, synthetic):
    output.mkdir(parents=True, exist_ok=False)
    public_episodes = [_public_episode(episode) for episode in episodes]
    payload = {
        "format": OUTPUT_FORMAT,
        "synthetic": synthetic,
        "baseline_config": baseline_config,
        "aggregation": {
            "success_rate": "all episodes",
            "latency_per_chunk": "successful chunks, chunk-weighted",
            "task_latency_and_stability": "successful episodes, episode-weighted",
        },
        "time_domains": {
            "endpoint_compute": "action and KV endpoint round trips",
            "endpoint_plus_control": "endpoint round trips plus evaluator-provided control equivalent; not wall time",
            "server_compute": "sum of reported server stages; skipped stages contribute zero",
        },
        "stability_definition": {
            "feature_source": "precomputed evaluator action components",
            "weights": STABILITY_WEIGHTS,
            "physical_si_jerk": False,
        },
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(
        markdown(rows, baseline_config), encoding="utf-8"
    )
    for name, data in (("summary.csv", rows), ("episodes.csv", public_episodes)):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=sorted({key for row in data for key in row})
            )
            writer.writeheader()
            writer.writerows(data)


def _parse_group(value):
    config, separator, path = value.partition("=")
    if not separator or not config or not path:
        raise ValueError("--group must be CONFIG=PATH")
    return config, Path(path).expanduser()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help=f"{INPUT_FORMAT} JSON manifest")
    inputs.add_argument(
        "--group",
        action="append",
        help="CONFIG=PATH; PATH is one report or a directory searched recursively",
    )
    parser.add_argument("--baseline", help="Baseline config; required with --group")
    parser.add_argument(
        "--synthetic", action="store_true", help="Mark --group input as synthetic"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.input:
            if args.baseline is not None or args.synthetic:
                raise ValueError("--baseline/--synthetic only apply to --group")
            groups, baseline, synthetic = load_manifest(args.input)
        else:
            if not args.baseline:
                raise ValueError("--baseline is required with --group")
            parsed = [_parse_group(value) for value in args.group]
            if len({config for config, _ in parsed}) != len(parsed):
                raise ValueError("--group config labels must be unique")
            groups = [(config, load_group(config, path)) for config, path in parsed]
            baseline, synthetic = (
                _config_label(args.baseline, "baseline"),
                args.synthetic,
            )
        episodes = [
            score_episode(report, config, digest)
            for config, reports in groups
            for report, digest in reports
        ]
        rows = summarize(episodes, baseline)
        write_outputs(args.output, episodes, rows, baseline, synthetic)
        print(
            f"Scored {len(episodes)} episodes into {len(rows)} task/config rows in {args.output}"
        )
    except (OSError, TypeError, ValueError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
