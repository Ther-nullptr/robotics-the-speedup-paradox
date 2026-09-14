"""Read Cartesian trajectories and compute finite-difference metrics using stdlib."""

import argparse
import csv
import json
import math
from pathlib import Path

TIME_SCALE = {"s": 1.0, "ms": 0.001}
POSITION_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001}
DEFAULT_UNIFORM_RTOL = 1e-5
DEFAULT_UNIFORM_ATOL_S = 1e-9


def _number(value, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number; bool is not a number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validated_samples(times, positions) -> tuple[list[float], list[tuple]]:
    try:
        times = list(times)
        positions = list(positions)
    except TypeError as exc:
        raise ValueError("times and positions must be sample sequences") from exc
    if not times:
        raise ValueError("trajectory requires at least one sample")
    if len(times) != len(positions):
        raise ValueError("times and positions must have the same length")
    clean_times = [_number(value, f"time[{i}]") for i, value in enumerate(times)]
    clean_positions = []
    for i, vector in enumerate(positions):
        try:
            vector = list(vector)
        except TypeError as exc:
            raise ValueError(f"position[{i}] must contain three coordinates") from exc
        if len(vector) != 3:
            raise ValueError(f"position[{i}] must contain three coordinates")
        clean_positions.append(
            tuple(
                _number(value, f"position[{i}][{j}]") for j, value in enumerate(vector)
            )
        )
    if any(right <= left for left, right in zip(clean_times, clean_times[1:])):
        raise ValueError(
            "timestamps must be strictly increasing; no sorting is applied"
        )
    return clean_times, clean_positions


def load_trajectory(
    path, *, time_unit: str = "s", position_unit: str = "m"
) -> tuple[list[float], list[tuple[float, float, float]]]:
    """Load one CSV trajectory with t,x,y,z, converting to seconds and metres.

    Optional episode_id values must be nonempty and identical. Extra named
    columns are ignored. Inputs are neither sorted nor resampled.
    """
    if time_unit not in TIME_SCALE:
        raise ValueError(f"unsupported time unit: {time_unit!r}; choose s or ms")
    if position_unit not in POSITION_SCALE:
        raise ValueError(
            f"unsupported position unit: {position_unit!r}; choose m, cm or mm"
        )
    times = []
    positions = []
    episode_id = None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"t", "x", "y", "z"}
        if not required.issubset(fields):
            missing = ", ".join(sorted(required - set(fields)))
            raise ValueError(f"CSV is missing required columns: {missing}")
        if len(fields) != len(set(fields)):
            raise ValueError("CSV header contains duplicate columns")
        for row in reader:
            line = reader.line_num
            if None in row:
                raise ValueError(f"CSV row {line} has more fields than its header")
            if "episode_id" in fields:
                current_episode = (row.get("episode_id") or "").strip()
                if not current_episode:
                    raise ValueError(f"CSV row {line} has an empty episode_id")
                if episode_id is not None and current_episode != episode_id:
                    raise ValueError("CSV must contain a single episode_id")
                episode_id = current_episode
            values = {}
            for field in ("t", "x", "y", "z"):
                try:
                    value = float(row[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"CSV row {line} column {field} is not a number"
                    ) from exc
                scale = (
                    TIME_SCALE[time_unit]
                    if field == "t"
                    else POSITION_SCALE[position_unit]
                )
                values[field] = _number(value * scale, f"CSV row {line} column {field}")
            times.append(values["t"])
            positions.append((values["x"], values["y"], values["z"]))
    return _validated_samples(times, positions)


def _differences(vectors: list[tuple]) -> list[tuple]:
    return [
        tuple(_number(b - a, "position difference") for a, b in zip(left, right))
        for left, right in zip(vectors, vectors[1:])
    ]


def _norm(vector: tuple) -> float:
    return _number(math.hypot(*vector), "vector norm")


def _series(times: list[float], vectors: list[tuple]) -> dict:
    return {"time_s": list(times), "xyz": vectors, "norm": [_norm(v) for v in vectors]}


def _finite_sum(values, name: str) -> float:
    try:
        return _number(math.fsum(values), name)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc


def _mean_squared_magnitude(vectors: list[tuple]) -> float:
    # Average over derivative samples, never over the three coordinate axes.
    total = _finite_sum(
        (coordinate * coordinate for vector in vectors for coordinate in vector),
        "sum of squared derivative magnitudes",
    )
    return total / len(vectors)


def _divide_by_dt_power(vectors: list[tuple], dt: float, order: int) -> list[tuple]:
    result = []
    for vector in vectors:
        scaled = list(vector)
        # Repeated division avoids first forming an underflowed dt**order.
        for _ in range(order):
            scaled = [_number(value / dt, "derivative") for value in scaled]
        result.append(tuple(scaled))
    return result


def analyze_trajectory(
    times,
    positions,
    *,
    uniform_rtol: float = DEFAULT_UNIFORM_RTOL,
    uniform_atol_s: float = DEFAULT_UNIFORM_ATOL_S,
) -> dict:
    """Analyze SI samples without smoothing, interpolation, padding or sorting.

    Velocity uses actual interval lengths and midpoint timestamps. For intervals
    close to the first dt, acceleration and jerk use second/third position
    differences divided by that first dt squared/cubed. Otherwise they are null.
    """
    for name, value in (
        ("uniform_rtol", uniform_rtol),
        ("uniform_atol_s", uniform_atol_s),
    ):
        if _number(value, name) < 0:
            raise ValueError(f"{name} must be >= 0")
    times, positions = _validated_samples(times, positions)
    count = len(times)
    intervals = [_number(b - a, "time interval") for a, b in zip(times, times[1:])]
    displacement_vectors = _differences(positions)
    path_length = _finite_sum((_norm(v) for v in displacement_vectors), "path length")
    duration = _number(times[-1] - times[0], "duration")
    displacement = _norm(tuple(b - a for a, b in zip(positions[0], positions[-1])))
    uniform = (
        all(
            math.isclose(dt, intervals[0], rel_tol=uniform_rtol, abs_tol=uniform_atol_s)
            for dt in intervals[1:]
        )
        if intervals
        else None
    )
    summary = {
        "sample_count": count,
        "duration_s": duration,
        "path_length_m": path_length,
        "displacement_m": displacement,
        "mean_speed_m_s": None,
        "peak_speed_m_s": None,
        "uniform_sampling": uniform,
        "dt_s": intervals[0] if uniform else None,
        "acceleration_rms_m_s2": None,
        "jerk_mean_squared_m2_s6": None,
        "jerk_rms_m_s3": None,
        "jerk_peak_m_s3": None,
    }
    series = {
        "position": _series(times, positions),
        "velocity": None,
        "acceleration": None,
        "jerk": None,
    }
    missing = {}
    if not intervals:
        reason = "requires_at_least_2_samples"
        for key in (
            "velocity",
            "mean_speed_m_s",
            "peak_speed_m_s",
            "uniform_sampling",
            "dt_s",
        ):
            missing[key] = reason
    else:
        velocity = [
            tuple(_number(value / dt, "velocity") for value in vector)
            for vector, dt in zip(displacement_vectors, intervals)
        ]
        midpoints = [t + dt / 2 for t, dt in zip(times, intervals)]
        series["velocity"] = _series(midpoints, velocity)
        summary["mean_speed_m_s"] = _number(path_length / duration, "mean speed")
        summary["peak_speed_m_s"] = max(series["velocity"]["norm"])
        if not uniform:
            missing["dt_s"] = "nonuniform_sampling_has_no_single_dt"

    derivative_vectors = displacement_vectors
    for name, order, fields in (
        ("acceleration", 2, ("acceleration_rms_m_s2",)),
        ("jerk", 3, ("jerk_mean_squared_m2_s6", "jerk_rms_m_s3", "jerk_peak_m_s3")),
    ):
        if count < order + 1:
            reason = f"requires_at_least_{order + 1}_samples"
        elif not uniform:
            reason = "nonuniform_sampling: higher_derivatives_not_computed"
        else:
            reason = None
        if reason:
            for key in (name, *fields):
                missing[key] = reason
            continue
        derivative_vectors = _differences(derivative_vectors)
        values = _divide_by_dt_power(derivative_vectors, intervals[0], order)
        at_times = (
            times[1:-1]
            if order == 2
            else [left + (right - left) / 2 for left, right in zip(times, times[3:])]
        )
        series[name] = _series(at_times, values)
        mean_square = _mean_squared_magnitude(values)
        if name == "acceleration":
            summary["acceleration_rms_m_s2"] = math.sqrt(mean_square)
        else:
            summary["jerk_mean_squared_m2_s6"] = mean_square
            summary["jerk_rms_m_s3"] = math.sqrt(mean_square)
            summary["jerk_peak_m_s3"] = max(series[name]["norm"])
    return {"summary": summary, "series": series, "missing_reasons": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with t,x,y,z")
    parser.add_argument("--time-unit", choices=TIME_SCALE, default="s")
    parser.add_argument("--position-unit", choices=POSITION_SCALE, default="m")
    parser.add_argument(
        "--frame", help="Coordinate-frame label; no frame transform is applied"
    )
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    parser.add_argument("--include-series", action="store_true")
    args = parser.parse_args()
    try:
        if args.output is not None and (
            args.output.resolve() == args.input.resolve()
            or (
                args.output.exists()
                and args.input.exists()
                and args.output.samefile(args.input)
            )
        ):
            raise ValueError("output must not overwrite the input trajectory")
        times, positions = load_trajectory(
            args.input, time_unit=args.time_unit, position_unit=args.position_unit
        )
        result = analyze_trajectory(times, positions)
        payload = {
            "summary": result["summary"],
            "missing_reasons": result["missing_reasons"],
            "metadata": {
                "frame": args.frame,
                "units": {
                    "time": "s",
                    "position": "m",
                    "velocity": "m/s",
                    "acceleration": "m/s^2",
                    "jerk": "m/s^3",
                    "jerk_mean_squared": "m^2/s^6",
                },
                "input_units": {"time": args.time_unit, "position": args.position_unit},
                "method": {
                    "velocity": "first_position_difference / actual_dt; interval_midpoint",
                    "acceleration": "second_position_difference / first_dt^2; central_sample",
                    "jerk": "third_position_difference / first_dt^3; stencil_midpoint",
                    "jerk_mean_squared": "mean_over_samples(sum_over_xyz(jerk^2))",
                    "acceleration_rms": "sqrt(mean_over_samples(sum_over_xyz(acceleration^2)))",
                    "mean_speed": "path_length / duration; time_weighted_interval_speed",
                    "uniform_sampling": "math.isclose(each_dt, first_dt)",
                    "uniform_rtol": DEFAULT_UNIFORM_RTOL,
                    "uniform_atol_s": DEFAULT_UNIFORM_ATOL_S,
                    "filtering": "none",
                    "interpolation": "none",
                    "padding": "none",
                    "sorting": "none",
                },
            },
        }
        if args.include_series:
            payload["series"] = result["series"]
        output = (
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        )
        if args.output is None:
            print(output, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
    except (OSError, ValueError, OverflowError, csv.Error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
