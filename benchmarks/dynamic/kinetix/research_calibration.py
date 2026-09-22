"""Audit native endpoint recovery and local delay-response calibration proposals."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def isolated_thruster(static):
    import jax.numpy as jnp
    from robotics_bench.kinetix.native.kinetix.environment.env import create_empty_env

    state = create_empty_env(static)
    active = jnp.arange(static.num_circles) == 0
    thruster_active = jnp.arange(static.num_thrusters) == 0
    return state.replace(
        gravity=jnp.zeros(2),
        polygon=state.polygon.replace(active=jnp.zeros_like(state.polygon.active)),
        circle=state.circle.replace(
            active=active,
            inverse_mass=active.astype(jnp.float32),
            inverse_inertia=active.astype(jnp.float32),
            radius=jnp.full(static.num_circles, 0.1),
            position=jnp.zeros_like(state.circle.position)
            .at[0]
            .set(jnp.array([2.5, 2.5])),
        ),
        joint=state.joint.replace(active=jnp.zeros_like(state.joint.active)),
        thruster=state.thruster.replace(
            active=thruster_active,
            object_index=jnp.full(static.num_thrusters, static.num_polygons),
            relative_position=jnp.zeros_like(state.thruster.relative_position),
            rotation=jnp.zeros_like(state.thruster.rotation),
            power=thruster_active.astype(jnp.float32),
        ),
    )


def run(args):
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        SDL_VIDEODRIVER="dummy",
        PYTHONDONTWRITEBYTECODE="1",
    )
    import jax
    import numpy as np
    from robotics_bench.kinetix.calibration import (
        ResidualCalibrator,
        continuous_state,
        state_equal,
    )
    from robotics_bench.kinetix.native.kinetix.util.saving import load_from_json_file

    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "factor": args.factor,
        "fine_model": args.fine_model,
        "command_hz": 10,
        "prototype_scope": "single_native_tick_proposals_with_isolated_body_continuation",
        "fine_warm_starting": False,
        "new_anchor_and_two_endpoint_compared": True,
        "task_state_source": "native zero-delay tape replay; local probes, not delayed closed-loop evaluation",
        "devices": [str(x) for x in jax.devices()],
        "tasks": [],
        "toy": [],
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/robotics_bench/kinetix/calibration.py",
                "benchmarks/dynamic/kinetix/research_calibration.py",
            )
        },
    }
    write_json(output / "manifest.json", manifest)
    _, static, params = load_from_json_file(
        str(ROOT / "src/robotics_bench/kinetix/levels/car_launch.json")
    )
    model = ResidualCalibrator(
        static, params, factor=args.factor, fine_model=args.fine_model
    )
    old = np.zeros(static.num_motor_bindings + static.num_thruster_bindings, np.float32)
    new = old.copy()
    new[static.num_motor_bindings] = 1
    toy = isolated_thruster(static)
    period = model.period
    h = float(params.dt) / args.factor
    delays = sorted(
        set(
            [
                0,
                1e-6,
                1e-4,
                0.004,
                h - 1e-6,
                h,
                h + 1e-6,
                period / 2,
                0.025,
                period - 1e-4,
                period - 1e-6,
                period,
            ]
        )
    )
    epsilons = [float(value) / 1000 for value in args.boundary_eps_ms.split(",")]
    if any(not np.isfinite(value) or not 0 < value < h for value in epsilons):
        raise ValueError(
            "boundary-eps-ms must be positive and smaller than the fine timestep"
        )
    delays = sorted(
        set(delays + [h + sign * value for value in epsilons for sign in (-1, 1)])
    )
    manifest["probe_delays_ms"] = [value * 1000 for value in delays]
    for index, delay in enumerate(delays):
        probe = model.probe(toy, old, new, delay)
        transition = probe.require_transition()
        expected_vx = float(params.base_thruster_power) * (period - delay)
        actual_vx = float(transition[1].circle.velocity[0, 0])
        assert abs(actual_vx - expected_vx) < 2e-5, (delay, actual_vx, expected_vx)
        if delay in (0, period):
            reference = model.native.step_env(
                jax.random.key(0), toy, new if delay == 0 else old, params
            )
            assert state_equal(transition, reference)
        row = {
            "delay_ms": delay * 1000,
            "expected_vx": expected_vx,
            "actual_vx": actual_vx,
            "balanced_dx": float(probe.proposal["circle"]["position"][0, 0] - 2.5),
            "one_sided_dx": float(
                probe.one_sided_proposal["circle"]["position"][0, 0] - 2.5
            ),
            **probe.diagnostics,
        }
        manifest["toy"].append(row)
        write_json(output / "manifest.json", manifest)
        print(
            "TOY",
            index,
            f"delay_ms={delay * 1000:.6g}",
            probe.diagnostics["branch"],
            flush=True,
        )
    assert abs(manifest["toy"][-1]["balanced_dx"]) < 1e-7
    assert abs(manifest["toy"][-1]["one_sided_dx"]) > 1e-4
    # An endpoint-only bounds check misses a clipped first step followed by
    # movement back inside. This regression must remain non-resumable.
    import jax.numpy as jnp

    clipped = toy.replace(
        circle=toy.circle.replace(
            position=toy.circle.position.at[0, 0].set(14.99),
            velocity=toy.circle.velocity.at[0, 0].set(20.0),
        ),
        thruster=toy.thruster.replace(rotation=toy.thruster.rotation.at[0].set(jnp.pi)),
    )
    strong = ResidualCalibrator(
        static,
        params.replace(base_thruster_power=900.0),
        factor=args.factor,
        fine_model=args.fine_model,
    )
    strong_old = new.copy()
    strong_new = new.copy()
    strong_new[static.num_motor_bindings] = 0.9
    clipped_probe = strong.probe(clipped, strong_old, strong_new, period / 2)
    assert not clipped_probe.diagnostics["resumable"]
    assert (
        "whole_tick_motion_envelope_may_clip"
        in clipped_probe.diagnostics["unsupported_reasons"]
    )
    manifest["intermediate_clipping_regression"] = clipped_probe.diagnostics
    current = toy
    for index in range(30):
        desired = old if (index // 3) % 2 == 0 else new
        previous = old if index == 0 or ((index - 1) // 3) % 2 == 0 else new
        result = model.probe(
            current, previous, desired, 0.004 if index % 3 == 0 else 0, audit=False
        )
        current = result.require_transition()[1]
    assert int(current.timestep) == 30
    manifest["toy_continuation"] = {
        "native_ticks": 30,
        "sim_seconds": 30 * period,
        "position": np.asarray(current.circle.position[0]).tolist(),
        "finite": bool(model._finite(current)),
    }
    if args.toy_only:
        manifest["status"] = "complete"
        write_json(output / "manifest.json", manifest)
        return
    for task in args.levels.split(","):
        level, static, params = load_from_json_file(
            str(ROOT / "src/robotics_bench/kinetix/levels" / f"{task}.json")
        )
        model = ResidualCalibrator(
            static, params, factor=args.factor, fine_model=args.fine_model
        )
        tape_path = args.action_tape_root / task / f"{task}_actions.npy"
        tape = np.load(tape_path, allow_pickle=False)
        source_manifest = json.loads((tape_path.parent / "manifest.json").read_text())
        if source_manifest.get("command_hz") != 10:
            raise ValueError("Task probes require a declared 10 Hz command tape")
        if (
            tape.ndim != 2
            or tape.shape[1:] != model.native.action_space(params).shape
            or not len(tape)
            or not np.isfinite(tape).all()
        ):
            raise ValueError("Task tape has invalid shape or nonfinite commands")
        if any(
            not np.all(tape[start : min(start + 3, len(tape))] == tape[start])
            for start in range(0, len(tape), 3)
        ):
            raise ValueError(
                "A 10 Hz tape must hold each command across three native ticks"
            )
        selected_ticks = [int(value) for value in args.snapshot_ticks.split(",")]
        if any(value < 0 or value >= len(tape) for value in selected_ticks):
            raise ValueError(
                "snapshot-ticks must index the supplied native action tape"
            )
        reference_state, dispatched_state, full_old_state = level, level, level
        snapshots = {}
        exact_ticks = 0
        for tick, action in enumerate(tape):
            if tick in selected_ticks:
                snapshots[tick] = reference_state
            reference = model.native.step_env(
                jax.random.key(0), reference_state, action, params
            )
            previous = np.zeros_like(action) if tick == 0 else tape[tick - 1]
            returned = model.probe(
                dispatched_state, previous, action, 0, audit=False
            ).require_transition()
            assert state_equal(reference, returned), (task, tick)
            full_old = model.probe(
                full_old_state, action, np.zeros_like(action), model.period, audit=False
            ).require_transition()
            assert state_equal(reference, full_old), (task, tick, "full_old")
            constant = model.probe(
                reference_state, action, action, model.period / 2, audit=False
            ).require_transition()
            assert state_equal(reference, constant), (task, tick, "constant")
            reference_state, dispatched_state = reference[1], returned[1]
            full_old_state = full_old[1]
            exact_ticks += 1
            if bool(reference[3]):
                break
        task_report = {
            "task": task,
            "zero_delay_full_transition_exact_ticks": exact_ticks,
            "full_old_transition_exact_ticks": exact_ticks,
            "constant_action_transition_exact_ticks": exact_ticks,
            "input_tape_sha256": hashlib.sha256(tape_path.read_bytes()).hexdigest(),
            "probes": [],
        }
        manifest["tasks"].append(task_report)
        for tick, state in snapshots.items():
            new = tape[tick]
            old = np.zeros_like(new) if tick == 0 else tape[tick - 1]
            native_new = model.native.step_env(jax.random.key(0), state, new, params)
            base_fields = continuous_state(native_new[1])
            movable = {
                name: np.asarray(
                    getattr(state, name).active
                    & (getattr(state, name).inverse_mass > 0)
                )
                for name in ("polygon", "circle")
            }
            for index, delay in enumerate(delays):
                result = model.probe(state, old, new, delay)
                endpoint = result.diagnostics["branch"].startswith("native_")
                if endpoint:
                    action = (
                        old
                        if result.diagnostics["branch"] == "native_old_exact"
                        else new
                    )
                    expected = model.native.step_env(
                        jax.random.key(0), state, action, params
                    )
                    assert state_equal(result.require_transition(), expected)
                else:
                    try:
                        result.require_transition()
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError(
                            "An articulated task proposal must not be resumed"
                        )
                position_effect = np.concatenate(
                    [
                        result.proposal[name]["position"][movable[name]]
                        - base_fields[name]["position"][movable[name]]
                        for name in movable
                    ]
                )
                row = {
                    "native_tick": tick,
                    "delay_ms": delay * 1000,
                    "position_effect_rmse_vs_no_delay": float(
                        np.sqrt(np.mean(position_effect**2))
                    ),
                    **result.diagnostics,
                }
                task_report["probes"].append(row)
                np.savez_compressed(
                    output / f"{task}_tick{tick}_delay{index}.npz",
                    **{
                        f"{mode}_{body}_{key}": value
                        for mode, data in (
                            ("balanced", result.proposal),
                            ("one_sided", result.one_sided_proposal),
                        )
                        for body, fields in data.items()
                        for key, value in fields.items()
                    },
                )
            print(
                "TASK",
                task,
                "native_tick",
                tick,
                "probed",
                len(delays),
                "delays",
                flush=True,
            )
            write_json(output / "manifest.json", manifest)
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument(
        "--fine-model", choices=("split", "impulse-blend"), default="split"
    )
    parser.add_argument("--levels", default="car_launch,mjc_walker")
    parser.add_argument("--action-tape-root", type=Path)
    parser.add_argument("--toy-only", action="store_true")
    parser.add_argument("--snapshot-ticks", default="0,3,12,30,60")
    parser.add_argument(
        "--boundary-eps-ms",
        default="0.001",
        help="Offsets around an internal fine-grid boundary",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.factor != 2:
        parser.error("This continuation audit currently supports factor=2 only")
    if not args.toy_only and args.action_tape_root is None:
        parser.error("task probes require --action-tape-root")
    run(args)


if __name__ == "__main__":
    main()
