"""GPU checks for unchanged native stepping and genuine physical substeps."""

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def frozen_engine(revision):
    from robotics_bench.kinetix.native.jax2d import engine

    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{revision}:src/robotics_bench/kinetix/native/jax2d/engine.py",
        ],
        text=True,
    )
    cls = next(
        x
        for x in ast.parse(source).body
        if isinstance(x, ast.ClassDef) and x.name == "PhysicsEngine"
    )
    method = next(
        x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "step"
    )
    namespace = dict(vars(engine))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            "<frozen-owned-engine>",
            "exec",
        ),
        namespace,
    )
    return type(
        "FrozenPhysicsEngine", (engine.PhysicsEngine,), {"step": namespace["step"]}
    )


def motor_check():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from robotics_bench.kinetix.native.jax2d.engine import (
        PhysicsEngine,
        create_empty_sim,
        select_shape,
    )
    from robotics_bench.kinetix.native.jax2d.joint import apply_motor
    from robotics_bench.kinetix.native.jax2d.sim_state import SimParams, StaticSimParams

    static = StaticSimParams(
        num_polygons=4,
        num_static_fixated_polys=0,
        num_circles=2,
        num_joints=1,
        num_thrusters=1,
    )
    state = create_empty_sim(static, add_floor=False, add_walls_and_ceiling=False)
    state = state.replace(
        gravity=jnp.zeros(2),
        circle=state.circle.replace(
            position=jnp.array([[1.0, 2.0], [3.0, 2.0]]),
            radius=jnp.full(2, 0.05),
            active=jnp.ones(2, bool),
            inverse_mass=jnp.ones(2),
            inverse_inertia=jnp.ones(2),
        ),
        joint=state.joint.replace(
            a_index=jnp.array([4]),
            b_index=jnp.array([5]),
            active=jnp.array([True]),
            motor_on=jnp.array([True]),
            motor_power=jnp.array([3.0]),
            motor_speed=jnp.array([1.0]),
            motor_has_joint_limits=jnp.array([False]),
            is_fixed_joint=jnp.array([False]),
        ),
    )
    params = SimParams(
        joint_stiffness=0.0,
        baumgarte_coefficient_joints_p=0.0,
        baumgarte_coefficient_joints_v=0.0,
    )
    engine = PhysicsEngine(static)
    actions = jnp.array([0.001, 0.0])
    joint = jax.tree.map(lambda x: x[0], state.joint)
    impulse = jnp.asarray(
        apply_motor(
            select_shape(state, 4, static),
            select_shape(state, 5, static),
            joint,
            actions[0],
            params,
        )
    )[None]
    reference, _ = jax.jit(engine.step)(state, params, actions)
    rows = {}
    for variant in ("direct", "held", "impulse"):

        def advance(current, index):
            override = (
                None
                if variant == "direct"
                else (
                    impulse / 40
                    if variant == "held"
                    else jnp.where(index == 0, impulse, 0)
                )
            )
            result, _ = engine.step(
                current,
                params.replace(dt=params.dt / 40),
                actions,
                motor_impulses=override,
            )
            return result, None

        result, _ = jax.jit(lambda s: jax.lax.scan(advance, s, jnp.arange(40)))(state)
        velocity_error = float(
            np.max(
                np.abs(
                    np.asarray(
                        result.circle.angular_velocity
                        - reference.circle.angular_velocity
                    )
                )
            )
        )
        rotation_error = float(
            np.max(
                np.abs(np.asarray(result.circle.rotation - reference.circle.rotation))
            )
        )
        rows[variant] = {
            "angular_velocity_max_abs_error": velocity_error,
            "rotation_max_abs_error": rotation_error,
        }
    assert rows["impulse"]["angular_velocity_max_abs_error"] < 1e-6
    assert rows["impulse"]["rotation_max_abs_error"] < 1e-6
    assert rows["held"]["angular_velocity_max_abs_error"] < 1e-6
    assert rows["held"]["rotation_max_abs_error"] > 1e-6
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--reference-revision", default="9add270")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        SDL_VIDEODRIVER="dummy",
    )
    import jax
    import jax.numpy as jnp
    import numpy as np
    from refinement import make_refined_env
    from robotics_bench.kinetix.native.kinetix.util.saving import load_from_json_file

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "reference_revision": args.reference_revision,
        "isolated_motor": motor_check(),
        "tasks": [],
    }
    print(json.dumps(report["isolated_motor"]), flush=True)
    frozen = frozen_engine(args.reference_revision)
    for task in ("mjc_walker", "car_launch"):
        level, static, base = load_from_json_file(
            str(ROOT / "src/robotics_bench/kinetix/levels" / f"{task}.json")
        )
        current, _, _ = make_refined_env(static, base, 1, "direct")
        original, _, _ = make_refined_env(static, base, 1, "direct")
        original.physics_engine = frozen(static)
        action = jnp.array([0.5, -0.5, 0.5, -0.5, 0, 0])
        a, b = level, level
        for index in range(4):
            _, a, ra, da, ia = current.step_env(jax.random.key(index), a, action, base)
            _, b, rb, db, ib = original.step_env(jax.random.key(index), b, action, base)
            assert all(
                np.array_equal(np.asarray(x), np.asarray(y))
                for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
            )
            assert (
                float(ra) == float(rb)
                and bool(da) == bool(db)
                and bool(ia["GoalR"]) == bool(ib["GoalR"])
            )
        rows = []
        for factor, variant in (
            (1, "direct"),
            (40, "direct"),
            (40, "impulse_motor_collision"),
        ):
            env, params, _ = make_refined_env(static, base, factor, variant)
            commands = env.action_type.process_action(action, level, static)
            plain = env.engine_step(level, commands, params)
            recorded, trace = env.record_physics_step(level, commands, params)
            differences = [
                float(
                    np.max(
                        np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))
                    )
                )
                for x, y in zip(
                    jax.tree.leaves(plain[1]), jax.tree.leaves(recorded[1]), strict=True
                )
                if np.size(x)
            ]
            error = max(differences)
            assert error < 1e-5
            assert bool(plain[3]) == bool(recorded[3]) and bool(
                plain[4]["GoalR"]
            ) == bool(recorded[4]["GoalR"])
            rewards, goals, states, contacts = jax.device_get(trace)
            assert rewards.shape == (int(static.frame_skip), factor)
            np.savez_compressed(
                args.output_dir / f"{task}_r{factor}_{variant}_substeps.npz",
                rewards=rewards,
                goals=goals,
                contacts=contacts,
                polygon_position=states.polygon.position,
                circle_position=states.circle.position,
                joint_position=states.joint.global_position,
            )
            rows.append(
                {
                    "factor": factor,
                    "variant": variant,
                    "executed_physics_steps": int(rewards.size),
                    "physics_dt_seconds": params.dt,
                    "reward_sample_micro_indices": [0, factor],
                    "recording_state_max_abs_error": error,
                }
            )
        report["tasks"].append(
            {"task": task, "native_4_controls_exact_vs_frozen": True, "traces": rows}
        )
        print(json.dumps(report["tasks"][-1]), flush=True)
    report["status"] = "passed"
    (args.output_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
