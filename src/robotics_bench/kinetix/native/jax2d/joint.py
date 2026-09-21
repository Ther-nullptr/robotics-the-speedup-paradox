import jax
import jax.numpy as jnp
from robotics_bench.kinetix.native.jax2d.sim_state import SimParams, RigidBody, Joint
from robotics_bench.kinetix.native.jax2d.maths import rmat, sv_cross, zero_to_one


def resolve_joint_warm_start(
    a: RigidBody,
    b: RigidBody,
    joint: Joint,
):
    should_resolve = (
        joint.active
        & jnp.logical_not((a.inverse_mass == 0) & (b.inverse_mass == 0))
        & a.active
        & b.active
    )

    # Impulse WS
    impulse = joint.acc_impulse

    r1 = jnp.matmul(rmat(a.rotation), joint.a_relative_pos)
    r2 = jnp.matmul(rmat(b.rotation), joint.b_relative_pos)

    a_dv = impulse * a.inverse_mass
    b_dv = -impulse * b.inverse_mass
    a_drv = a.inverse_inertia * jnp.cross(r1, impulse)
    b_drv = -b.inverse_inertia * jnp.cross(r2, impulse)

    # Rotational impulse WS
    r_impulse = joint.acc_r_impulse
    a_drv += r_impulse * a.inverse_inertia
    b_drv -= r_impulse * b.inverse_inertia

    rvals = (a_dv, b_dv, a_drv, b_drv)
    return jax.tree.map(
        lambda x: jax.lax.select(should_resolve, x, jnp.zeros_like(x)), rvals
    )


def resolve_joint(a: RigidBody, b: RigidBody, joint: Joint, sim_params: SimParams):
    should_resolve = (
        jnp.logical_not((a.inverse_mass == 0) & (b.inverse_mass == 0))
        & joint.active
        & a.active
        & b.active
    )

    # Useful values
    sum_inv_mass = zero_to_one(a.inverse_mass + b.inverse_mass)
    sum_inv_inertia = zero_to_one(a.inverse_inertia + b.inverse_inertia)

    joint_point, a_point, b_point, r_a, r_b = get_global_joint_position(
        a, b, joint.a_relative_pos, joint.b_relative_pos
    )

    # Calculate direction of impulse
    a_v = a.velocity + sv_cross(a.angular_velocity, joint_point - a.position)
    b_v = b.velocity + sv_cross(b.angular_velocity, joint_point - b.position)

    r_v = b_v - a_v

    impulse_direction = r_v
    impulse_direction /= zero_to_one(jnp.linalg.norm(impulse_direction))

    # Calculate impulse magnitude
    impulse = r_v + sim_params.baumgarte_coefficient_joints_v * (b_point - a_point)
    impulse /= zero_to_one(
        sum_inv_mass
        + (
            jnp.square(jnp.cross(r_a, impulse_direction)) * a.inverse_inertia
            + jnp.square(jnp.cross(r_b, impulse_direction)) * b.inverse_inertia
        )
    )
    impulse *= sim_params.joint_stiffness

    # Apply impulse
    a_dv = a.inverse_mass * impulse
    a_drv = jnp.cross(r_a, impulse) * a.inverse_inertia

    b_dv = -b.inverse_mass * impulse
    b_drv = -jnp.cross(r_b, impulse) * b.inverse_inertia

    # Apply positional correction
    a_dp = (
        ((b_point - a_point) / sum_inv_mass)
        * a.inverse_mass
        * sim_params.baumgarte_coefficient_joints_p
    )
    b_dp = (
        -((b_point - a_point) / sum_inv_mass)
        * b.inverse_mass
        * sim_params.baumgarte_coefficient_joints_p
    )

    # Calculate rotational impulse
    # For rotating joints with limits
    relative_rotation = b.rotation - a.rotation - joint.rotation
    target_relative_rotation = jnp.clip(
        relative_rotation, joint.min_rotation, joint.max_rotation
    )
    rj_bias = (
        relative_rotation - target_relative_rotation
    ) * sim_params.baumgarte_coefficient_rjoint_limit_av

    # For fixed joint
    fj_bias = (
        b.rotation - a.rotation - joint.rotation
    ) * sim_params.baumgarte_coefficient_fjoint_av

    # Calculate the appropriate rotational impulse
    raw_dav = b.angular_velocity + b_drv - a.angular_velocity - a_drv
    dav = jax.lax.select(
        joint.is_fixed_joint,
        raw_dav + fj_bias,
        # If the joint limit is already being sorted out (e.g. by a motor) then we don't apply the corrective impulse
        (jnp.sign(raw_dav) == jnp.sign(rj_bias)) * (raw_dav + rj_bias),
    )

    # Apply rotational impulse
    r_impulse = dav / sum_inv_inertia

    is_applying_r_impulse = joint.is_fixed_joint | (
        joint.motor_has_joint_limits & (target_relative_rotation != relative_rotation)
    )
    r_impulse *= is_applying_r_impulse

    a_drv += r_impulse * a.inverse_inertia
    b_drv -= r_impulse * b.inverse_inertia

    # We don't do impulse accumulation for joint limits
    ws_r_impulse = jax.lax.select(
        joint.is_fixed_joint, r_impulse + joint.acc_r_impulse, 0.0
    )

    rvals = (
        a_dv,
        b_dv,
        a_drv,
        b_drv,
        a_dp,
        b_dp,
        joint_point,
        impulse + joint.acc_impulse,
        ws_r_impulse,
    )
    return jax.tree.map(
        lambda x: jax.lax.select(should_resolve, x, jnp.zeros_like(x)), rvals
    )


def get_global_joint_position(
    a: RigidBody, b: RigidBody, a_relative_pos, b_relative_pos
):
    r_a = jnp.matmul(rmat(a.rotation), a_relative_pos)
    r_b = jnp.matmul(rmat(b.rotation), b_relative_pos)
    a_point = a.position + r_a
    b_point = b.position + r_b

    a_inverse_mass = zero_to_one(a.inverse_mass)
    b_inverse_mass = zero_to_one(b.inverse_mass)

    joint_point = (a_point / a_inverse_mass + b_point / b_inverse_mass) / (
        (1 / a_inverse_mass) + (1 / b_inverse_mass)
    )

    # If we have an infinite mass object we just snap to these
    joint_point = jax.lax.select(
        a.inverse_mass == 0,
        a_point,
        jax.lax.select(b.inverse_mass == 0, b_point, joint_point),
    )
    return joint_point, a_point, b_point, r_a, r_b


def apply_motor(
    a: RigidBody, b: RigidBody, joint: Joint, motor_action: float, sim_params: SimParams
):
    should_resolve = (
        jnp.logical_not((a.inverse_mass == 0) & (b.inverse_mass == 0))
        & joint.active
        & a.active
        & b.active
        & joint.motor_on
        & (motor_action != 0)
        & (jnp.logical_not(joint.is_fixed_joint))
    )

    # Calculate motor
    axial_mass = 1.0 / zero_to_one(a.inverse_inertia + b.inverse_inertia)
    motor_power = (
        sim_params.base_motor_power * joint.motor_power * sim_params.dt * axial_mass
    )
    av_target = (
        b.angular_velocity
        - a.angular_velocity
        - joint.motor_speed * motor_action * sim_params.base_motor_speed
    )
    torque_direction = jnp.tanh(av_target * sim_params.motor_decay_coefficient)

    # The motor should gradually weaken as it approaches a joint limit
    relative_rotation = b.rotation - a.rotation - joint.rotation
    target_relative_rotation = jnp.clip(
        relative_rotation,
        joint.min_rotation + sim_params.motor_joint_limit,
        joint.max_rotation - sim_params.motor_joint_limit,
    )
    rj_bias = relative_rotation - target_relative_rotation
    motor_power *= jnp.maximum(
        jax.lax.select(
            jnp.logical_or(rj_bias == 0, jnp.logical_not(joint.motor_has_joint_limits)),
            1.0,
            jnp.maximum(0.0, 1.0 - (jnp.abs(rj_bias) / sim_params.motor_joint_limit)),
        ),
        jnp.sign(rj_bias) != jnp.sign(motor_action),
    )

    # Apply rotational impulse
    a_drv = motor_power * torque_direction * a.inverse_inertia
    b_drv = -motor_power * torque_direction * b.inverse_inertia

    rvals = (a_drv, b_drv)
    return jax.tree.map(
        lambda x: jax.lax.select(should_resolve, x, jnp.zeros_like(x)), rvals
    )
