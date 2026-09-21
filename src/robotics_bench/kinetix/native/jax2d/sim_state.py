from dataclasses import field
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RigidBody:
    position: jnp.ndarray  # Centroid
    rotation: float  # Radians
    velocity: jnp.ndarray  # m/s
    angular_velocity: float  # rad/s

    inverse_mass: float  # We use 0 to denote a fixated object with infinite mass (constant velocity)
    inverse_inertia: float  # Similarly, 0 denotes an object with infinite inertia (constant angular velocity)

    friction: float
    restitution: float  # Due to baumgarte, the actual restitution is a bit higher, so setting this to 1 will cause energy to be created on collision

    collision_mode: int  # 0 == doesn't collide with 1's. 1 = normal, i.e., it collides. 2 == collides with everything (including 0's).
    active: bool

    # Polygon
    n_vertices: int  # >=3 or things blow up
    vertices: jnp.ndarray  # Clockwise or things blow up

    # Circle
    radius: float


@struct.dataclass
class CollisionManifold:
    normal: jnp.ndarray
    penetration: float
    collision_point: jnp.ndarray
    active: bool

    # Accumulated impulses
    acc_impulse_normal: float
    acc_impulse_tangent: float

    # Set at collision time to 'remember' the correct amount of bounce
    restitution_velocity_target: jnp.ndarray


@struct.dataclass
class Joint:
    a_index: int
    b_index: int
    a_relative_pos: jnp.ndarray
    b_relative_pos: jnp.ndarray
    global_position: jnp.ndarray  # Cached
    active: bool

    # Accumulated impulses
    acc_impulse: jnp.ndarray
    acc_r_impulse: jnp.ndarray

    # Motor
    motor_speed: float
    motor_power: float
    motor_on: bool

    # Revolute Joint
    motor_has_joint_limits: bool
    min_rotation: float
    max_rotation: float

    # Fixed joint
    is_fixed_joint: bool
    rotation: float


@struct.dataclass
class Thruster:
    object_index: int
    relative_position: jnp.ndarray
    rotation: float
    power: float
    global_position: jnp.ndarray  # Cached
    active: jnp.ndarray


@struct.dataclass
class SimState:
    polygon: RigidBody
    circle: RigidBody
    joint: Joint
    thruster: Thruster
    collision_matrix: jnp.ndarray

    # Impulse accumulation
    acc_rr_manifolds: CollisionManifold
    acc_cr_manifolds: CollisionManifold
    acc_cc_manifolds: CollisionManifold

    # Defaults
    gravity: jnp.ndarray


@struct.dataclass
class SimParams:
    # Timestep size
    dt: float = 1 / 60

    # Collision and joint coefficients
    slop: float = 0.01
    baumgarte_coefficient_joints_v: float = 2.0
    baumgarte_coefficient_joints_p: float = 0.7
    baumgarte_coefficient_fjoint_av: float = 2.0
    baumgarte_coefficient_rjoint_limit_av: float = 5.0
    baumgarte_coefficient_collision: float = 0.2
    joint_stiffness: float = 0.6

    # State clipping
    clip_position: float = 15
    clip_velocity: float = 100
    clip_angular_velocity: float = 50

    # Motors and thrusters
    base_motor_speed: float = 6.0  # rad/s
    base_motor_power: float = 900.0
    base_thruster_power: float = 10.0
    motor_decay_coefficient: float = 0.1
    motor_joint_limit: float = 0.1  # rad

    # Other defaults
    base_friction: float = 0.4


@struct.dataclass
class StaticSimParams:
    # State size
    num_polygons: int = 12
    num_circles: int = 12
    num_joints: int = 12
    num_thrusters: int = 12
    max_polygon_vertices: int = 4

    # Compute amount
    num_solver_iterations: int = 10
    solver_batch_size: int = 16
    do_warm_starting: bool = True
    num_static_fixated_polys: int = 4
