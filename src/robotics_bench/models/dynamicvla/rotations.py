# Derived from DynamicVLA utils/helpers.py; see LICENSE.
import numpy as np
import scipy.spatial.transform


def get_rotation_vector(quat, format="quat", scalar_first=True):
    if format == "quat":
        return quat.astype(np.float32)
    elif format == "euler":
        return _get_euler_angle_from_quaternion(quat, scalar_first).astype(np.float32)
    elif format == "rotvec":
        # return (
        #     scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=scalar_first)
        #     .as_rotvec()
        #     .astype(np.float32)
        # )
        # This implementation is aligned with LIBERO dataset
        return _get_axis_angle_from_quaternion(quat, scalar_first).astype(np.float32)
    else:
        raise ValueError(
            "Unsupported format: %s. Use 'quat', 'euler', or 'rotvec'." % format
        )


def _get_euler_angle_from_quaternion(quat, scalar_first=True):
    euler_angles = (
        _rotation_from_quat(quat, scalar_first)
        .as_euler("xyz", degrees=False)
        .astype(np.float32)
    )
    # Make euler angles in the range [0, 2 * pi) for rX and rZ -> Make the values continuous
    euler_angles[..., [0, 2]] = np.mod(euler_angles[..., [0, 2]], 2 * np.pi)

    return euler_angles


def _get_axis_angle_from_quaternion(quat, scalar_first=True):
    # Ref: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    # assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    if scalar_first:
        quat = quat[..., [1, 2, 3, 0]]  # wxyz to xyzw

    # Clamp w (quat[:, 3]) to [-1.0, 1.0]
    w = np.clip(quat[..., 3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)

    # Angle part in radians
    angles = 2.0 * np.arccos(w)
    # Avoid division by zero
    zero_mask = den < 1e-8

    # Normalize axis and multiply by angle
    axis = np.zeros_like(quat[..., :3])
    axis[~zero_mask] = quat[~zero_mask, :3] / den[~zero_mask, np.newaxis]
    return axis * angles[..., np.newaxis]


def _rotation_from_quat(quat, scalar_first=True):
    try:
        return scipy.spatial.transform.Rotation.from_quat(
            quat, scalar_first=scalar_first
        )
    except TypeError:
        if scalar_first:
            quat = quat[..., [1, 2, 3, 0]]
        return scipy.spatial.transform.Rotation.from_quat(quat)


def _rotation_as_quat(rotation, scalar_first=True):
    try:
        return rotation.as_quat(scalar_first=scalar_first)
    except TypeError:
        quat = rotation.as_quat()
        return quat[..., [3, 0, 1, 2]] if scalar_first else quat


def get_quaternion(rotation, format="rotvec", scalar_first=True):
    if format == "rotvec":
        return _rotation_as_quat(
            scipy.spatial.transform.Rotation.from_rotvec(rotation), scalar_first
        ).astype(np.float32)
    elif format == "euler":
        # Align with the convention in _get_euler_angle_from_quaternion (inverse)
        rotation[..., [0, 2]] = (rotation[..., [0, 2]] + np.pi) % (2 * np.pi) - np.pi
        return _rotation_as_quat(
            scipy.spatial.transform.Rotation.from_euler("xyz", rotation, degrees=False),
            scalar_first,
        ).astype(np.float32)
    elif format == "quat":
        return rotation.astype(np.float32)
    else:
        raise ValueError(
            "Unsupported format: %s. Use 'rotvec', 'euler', or 'quat'." % format
        )
