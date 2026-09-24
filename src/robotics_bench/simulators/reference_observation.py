"""Explicit replay of a reference reset image under a pixel-specific certificate.

Physics, XML, language, identity and proprio remain exact. This is a distinct
initial-observation protocol, not a claim of deterministic native rendering.
"""

import hashlib
import json
from pathlib import Path

IMAGE_KEYS = ("primary_image", "secondary_image", "wrist_image")
KEYS = (*IMAGE_KEYS, "proprio")
REFERENCE_KEYS = (
    "task",
    "init_state_id",
    "env_seed",
    "layout_id",
    "style_id",
    "description",
    "initial_state_sha256",
    "initial_xml_sha256",
    "initial_observation_sha256",
)


def observation_hash(observation):
    digest = hashlib.sha256()
    for key in KEYS:
        value = observation[key]
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def restore_reference_observation(directory, index, expected, actual, observation):
    """Return a fresh exact reference observation or fail closed.

    Certificates whitelist specific camera coordinates and two adjacent uint8
    values observed in reset diagnostics. No global pixel tolerance is applied.
    A missing certificate for an episode returns None, preserving native retry.
    """
    import numpy as np

    directory = Path(directory)
    certificate_path = directory / f"{index:06d}.json"
    snapshot_path = directory / f"{index:06d}.npz"
    if not certificate_path.exists() and not snapshot_path.exists():
        return None
    certificate = json.loads(certificate_path.read_text())
    if certificate.get("format") != "robocasa-reference-observation-v1":
        raise ValueError("Unknown reference observation certificate")
    reference = certificate["reference"]
    for key in REFERENCE_KEYS:
        if key not in expected or reference.get(key) != expected[key]:
            raise ValueError(f"Certificate does not bind the reference: {key}")
        if key != "initial_observation_sha256" and actual.get(key) != expected[key]:
            raise ValueError(f"Native initialization differs: {key}")
    with np.load(snapshot_path, allow_pickle=False) as data:
        if set(data.files) != set(KEYS):
            raise ValueError(
                "Snapshot must contain exactly the three cameras and proprio"
            )
        canonical = {key: np.array(data[key], copy=True) for key in KEYS}
    if observation_hash(canonical) != expected["initial_observation_sha256"]:
        raise ValueError(
            "Snapshot does not match the original reference observation hash"
        )
    if observation_hash(observation) != actual["initial_observation_sha256"]:
        raise ValueError("Rendered observation disagrees with its recorded hash")
    for key in KEYS:
        current, target = observation[key], canonical[key]
        if current.shape != target.shape or current.dtype != target.dtype:
            raise ValueError(f"Observation shape/dtype changed: {key}")
        if key in IMAGE_KEYS and (
            target.dtype != np.uint8 or target.ndim != 3 or target.shape[-1] != 3
        ):
            raise ValueError("Snapshot cameras must be RGB uint8")
    if (
        not np.isfinite(canonical["proprio"]).all()
        or observation["proprio"].tobytes() != canonical["proprio"].tobytes()
    ):
        raise ValueError("Proprio must match exactly")
    entries = certificate["allowed_pixel_variations"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
        raise ValueError("Certificate must whitelist 1..32 specific channels")
    allowed = {}
    for entry in entries:
        key, index_value, values = entry["key"], entry["index"], entry["values"]
        if (
            key not in IMAGE_KEYS
            or len(index_value) != 3
            or any(type(i) is not int or i < 0 for i in index_value)
        ):
            raise ValueError("Invalid pixel coordinate")
        coordinate = tuple(index_value)
        if any(i >= extent for i, extent in zip(coordinate, canonical[key].shape)):
            raise ValueError("Pixel coordinate outside image")
        if (
            len(values) != 2
            or any(type(v) is not int or not 0 <= v <= 255 for v in values)
            or max(values) - min(values) != 1
        ):
            raise ValueError("Whitelist values must be two adjacent uint8 values")
        if canonical[key][coordinate] not in values or (key, coordinate) in allowed:
            raise ValueError(
                "Duplicate coordinate or reference value outside whitelist"
            )
        allowed[key, coordinate] = values
    changed = 0
    for key in IMAGE_KEYS:
        for coordinate in np.argwhere(observation[key] != canonical[key]):
            coordinate = tuple(map(int, coordinate))
            if int(observation[key][coordinate]) not in allowed.get(
                (key, coordinate), ()
            ):
                raise ValueError(f"Uncertified rendered difference: {key} {coordinate}")
            changed += 1
    return canonical, {
        "source": "reference_snapshot",
        "rendered_observation_sha256": actual["initial_observation_sha256"],
        "consumed_observation_sha256": expected["initial_observation_sha256"],
        "certificate_sha256": hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "changed_channels": changed,
    }
