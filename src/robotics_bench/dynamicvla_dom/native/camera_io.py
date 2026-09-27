from collections.abc import Mapping, Sequence
from typing import Any


def camera_is_requested(name: str, camera_names: set[str] | None) -> bool:
    return camera_names is None or name in camera_names


def select_requested_camera_sensors(
    sensors: Mapping[str, Any], camera_names: set[str] | None
) -> dict[str, Any]:
    return {
        name: sensor
        for name, sensor in sensors.items()
        if camera_is_requested(name, camera_names)
    }


def select_native_camera_outputs(
    outputs: Mapping[str, Any], views: Sequence[str]
) -> dict[str, Any]:
    native_views = {
        "semantic_segmentation" if view == "seg" else view for view in views
    }
    return {name: value for name, value in outputs.items() if name in native_views}
