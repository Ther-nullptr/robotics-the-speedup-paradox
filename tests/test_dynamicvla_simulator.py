"""CPU contracts for DOM transport boundaries, independent of Isaac installation."""

import importlib
import sys

import numpy as np

from robotics_bench.dynamicvla_dom.server import (
    select_action,
    terminal_ack,
    wait_for_ack,
)


def packet(episode=3, index=4, x=1.0):
    return {
        "action": [[x, 0, 0, 1, 0, 0, 0, 1]],
        "episode_id": episode,
        "observation_index": index,
    }


def test_import_and_hold_are_cpu_only():
    before = set(sys.modules)
    importlib.reload(importlib.import_module("robotics_bench.dynamicvla_dom.server"))
    assert not any(
        name.startswith(("isaaclab", "omni")) for name in set(sys.modules) - before
    )
    value, index, rejected = select_action([], 3, 4, 2)
    assert value is None and index == 2 and rejected == 0


def test_newest_valid_action_and_episode_boundary():
    # Model may emit multiple actions for the same observation: equality is legal.
    messages = [
        packet(x=1),
        packet(episode=2),
        packet(index=5),
        packet(index=3),
        packet(x=2),
    ]
    value, index, rejected = select_action(messages, 3, 4)
    np.testing.assert_array_equal(value, [[2, 0, 0, 1, 0, 0, 0, 1]])
    assert index == 4 and rejected == 3
    assert select_action([packet(episode=3)], 4, 0)[0] is None


def test_invalid_action_cannot_replace_held_target():
    malformed = [packet(x=float("nan")), packet(index=True), packet(episode=True)]
    malformed += [{**packet(), "action": [1] * 8}, {**packet(), "action": [[0] * 8]}]
    assert select_action(malformed, 3, 4) == (None, -1, 5)


def test_ack_must_name_current_episode():
    assert terminal_ack({"ack": True, "episode_id": 3}, 3)
    assert not terminal_ack({"ack": True}, 3)
    assert not terminal_ack({"ack": True, "episode_id": 2}, 3)
    assert not terminal_ack({"ack": 1, "episode_id": 3}, 3)


def test_terminal_is_republished_until_matching_ack(monkeypatch):
    from robotics_bench.dynamicvla_dom import server

    class Socket:
        messages = []

        def send_pyobj(self, value):
            self.messages.append(value)

    class App:
        def is_running(self):
            return True

    replies = iter([[{"ack": True, "episode_id": 2}], [{"ack": True, "episode_id": 3}]])
    monkeypatch.setattr(server, "_drain", lambda _: next(replies))
    monkeypatch.setattr(server.time, "sleep", lambda _: None)
    socket = Socket()
    wait_for_ack(socket, object(), {"success": True, "episode_id": 3}, 1.0, App())
    assert len(socket.messages) == 2


def test_selected_chunk_metadata_excludes_overwritten_and_stale_messages():
    messages = [
        {**packet(x=1), "chunk_id": 1, "action_index": 4},
        {**packet(x=2), "chunk_id": 2, "action_index": 5},
        {**packet(episode=2), "chunk_id": 99, "action_index": 6},
    ]
    action, index, rejected, metadata = select_action(
        messages, 3, 4, with_metadata=True
    )
    assert action[0, 0] == 2 and index == 4 and rejected == 1
    assert metadata == {"chunk_id": 2, "action_index": 5}
    assert select_action([packet()], 3, 4, with_metadata=True)[3] == {}


def test_scene_reconstruction_handles_multiple_objects_and_cameras(monkeypatch):
    # Execute the real scene assembly functions with CPU configuration stand-ins;
    # importing the full native module requires the running Isaac application.
    import ast
    import logging
    import os
    from pathlib import Path
    from types import SimpleNamespace
    from robotics_bench.dynamicvla_dom import native
    from robotics_bench.dynamicvla_dom.native import configs

    path = Path(native.__file__).parent / "environment.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"os": os, "logging": logging}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    scene = {}

    def add_object(actual_scene, name, value):
        assert actual_scene is scene
        actual_scene[name] = value
        return actual_scene

    scene_module = SimpleNamespace(
        set_target_object=lambda current, value: add_object(current, "object", value),
        add_object=add_object,
        add_scene_camera=add_object,
        get_camera_cfg=lambda value: value,
        set_light_asset=lambda current, **kwargs: add_object(current, "light", kwargs),
    )
    object_module = SimpleNamespace(
        get_object_cfg=lambda path, state, spawn: (path, state, spawn),
        get_spawner_cfg=lambda *args: args,
    )
    monkeypatch.setattr(configs, "scene_cfg", scene_module, raising=False)
    monkeypatch.setattr(configs, "object_cfg", object_module, raising=False)
    objects = {
        name: {
            "class_type": "isaaclab.assets.rigid_object.rigid_object:RigidObject",
            "prim_path": f"/World/envs/env_.*/{name}",
            "spawn": {
                "mass_props": {"mass": 1},
                "rigid_props": {"angular_damping": 0},
                "semantic_tags": [],
            },
            "init_state": {
                "pos": [0, 0, 0],
                "rot": [1, 0, 0, 0],
                "lin_vel": [0, 0, 0],
                "ang_vel": [0, 0, 0],
            },
        }
        for name in ("object", "container")
    }
    assert namespace["_set_up_scene_objects"](scene, objects, ".") is scene
    assert scene["object"][0] == "/object"
    assert scene["container"][0] == "/container"
    cameras = {
        name: {
            "class_type": "isaaclab.sensors.camera.camera:Camera",
            "prim_path": f"/World/envs/env_.*/Robot/{name}",
            "update_period": 0.04,
            "width": 480,
            "height": 360,
            "data_types": ["rgb"],
            "spawn": {
                "focal_length": 2.3,
                "focus_distance": 400,
                "horizontal_aperture": 4.6,
                "clipping_range": [0.01, 10000],
            },
            "offset": {"pos": [0, 0, 0], "rot": [1, 0, 0, 0], "convention": "ros"},
        }
        for name in ("side_cam", "wrist_cam")
    }
    assert namespace["_set_up_scene_cameras"](scene, cameras) is scene
    assert scene["side_cam"]["prim_path"] == "/Robot/side_cam"
    assert scene["wrist_cam"]["prim_path"] == "/Robot/wrist_cam"
    assert (
        namespace["_set_up_scene_distant_light"](
            scene,
            {
                "init_state": {"pos": [0, 0, 1]},
                "spawn": {"color_temperature": 6000, "intensity": 10},
            },
        )
        is scene
    )
    assert scene["light"]["intensity"] == 10


def test_native_dropping_is_failure_not_exhausted_time_budget():
    from robotics_bench.dynamicvla_dom.server import termination_reason

    assert termination_reason(False, ["object_dropping"]) == "native_failure"
    assert termination_reason(False, ["container_dropping"]) == "native_failure"
    assert termination_reason(False, ["time_out"]) == "timeout"
    assert termination_reason(False, ["object_dropping", "time_out"]) == "timeout"
    assert termination_reason(True, ["object_picked", "time_out"]) == "success"
