"""Evaluation state, RGB capture and MP4 helpers from DynamicVLA; see LICENSE."""

import ast
import logging
import cv2
import imageio.v3
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from . import helpers
from .camera_io import select_native_camera_outputs, select_requested_camera_sensors


def set_object_material(target_object, n_envs=1):
    materials = target_object.root_physx_view.get_material_properties()
    materials[..., 0] = 0.9  # Static friction.
    materials[..., 1] = 1.0  # Dynamic friction.
    materials[..., 2] = 0.0  # Restitution
    target_object.root_physx_view.set_material_properties(
        materials, torch.arange(n_envs)
    )


def get_curr_state(
    ee_state,
    robot_joint_pos=None,
    object_state=None,
    object_size=None,
    container_state=None,
    container_size=None,
    env_origins=None,
    robot_quat=None,
    device="cpu",
):
    # _get_merged_object_state = lambda object_state, key: torch.cat(
    #     [getattr(os, key)[i:i+1] for i, os in enumerate(object_state)], dim=0
    # )
    curr_state = {}
    if ee_state is not None:
        curr_state["end_effector"] = {
            "pos": helpers.get_robot_relative_position(
                ee_state.target_pos_w[..., 0, :] - env_origins, robot_quat
            ),
            "quat": _get_robot_relative_quaternion(
                ee_state.target_quat_w[..., 0, :], robot_quat
            ),
        }
    if robot_joint_pos is not None:
        curr_state["joints"] = robot_joint_pos
    if object_state is not None:
        # object_root_pos_w = _get_merged_object_state(object_state, "root_pos_w")
        # object_root_quat_w = _get_merged_object_state(object_state, "root_quat_w")
        # object_root_lin_vel_w = _get_merged_object_state(object_state, "root_lin_vel_w")
        curr_state["object"] = {
            "pos": helpers.get_robot_relative_position(
                object_state.root_pos_w - env_origins, robot_quat
            ),
            "quat": _get_robot_relative_quaternion(
                object_state.root_quat_w, robot_quat
            ),
            "velocity": helpers.get_robot_relative_position(
                object_state.root_lin_vel_w, robot_quat
            ),
        }
        if object_size is not None:
            curr_state["object"]["size"] = helpers.get_object_relative_bbox(
                object_size, object_state.root_quat_w, robot_quat
            )
    if container_state is not None:
        curr_state["container"] = {
            "pos": helpers.get_robot_relative_position(
                container_state.root_pos_w - env_origins, robot_quat
            ),
            "quat": _get_robot_relative_quaternion(
                container_state.root_quat_w, robot_quat
            ),
        }
    if container_size is not None:
        if "container" not in curr_state:
            curr_state["container"] = {}

        curr_state["container"]["size"] = helpers.get_object_relative_bbox(
            container_size, container_state.root_quat_w, robot_quat
        )
    if device == "cpu":
        for csk, csv in curr_state.items():
            if isinstance(csv, dict):
                for k, v in csv.items():
                    if isinstance(v, torch.Tensor):
                        curr_state[csk][k] = v.cpu().numpy()
            elif isinstance(csv, torch.Tensor):
                curr_state[csk] = csv.cpu().numpy()

    return curr_state


def _get_robot_relative_quaternion(w_quat, robot_quat):
    from isaaclab.utils.math import quat_inv, quat_mul

    return quat_mul(quat_inv(robot_quat), w_quat)


def get_camera_views(sensors, views=["rgb"], camera_names=None):
    # NOTE: import isaaclab.utils does not work
    from isaaclab.utils import convert_dict_to_backend

    cam_views = {}
    for name, sensor in select_requested_camera_sensors(sensors, camera_names).items():
        if type(sensor).__name__ == "Camera":
            cam_views[name] = convert_dict_to_backend(
                select_native_camera_outputs(sensor.data.output, views),
                backend="numpy",
            )
            # Make semantic segmentation consistent in all views
            if "semantic_segmentation" in cam_views[name]:
                cam_views[name]["seg"] = _get_semantic_segmentation(
                    cam_views[name]["semantic_segmentation"],
                    [
                        i["semantic_segmentation"]["idToLabels"]
                        for i in sensor.data.info
                    ],
                )
                # Remove the original semantic segmentation (with New Key: "seg")
                del cam_views[name]["semantic_segmentation"]

            cam_views[name] = {k: v for k, v in cam_views[name].items() if k in views}

    return cam_views


def _get_semantic_segmentation(rgba_seg_maps, semantic_tags):
    known_tags = helpers.get_semantic_tags()
    seg_maps = np.zeros_like(rgba_seg_maps[..., :1], dtype=np.uint8)

    # Iterate over each image (since the tags may not be the same for each image)
    for si, st in enumerate(semantic_tags):
        for color, tag in st.items():
            tag_name = tag["class"].upper()
            if tag_name in ["BACKGROUND", "UNLABELLED"]:
                continue
            elif tag_name not in known_tags.keys():
                logging.warning("Unknown semantic tag %s.", tag_name)
                continue

            # Convert the color string to a tuple (Unbelievable string here!)
            mask = np.all(rgba_seg_maps[si] == ast.literal_eval(color), axis=-1)
            seg_maps[si][mask] = known_tags[tag_name]

    return seg_maps


def get_frames(
    env_state, state_keys=["sm_state", "ee_pos", "object_pos", "object_vel"]
):
    MAX_DEPTH = 25

    cam_name = None
    cam_frames = {}
    for st_key, frames in env_state.items():
        cam_idx = st_key.find("_cam_")
        if cam_idx == -1:
            continue

        cam_name = st_key[:cam_idx]
        img_name = st_key[cam_idx + 5 :]
        if cam_name not in cam_frames:
            cam_frames[cam_name] = {}
        if img_name not in cam_frames[cam_name]:
            cam_frames[cam_name][img_name] = []

        for frame in frames:
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=-1)
            elif frame.ndim == 3:
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                elif frame.shape[-1] >= 3:
                    frame = frame[:, :, :3]
            else:
                raise ValueError("Unknown camera data shape: %s" % (frame.shape,))

            # Normalize the depth image to 0-255
            if img_name in ["depth", "distance_to_image_plane", "distance_to_camera"]:
                frame = np.clip(frame, 0, MAX_DEPTH)
                frame = (frame / np.max(frame) * 255).astype(np.uint8)
            if img_name in ["seg", "semantic_segmentation"]:
                # Assign a color to each semantic class
                frame = helpers.get_semantic_map(frame[..., 0])

            cam_frames[cam_name][img_name].append(frame)

    if cam_name is None:
        raise ValueError("No camera frames found in the environment state.")

    n_frames = len(cam_frames[cam_name][img_name])
    frames = [[[] for _ in range(len(cam_frames))] for _ in range(n_frames)]
    for cam_idx, cam_imgs in enumerate(cam_frames.values()):
        for img in cam_imgs.values():
            for frame_idx in range(n_frames):
                frames[frame_idx][cam_idx].append(img[frame_idx])

    for frame_idx in range(n_frames):
        frame = np.concatenate(
            [np.concatenate(r, axis=0) for r in frames[frame_idx]], axis=1
        )
        if state_keys:
            frame = _print_state_on_frame(
                frame,
                {k: env_state[k][frame_idx] for k in state_keys if k in env_state},
            )

        frames[frame_idx] = frame

    return frames


def _print_state_on_frame(frame, state):
    TEXT_MARGIN = 10
    TEXT_SCALE = 0.5
    TEXT_THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

    lines = _get_state_text(state).split("\n")
    _, img_width = frame.shape[:2]
    # Print the text on the image
    y = TEXT_MARGIN
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(
            line, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS
        )
        x = img_width - text_width - TEXT_MARGIN
        frame = cv2.putText(
            np.ascontiguousarray(frame),
            line,
            (x, y + text_height),
            TEXT_FONT,
            TEXT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        y += text_height + TEXT_MARGIN

    return frame


def _get_state_text(state):
    text = ""
    for k, v in state.items():
        k = k.replace("_", " ").title()
        if isinstance(v, (int, np.int32, np.int64)) or v.ndim == 0:
            text += "%s: %d\n" % (k, v)
        elif isinstance(v, np.ndarray):
            # Convert all quaternions to Euler angles
            if k.find("Quat") != -1:
                k = k.replace("Quat", "Rot")
                v = R.from_quat(v).as_euler("xyz", degrees=True)

            text += "%s: " % k
            text += " ".join(["%.3f" % i for i in v]) + "\n"
        else:
            raise ValueError("Unknown State Value Type: %s" % (type(v),))

    return text


def dump_video(frames, output_path, fps=24):
    if len(frames) == 0:
        return

    imageio.v3.imwrite(
        str(output_path),
        frames,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )
