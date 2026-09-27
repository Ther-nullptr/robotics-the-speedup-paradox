from dataclasses import MISSING

import isaaclab.sim as sim_utils

import numpy as np


import scipy.spatial.transform

from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg

from isaaclab.sensors import CameraCfg

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg

from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: placeholder. Can be replaced by calling `set_target_object`
    # more objects can be added to the scene by calling `add_object`
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Default house asset (as background): will be populated by agent env cfg
    house: AssetBaseCfg = MISSING

    # Default lightings
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            enable_color_temperature=True,
            color_temperature=6500,
            intensity=350,
        ),
    )
    distant_light: AssetBaseCfg = MISSING

    # Default ground plane assets
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.1]),
        spawn=GroundPlaneCfg(visible=False, color=(0.0, 0.0, 0.0)),
    )


def get_camera_cfg(cam_cfg: dict, cam_extra_cfg: dict = {}) -> SceneCfg:
    for k, v in cam_extra_cfg.items():
        cam_cfg[k] = v

    # Set default values
    if "pos" not in cam_cfg:
        cam_cfg["pos"] = [0, 0, 0]
    if "quat" not in cam_cfg:
        cam_cfg["quat"] = [1, 0, 0, 0]
    if "convention" not in cam_cfg:
        cam_cfg["convention"] = "ros"

    camera_cfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}" + cam_cfg["prim_path"],
        update_period=1 / cam_cfg["fps"],
        height=cam_cfg["height"],
        width=cam_cfg["width"],
        data_types=cam_cfg["data_types"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam_cfg["focal_length"],
            focus_distance=cam_cfg["focus_distance"],
            horizontal_aperture=cam_cfg["horizontal_aperture"],
            clipping_range=(cam_cfg["clip"]["near"], cam_cfg["clip"]["far"]),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=cam_cfg["pos"], rot=cam_cfg["quat"], convention=cam_cfg["convention"]
        ),
    )
    return camera_cfg


def add_scene_camera(
    scene_cfg: SceneCfg, cam_name: str, cam_cfg: CameraCfg
) -> SceneCfg:
    scene_cfg.__setattr__(cam_name, cam_cfg)
    return scene_cfg


def set_light_asset(
    scene_cfg: SceneCfg,
    position: list = [0.0, 0.0, 0.0],
    temperature: int = 6500,
    intensity: float = 1000,
) -> SceneCfg:
    quat = get_quat_from_look_at(position, [0.0, 0.0, 0.0])

    scene_cfg.distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=quat),
        spawn=sim_utils.DistantLightCfg(
            enable_color_temperature=True,
            color_temperature=temperature,
            intensity=intensity,
        ),
    )
    return scene_cfg


def get_quat_from_look_at(cam_pos, cam_look_at):
    fwd_vec = np.array(
        [
            cam_look_at[0] - cam_pos[0],
            cam_look_at[1] - cam_pos[1],
            cam_look_at[2] - cam_pos[2],
        ]
    )
    fwd_vec /= np.linalg.norm(fwd_vec)
    up_vec = np.array([0, 0, 1])
    right_vec = np.cross(up_vec, fwd_vec)
    right_vec /= np.linalg.norm(right_vec)
    up_vec = np.cross(fwd_vec, right_vec)
    R = np.stack([fwd_vec, right_vec, up_vec], axis=1)
    quat = scipy.spatial.transform.Rotation.from_matrix(R).as_quat()
    return [quat[3], quat[0], quat[1], quat[2]]


def add_object(
    scene_cfg: SceneCfg,
    object_name: str,
    object_cfg: RigidObjectCfg | DeformableObjectCfg,
) -> SceneCfg:
    scene_cfg.__setattr__(object_name, object_cfg)
    return scene_cfg


def set_target_object(
    scene_cfg: SceneCfg, object_cfg: RigidObjectCfg | DeformableObjectCfg
) -> SceneCfg:
    scene_cfg.object = object_cfg
    return scene_cfg


def set_house_asset(
    scene_cfg: SceneCfg, scene_asset_usd_file: str, scene_offset: list = [0, 0, 0]
) -> SceneCfg:
    scene_cfg.house = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/House",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=scene_offset, rot=[0.7071068, 0.7071068, 0, 0]
        ),
        spawn=UsdFileCfg(usd_path=scene_asset_usd_file),
    )
    return scene_cfg
