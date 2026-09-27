"""Evaluation scene reconstruction adapted from DynamicVLA; see LICENSE."""

import copy
import logging
import os
import gymnasium as gym
import torch
from . import simulation as sim


def get_test_env(
    cfg,
    num_envs,
    scene_dir,
    object_dir,
    physics_time_step,
    tolerance,
    device,
    disable_fabric,
    path_tracing,
):
    import omni.replicator.core as rep

    cfg = copy.deepcopy(cfg)
    # Create the environment
    env_cfg = _get_env_cfg(
        cfg, num_envs, scene_dir, object_dir, tolerance, device, disable_fabric
    )
    env_cfg.sim.dt = physics_time_step
    env_cfg.episode_length_s = cfg["episode_length_s"]
    env = gym.make("Robot-Env-Cfg-v0", cfg=env_cfg, seed=cfg["seed"])
    # Increase the fictional frictions of the object
    sim.set_object_material(
        env.unwrapped.scene["object"],
        n_envs=env.unwrapped.num_envs,
    )

    # Enable Path Tracing
    if path_tracing:
        rep.settings.set_render_pathtraced()

    return env


def _get_env_cfg(
    cfg, num_envs, scene_dir, object_dir, tolerance, device, disable_fabric
):
    from robotics_bench.dynamicvla_dom.native.configs import event_cfg
    from robotics_bench.dynamicvla_dom.native.configs import robot_cfg
    from robotics_bench.dynamicvla_dom.native.configs import (
        scene_cfg as scene_cfg_module,
    )
    import isaaclab_tasks
    import omni.usd
    from robotics_bench.dynamicvla_dom.native.configs import env_cfg as native_env_cfg

    gym.register(
        id="Robot-Env-Cfg-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": "robotics_bench.dynamicvla_dom.native.configs.env_cfg:EnvCfg",
        },
        disable_env_checker=True,
    )
    env_cfg = isaaclab_tasks.utils.parse_cfg.parse_env_cfg(
        "Robot-Env-Cfg-v0",
        device=device,
        num_envs=num_envs,
        use_fabric=not disable_fabric,
    )
    env_cfg.events = event_cfg.get_event_cfg(
        None
        if "perturbation" not in cfg["events"]
        else {
            "force": cfg["events"]["perturbation"]["params"]["force_range"],
            "torque": cfg["events"]["perturbation"]["params"]["torque_range"],
        }
    )
    env_cfg.terminations = _get_terimation_cfg(cfg["terminations"], tolerance, device)

    scene_usd_path = os.path.join(
        scene_dir, os.path.basename(cfg["scene"]["house"]["spawn"]["usd_path"])
    )
    logging.info("Loading scene from %s" % scene_usd_path)
    env_cfg.scene = scene_cfg_module.set_house_asset(
        env_cfg.scene, os.path.join(scene_dir, scene_usd_path)
    )
    usd_context = omni.usd.get_context()
    usd_context.new_stage()

    # Set up the robot arm
    robot_name = robot_cfg.get_robot_name(cfg["scene"]["robot"]["spawn"]["usd_path"])
    env_cfg = native_env_cfg.set_robot(
        robot_name,
        env_cfg,
        {
            "pos": cfg["scene"]["robot"]["init_state"]["pos"],
            "quat": cfg["scene"]["robot"]["init_state"]["rot"],
        },
    )
    # Set up cameras in the scene
    env_cfg.scene = _set_up_scene_cameras(env_cfg.scene, cfg["scene"])

    # Set the light intensity and color
    assert "distant_light" in cfg["scene"]
    env_cfg.scene = _set_up_scene_distant_light(
        env_cfg.scene, cfg["scene"]["distant_light"]
    )

    # Dynamically add objects / containers to scene
    env_cfg.scene = _set_up_scene_objects(env_cfg.scene, cfg["scene"], object_dir)
    return env_cfg


def _get_terimation_cfg(cfg, tolerance, device):
    from robotics_bench.dynamicvla_dom.native.configs import termination_cfg

    if "object_picked" in cfg:
        task = "pick"
        args = cfg["object_picked"]["params"]
    elif "objects_placed" in cfg:
        task = "place"
        args = cfg["objects_placed"]["params"]
    elif "object_placed" in cfg:
        task = "place"
        # Competible with single-object placement (legacy implementation)
        args = cfg["object_placed"]["params"]
        args["objects"] = ["object"]
        args["object_sizes"] = {"object": args["object_size"]}
        del args["object_size"]
    else:
        raise NotImplementedError("Unsupported termination config.")

    args["tolerance"] = tolerance
    for k, v in args.items():
        # Tensorize the arguments
        if isinstance(v, (int, float)) or (
            isinstance(v, list) and all(isinstance(x, str) for x in v)
        ):
            args[k] = v
        elif isinstance(v, list):
            args[k] = torch.tensor(v, dtype=torch.float32, device=device)
        elif isinstance(v, dict):
            args[k] = {
                _k: torch.tensor(_v, dtype=torch.float32, device=device)
                for _k, _v in v.items()
            }
        else:
            raise ValueError("Unsupported termination argument type: %s" % type(v))

    return termination_cfg.get_termination_cfg(task, args)


def _set_up_scene_cameras(scene_cfg, cfg):
    from robotics_bench.dynamicvla_dom.native.configs import (
        scene_cfg as scene_cfg_module,
    )

    for k, v in cfg.items():
        if (
            not isinstance(v, dict)
            or "class_type" not in v
            or not isinstance(v["class_type"], str)
            or not v["class_type"].startswith("isaaclab.sensors.camera")
        ):
            continue

        # Remove prefix: '/World/envs/env_.*'
        prim_path = v["prim_path"]
        prim_path = prim_path[prim_path.rfind("/Robot") :]

        scene_cfg = scene_cfg_module.add_scene_camera(
            scene_cfg,
            k,
            scene_cfg_module.get_camera_cfg(
                {
                    "prim_path": prim_path,
                    "fps": 1 / v["update_period"],
                    "width": v["width"],
                    "height": v["height"],
                    "data_types": v["data_types"],
                    "focal_length": v["spawn"]["focal_length"],
                    "focus_distance": v["spawn"]["focus_distance"],
                    "horizontal_aperture": v["spawn"]["horizontal_aperture"],
                    "clip": {
                        "near": v["spawn"]["clipping_range"][0],
                        "far": v["spawn"]["clipping_range"][1],
                    },
                    "pos": v["offset"]["pos"],
                    "quat": v["offset"]["rot"],
                    "convention": v["offset"]["convention"],
                }
            ),
        )

    return scene_cfg


def _set_up_scene_distant_light(scene_cfg, cfg):
    from robotics_bench.dynamicvla_dom.native.configs import (
        scene_cfg as scene_cfg_module,
    )

    scene_cfg = scene_cfg_module.set_light_asset(
        scene_cfg,
        position=cfg["init_state"]["pos"],
        temperature=cfg["spawn"]["color_temperature"],
        intensity=cfg["spawn"]["intensity"],
    )
    return scene_cfg


def _set_up_scene_objects(scene_cfg, cfg, object_dir):
    from robotics_bench.dynamicvla_dom.native.configs import (
        scene_cfg as scene_cfg_module,
    )
    from robotics_bench.dynamicvla_dom.native.configs import (
        object_cfg as object_cfg_module,
    )

    for k, v in cfg.items():
        if not (
            isinstance(v, dict)
            and "class_type" in v
            and v["class_type"]
            == "isaaclab.assets.rigid_object.rigid_object:RigidObject"
        ):
            continue

        usd_file_path = None
        if "usd_path" in v["spawn"]:
            usd_folder = os.path.basename(os.path.dirname(v["spawn"]["usd_path"]))
            usd_file_path = os.path.join(
                object_dir, usd_folder, os.path.basename(v["spawn"]["usd_path"])
            )
            logging.info("Loading object from %s" % usd_file_path)
            assert os.path.exists(usd_file_path)

        # Remove prefix: '/World/envs/env_.*'
        prim_path = v["prim_path"]
        prim_path = prim_path[prim_path.rfind("/") :]
        object_cfg = object_cfg_module.get_object_cfg(
            prim_path,
            {
                "pos": v["init_state"]["pos"],
                "quat": v["init_state"]["rot"],
                "lin_vel": v["init_state"]["lin_vel"],
                "ang_vel": v["init_state"]["ang_vel"],
            },
            object_cfg_module.get_spawner_cfg(
                usd_file_path,
                v["spawn"]["mass_props"]["mass"],
                v["spawn"]["rigid_props"]["angular_damping"],
                v["spawn"]["semantic_tags"],
            ),
        )
        if k == "object":
            scene_cfg = scene_cfg_module.set_target_object(scene_cfg, object_cfg)
        else:
            scene_cfg = scene_cfg_module.add_object(scene_cfg, k, object_cfg)

    return scene_cfg
