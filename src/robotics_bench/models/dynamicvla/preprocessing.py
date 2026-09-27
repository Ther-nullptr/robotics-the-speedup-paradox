"""Raw DOM observation transforms ported from DynamicVLA scripts/inference.py."""

import numpy as np
import torch
import torchvision.transforms.v2.functional as F

from .rotations import get_rotation_vector


def transform_observations(observations, rotation, feature_config, device):
    transformed = {}
    for observation in observations:
        ee = observation["observation.state"]["end_effector"]
        pose = np.concatenate(
            [
                ee["pos"],
                get_rotation_vector(ee["quat"], rotation),
                ee.get("gripper", np.empty((1, 0))),
            ],
            axis=-1,
        ).astype(np.float32)
        result = {"observation.state": torch.from_numpy(pose).to(device)}
        for key, value in observation.items():
            if key.startswith("observation.image") and key in feature_config:
                image = (
                    torch.from_numpy(value.astype(np.float32) / 255)
                    .permute(0, 3, 1, 2)
                    .to(device)
                )
                result[key] = F.resize(image, feature_config[key].shape[-2:])
        for key, value in result.items():
            transformed.setdefault(key, []).append(value)
    batch = {key: torch.stack(value, dim=1) for key, value in transformed.items()}
    batch.update(
        {key: observations[-1].get(key) for key in ("task", "index", "dt_scale")}
    )
    return batch
