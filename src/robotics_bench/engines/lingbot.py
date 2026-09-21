"""LingBot RoboTwin client; the model runs in its own Python environment."""

import importlib.util
import inspect
from pathlib import Path


def load_codec(source):
    path = Path(source) / "evaluation/robotwin/msgpack_numpy.py"
    spec = importlib.util.spec_from_file_location("_lingbot_numpy_codec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LingBotEngine:
    def __init__(self, source, port, *, timeout=600):
        from websockets.sync.client import connect

        self.codec = load_codec(source)
        options = dict(compression=None, max_size=None, open_timeout=30)
        if "proxy" in inspect.signature(connect).parameters:
            options["proxy"] = None
        self.connection = connect(f"ws://127.0.0.1:{port}", **options)
        self.timeout = timeout
        self.server_metadata = self.codec.unpackb(self.connection.recv(timeout=30))
        self._prompt = None
        self._pending_cache = False
        self._inference_count = 0

    def _request(self, value):
        self.connection.send(self.codec.packb(value))
        response = self.connection.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError("LingBot server failed: " + response)
        result = self.codec.unpackb(response)
        if not isinstance(result, dict):
            raise RuntimeError("LingBot server returned an invalid response")
        return result

    def reset(self, prompt):
        self._request({"reset": True, "prompt": prompt})
        self._prompt = prompt
        self._pending_cache = False
        self._inference_count = 0

    def infer_chunk(self, observation, prompt):
        import numpy as np

        if self._prompt != prompt or self._pending_cache:
            raise RuntimeError(
                "Reset or update observed history before the next inference"
            )
        result = self._request({"obs": observation, "prompt": prompt})
        actions = np.array(result["action"], copy=True)
        if (
            actions.shape != (16, 2, 16)
            or actions.dtype.kind != "f"
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Invalid LingBot RoboTwin action chunk")
        self._inference_count += 1
        self._pending_cache = True
        return actions

    def update_cache(self, observations, actions):
        expected = 4 if self._inference_count == 1 else 8
        if not self._pending_cache or len(observations) != expected:
            raise ValueError(
                "Observed keyframes do not match the executed LingBot chunk"
            )
        self._request(
            {
                "obs": observations,
                "state": actions,
                "compute_kv_cache": True,
                "imagine": False,
            }
        )
        self._pending_cache = False

    def close(self):
        self.connection.close()
