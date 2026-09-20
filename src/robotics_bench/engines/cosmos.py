"""Optional Cosmos Policy engine for the fixed LIBERO inference case."""

from contextlib import contextmanager
import gc
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

from .cosmos_load_guard import checkpoint_load_guard


@contextmanager
def _libero_import_context(source):
    """Bind native platform detection independently of the caller's argv."""
    previous_argv = sys.argv
    previous_path = sys.path[:]
    sys.argv = ["robotics_bench_libero"]
    sys.path.insert(0, str(source))
    try:
        yield
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


def _check_import_sources(source):
    package_root = (source / "cosmos_policy").resolve()
    for name, module in list(sys.modules.items()):
        if name == "cosmos_policy" or name.startswith("cosmos_policy."):
            origin = getattr(module, "__file__", None)
            paths = [origin] if origin else list(getattr(module, "__path__", []))
            if not paths or any(
                not Path(path).resolve().is_relative_to(package_root) for path in paths
            ):
                raise RuntimeError(
                    f"Cosmos module {name} was imported outside the requested source"
                )


def _import_native(source):
    _check_import_sources(source)
    constants = importlib.import_module("cosmos_policy.constants")
    utils = importlib.import_module("cosmos_policy.experiments.robot.cosmos_utils")
    loader = importlib.import_module("cosmos_policy._src.predict2.utils.model_loader")
    _check_import_sources(source)
    return SimpleNamespace(
        np=importlib.import_module("numpy"),
        torch=importlib.import_module("torch"),
        constants=constants,
        utils=utils,
        loader=loader,
    )


class _CPUUnpickler(pickle.Unpickler):
    """Read trusted user-supplied native Torch embedding pickles onto CPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            torch = importlib.import_module("torch")
            return lambda data: torch.load(
                io.BytesIO(data), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def _load_text_embeddings(path, descriptions, np):
    with path.open("rb") as stream:
        cache = _CPUUnpickler(stream).load()
    if not isinstance(cache, dict):
        raise ValueError("text embedding pickle must contain an instruction dictionary")
    missing = sorted(set(descriptions) - set(cache))
    if missing:
        raise ValueError(f"missing text embeddings for instructions: {missing}")
    embeddings = {}
    for description in descriptions:
        value = cache[description]
        if hasattr(value, "detach"):
            # NumPy does not support BF16; float32 represents BF16 values exactly.
            value = value.detach().cpu().float().numpy()
        value = np.array(value, copy=True)
        if (
            value.shape != (1, 512, 1024)
            or value.dtype.kind != "f"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"invalid T5 embedding for instruction {description!r}")
        embeddings[description] = value
    return embeddings


def _load_statistics(path, np):
    values = json.loads(path.read_text())
    stats = {key: np.asarray(value) for key, value in values.items()}
    for name, size in (("actions", 7), ("proprio", 9)):
        for bound in ("min", "max"):
            key = f"{name}_{bound}"
            value = stats.get(key)
            if (
                value is None
                or value.shape != (size,)
                or value.dtype.kind not in "fiu"
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"invalid dataset statistics: {key}")
        difference = stats[f"{name}_max"] - stats[f"{name}_min"]
        if (difference < 0).any() or (name == "proprio" and (difference == 0).any()):
            raise ValueError(f"invalid dataset statistics range: {name}")
    return stats


class CosmosEngine:
    """Serial LIBERO action inference using explicit local Cosmos assets.

    Images are raw uint8 HWC simulator frames. Proprio is the native nine-value
    gripper-position / end-effector-position / quaternion representation.
    Model weights and immutable task embeddings survive reset; request latents
    and action history do not. CUDA and simulation remain optional dependencies.
    """

    def __init__(
        self,
        source,
        checkpoint,
        dataset_stats,
        text_embeddings,
        vae_checkpoint,
        *,
        config_name="cosmos_predict2_2b_480p_libero__inference_only",
        num_inference_steps=5,
        action_horizon=16,
    ):
        if type(action_horizon) is not int or action_horizon != 16:
            raise ValueError("Cosmos LIBERO action_horizon must be 16")
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        self.source = Path(source).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_stats = Path(dataset_stats).expanduser().resolve()
        self.text_embeddings = Path(text_embeddings).expanduser().resolve()
        self.vae_checkpoint = Path(vae_checkpoint).expanduser().resolve()
        self.config_name = config_name
        self.num_inference_steps = num_inference_steps
        self.action_horizon = action_horizon
        self._model = None
        self._backend = None
        self._embeddings = {}
        self._stats = None
        self._episode_id = None
        self._cfg = SimpleNamespace(
            suite="libero",
            use_third_person_image=True,
            num_third_person_images=1,
            use_wrist_image=True,
            num_wrist_images=1,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            use_jpeg_compression=True,
            trained_with_image_aug=True,
            use_variance_scale=False,
            chunk_size=action_horizon,
        )
        self.metadata = {
            "engine": "cosmos_policy",
            "source": str(self.source),
            "checkpoint": str(self.checkpoint),
            "vae_checkpoint": str(self.vae_checkpoint),
            "config_name": config_name,
            "num_inference_steps": num_inference_steps,
            "action_horizon": action_horizon,
            "batch_size": 1,
            "randomize_seed": False,
            "generate_future_state_and_value_in_parallel": False,
            "preprocessing": "vertical_flip_once_then_native_jpeg_resize224_center_crop",
            "proprio": "gripper_qpos2_eef_pos3_eef_quat4_native_normalization",
            "text_embedding_scope": "requested_tasks_cpu_numpy",
            "request_latents_retained": False,
            "quantization": "none",
        }

    def load(self, task_descriptions: list[str], audit_path: Path):
        if self._model is not None:
            raise RuntimeError("CosmosEngine is already loaded")
        audit_path = Path(audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps({"status": "loading"}) + "\n")
        try:
            if os.environ.get("COSMOS_SMOKE", "0").lower() in ("true", "1", "yes", "y"):
                raise RuntimeError(
                    "COSMOS_SMOKE disables checkpoint loading and is rejected"
                )
            if not task_descriptions or any(
                not isinstance(task, str) or not task.strip()
                for task in task_descriptions
            ):
                raise ValueError(
                    "task_descriptions must contain nonempty natural-language instructions"
                )
            for path in (
                self.source / "cosmos_policy/config/config.py",
                self.checkpoint,
                self.dataset_stats,
                self.text_embeddings,
                self.vae_checkpoint,
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if self.checkpoint.suffix != ".pt":
                raise ValueError("Cosmos LIBERO requires a local .pt checkpoint")
            with _libero_import_context(self.source):
                backend = _import_native(self.source)
                constants = backend.constants
                if (
                    constants.ROBOT_PLATFORM != "LIBERO"
                    or constants.NUM_ACTIONS_CHUNK != 16
                    or constants.ACTION_DIM != 7
                    or constants.PROPRIO_DIM != 9
                ):
                    raise RuntimeError(
                        "loaded Cosmos constants do not match LIBERO 16/7/9"
                    )
                if backend.loader.SMOKE:
                    raise RuntimeError(
                        "native COSMOS_SMOKE is enabled; checkpoint load rejected"
                    )
                embeddings = _load_text_embeddings(
                    self.text_embeddings, task_descriptions, backend.np
                )
                stats = _load_statistics(self.dataset_stats, backend.np)
                if not backend.torch.cuda.is_available():
                    raise RuntimeError(
                        "Cosmos native inference requires a visible CUDA device"
                    )
                with checkpoint_load_guard(backend.loader, audit_path) as audit:
                    model, resolved_config = backend.loader.load_model_from_checkpoint(
                        experiment_name=self.config_name,
                        s3_checkpoint_dir=str(self.checkpoint),
                        config_file="cosmos_policy/config/config.py",
                        load_ema_to_reg=False,
                        experiment_opts=[
                            "model.config.tokenizer.vae_pth=" + str(self.vae_checkpoint)
                        ],
                    )
                    resolved_vae = resolved_config.model.config.tokenizer.vae_pth
                    resolved_chunk_size = (
                        resolved_config.dataloader_train.dataset.chunk_size
                    )
                    resolved_values = {
                        "tokenizer_vae_pth": str(resolved_vae),
                        "dataset_chunk_size": resolved_chunk_size,
                    }
                    audit["resolved_config"] = resolved_values
                    if Path(resolved_vae).expanduser().resolve() != self.vae_checkpoint:
                        raise RuntimeError(
                            "resolved tokenizer.vae_pth does not match the requested VAE checkpoint"
                        )
                    if (
                        type(resolved_chunk_size) is not int
                        or resolved_chunk_size != self.action_horizon
                    ):
                        raise RuntimeError(
                            "resolved dataloader_train.dataset.chunk_size must be 16"
                        )
                model.eval()
                model = model.to(backend.utils.DEVICE)
            self._backend = backend
            self._model = model
            self._embeddings = embeddings
            self._stats = stats
            self.metadata["checkpoint_audit"] = str(audit_path.resolve())
            self.metadata["loaded_task_count"] = len(embeddings)
            self.metadata["resolved_config"] = resolved_values
            self.metadata["torch_version"] = getattr(backend.torch, "__version__", None)
            if hasattr(model, "named_parameters"):
                dtypes = {}
                for _, parameter in model.named_parameters():
                    dtype = str(parameter.dtype)
                    dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
                self.metadata["parameter_dtype_numel"] = dtypes
        except BaseException as exc:
            report = json.loads(audit_path.read_text())
            report["status"] = "failed"
            report.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
            audit_path.write_text(json.dumps(report, indent=2) + "\n")
            self.close()
            raise

    def reset(self, episode_id: tuple):
        if self._model is None:
            raise RuntimeError("load CosmosEngine before reset")
        if not isinstance(episode_id, tuple) or not episode_id:
            raise ValueError("episode_id must be a nonempty tuple")
        self._episode_id = episode_id

    def infer_chunk(self, observation: dict, task: str, sampling_seed: int):
        if self._model is None:
            raise RuntimeError("load CosmosEngine before inference")
        if self._episode_id is None:
            raise RuntimeError("reset CosmosEngine before inference")
        if task not in self._embeddings:
            raise ValueError(f"task has no loaded text embedding: {task!r}")
        if type(sampling_seed) is not int or sampling_seed < 0:
            raise ValueError("sampling_seed must be a nonnegative integer")
        np = self._backend.np
        native_observation = {}
        for name in ("primary_image", "wrist_image"):
            value = observation[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.uint8
                or value.ndim != 3
                or value.shape[-1] != 3
                or min(value.shape[:2]) < 1
            ):
                raise ValueError(f"{name} must be a raw uint8 HWC RGB image")
            native_observation[name] = value[::-1].copy()
        proprio = np.asarray(observation["proprio"])
        if (
            proprio.shape != (9,)
            or proprio.dtype.kind not in "fiu"
            or not np.isfinite(proprio).all()
        ):
            raise ValueError("proprio must contain nine finite native LIBERO values")
        native_observation["proprio"] = proprio.copy()
        result = self._backend.utils.get_action(
            self._cfg,
            self._model,
            self._stats,
            native_observation,
            self._embeddings[task],
            seed=sampling_seed,
            randomize_seed=False,
            num_denoising_steps_action=self.num_inference_steps,
            generate_future_state_and_value_in_parallel=False,
            batch_size=1,
        )
        try:
            actions = np.array(result["actions"], copy=True)
        finally:
            # Native result also owns generated latent, data_batch, and T5 CUDA tensors.
            del result
        if (
            actions.shape != (self.action_horizon, 7)
            or actions.dtype.kind != "f"
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                "Cosmos actions must be a finite floating-point (16, 7) CPU array"
            )
        self.metadata["action_dtype"] = str(actions.dtype)
        return actions

    def close(self):
        backend = self._backend
        self._model = None
        self._backend = None
        self._embeddings.clear()
        self._stats = None
        self._episode_id = None
        gc.collect()
        if backend is not None:
            empty_cache = getattr(backend.torch.cuda, "empty_cache", None)
            if empty_cache is not None and backend.torch.cuda.is_available():
                empty_cache()
