"""Cosmos boundary and checkpoint audits, without a Torch dependency."""

from collections import namedtuple
import importlib
import json
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


LoadResult = namedtuple("LoadResult", "missing_keys unexpected_keys")


class Tensor:
    shape = (2, 3)


class FakeNet:
    def __init__(self, *, result=None, skip=False):
        self.result = result or LoadResult([], [])
        self.skip = skip

    def load_state_dict(self, state, strict=False, assign=False):
        return self.result


class FakeModel:
    def __init__(self, *, skip=False, result=None):
        self.net = FakeNet(result=result)
        self.skip = skip

    def state_dict(self):
        return {"net.weight": Tensor()}

    def named_modules(self):
        return [("", self), ("net", self.net)]

    def load_state_dict(self, state, strict=False, assign=False):
        # Native Cosmos returns None and can hide incomplete submodule loads.
        if not self.skip:
            self.net.load_state_dict(
                {k.removeprefix("net."): v for k, v in state.items()},
                strict=False,
                assign=assign,
            )


def guard_api():
    path = (
        Path(__file__).resolve().parents[1]
        / "src/robotics_bench/engines/cosmos_load_guard.py"
    )
    assert path.is_file(), "Cosmos checkpoint guard is not implemented"
    return importlib.import_module("robotics_bench.engines.cosmos_load_guard")


def run_guard(tmp_path, model, state, *, swallow=False):
    loader = SimpleNamespace(instantiate=lambda config: model)
    original = loader.instantiate
    path = tmp_path / "audit.json"
    try:
        with guard_api().checkpoint_load_guard(loader, path):
            loaded = loader.instantiate(None)
            try:
                loaded.load_state_dict(state, strict=False)
            except RuntimeError:
                if not swallow:
                    raise
    finally:
        assert loader.instantiate is original
        assert "load_state_dict" not in vars(model)
        assert "load_state_dict" not in vars(model.net)
    return json.loads(path.read_text())


def test_guard_accepts_complete_native_none_return(tmp_path):
    report = run_guard(tmp_path, FakeModel(), {"net.weight": Tensor()})
    assert report["status"] == "passed"
    assert report["load_calls"] == 1
    assert report["submodule_load_calls"] == {"net": 1}
    assert report["missing_keys"] == []


@pytest.mark.parametrize(
    "state, message",
    [
        ({}, "missing"),
        ({"net.weight": SimpleNamespace(shape=(3, 2))}, "shape"),
        ({"net.weight": Tensor(), "unknown": Tensor()}, "unexpected"),
        ({"net.weight": Tensor(), "net.weight_extra_state": Tensor()}, "unexpected"),
    ],
)
def test_guard_rejects_incomplete_state_even_when_caller_swallows(
    tmp_path, state, message
):
    with pytest.raises(RuntimeError, match=message):
        run_guard(tmp_path, FakeModel(), state, swallow=True)
    assert json.loads((tmp_path / "audit.json").read_text())["status"] == "failed"


def test_guard_rejects_hidden_submodule_mismatch(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        run_guard(
            tmp_path,
            FakeModel(result=LoadResult(["weight"], [])),
            {"net.weight": Tensor()},
            swallow=True,
        )


def test_guard_rejects_skipped_network_load(tmp_path):
    with pytest.raises(RuntimeError, match="net.*load"):
        run_guard(tmp_path, FakeModel(skip=True), {"net.weight": Tensor()})


def test_guard_rejects_loader_that_never_loads_checkpoint(tmp_path):
    loader = SimpleNamespace(instantiate=lambda config: FakeModel())
    with pytest.raises(RuntimeError, match="load_state_dict"):
        with guard_api().checkpoint_load_guard(loader, tmp_path / "audit.json"):
            loader.instantiate(None)


def test_guard_allows_only_verified_transformer_engine_extra_state(tmp_path):
    class TELinear:
        pass

    TELinear.__module__ = "transformer_engine.pytorch.module.linear"

    class Model(FakeModel):
        def state_dict(self):
            return {"net.weight": Tensor(), "net.linear._extra_state": None}

        def named_modules(self):
            return super().named_modules() + [("net.linear", TELinear())]

    report = run_guard(
        tmp_path,
        Model(result=LoadResult(["linear._extra_state"], [])),
        {"net.weight": Tensor()},
    )
    assert report["allowed_extra_state_keys"] == ["net.linear._extra_state"]
    assert report["status"] == "passed"


def test_extra_state_on_non_transformer_engine_module_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="unexpected"):
        run_guard(
            tmp_path,
            FakeModel(),
            {"net.weight": Tensor(), "net._extra_state": Tensor()},
        )


class MinimalA2AAttnOp:
    def __init__(self, *, learned=False, buffer=False):
        self.learned = learned
        self.buffer = buffer

    def named_parameters(self):
        return [("weight", Tensor())] if self.learned else []

    def named_buffers(self):
        return [("running", Tensor())] if self.buffer else []

    def state_dict(self):
        return dict(self.named_parameters()) | dict(self.named_buffers())


MinimalA2AAttnOp.__module__ = "cosmos_policy._src.predict2.networks.a2a_cp"
CheckpointWrapper = type("CheckpointWrapper", (), {})
CheckpointWrapper.__module__ = (
    "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
)


class WrappedAttentionModel(FakeModel):
    def __init__(self, op=None):
        self.op = op if op is not None else MinimalA2AAttnOp()
        self.wrapper = CheckpointWrapper()
        super().__init__(
            result=LoadResult(
                [],
                ["blocks.0._checkpoint_wrapped_module.cross_attn.attn_op._extra_state"],
            )
        )

    def named_modules(self):
        return super().named_modules() + [
            ("net.blocks.0", self.wrapper),
            ("net.blocks.0._checkpoint_wrapped_module.cross_attn.attn_op", self.op),
        ]


def test_legacy_attention_extra_state_matches_wrapped_stateless_operator(tmp_path):
    key = "net.blocks.0.cross_attn.attn_op._extra_state"
    report = run_guard(
        tmp_path,
        WrappedAttentionModel(),
        {"net.weight": Tensor(), key: None},
    )
    assert report["status"] == "passed"
    assert report["unexpected_keys"] == [key]
    detail = report["allowed_extra_state_details"][key]
    assert detail["module_type"].endswith("a2a_cp.MinimalA2AAttnOp")
    assert "stateless" in detail["reason"]


@pytest.mark.parametrize("variant", ["learned", "buffer", "unknown", "absent"])
def test_legacy_attention_extra_state_needs_known_existing_stateless_operator(
    tmp_path, variant
):
    if variant == "learned":
        model = WrappedAttentionModel(MinimalA2AAttnOp(learned=True))
    elif variant == "buffer":
        model = WrappedAttentionModel(MinimalA2AAttnOp(buffer=True))
    elif variant == "unknown":
        model = WrappedAttentionModel(SimpleNamespace())
    else:
        model = FakeModel()
    with pytest.raises(RuntimeError, match="unexpected"):
        run_guard(
            tmp_path,
            model,
            {
                "net.weight": Tensor(),
                "net.blocks.0.cross_attn.attn_op._extra_state": None,
            },
        )


def test_legacy_attention_compatibility_never_allows_learned_keys(tmp_path):
    with pytest.raises(RuntimeError, match="unexpected"):
        run_guard(
            tmp_path,
            WrappedAttentionModel(),
            {
                "net.weight": Tensor(),
                "net.blocks.0.cross_attn.attn_op.weight": Tensor(),
            },
        )


def test_transformer_engine_extra_state_uses_checkpoint_wrapper_key_names(tmp_path):
    class TELinear:
        pass

    TELinear.__module__ = "transformer_engine.pytorch.module.linear"

    class Model(WrappedAttentionModel):
        def state_dict(self):
            return {
                "net.weight": Tensor(),
                "net.blocks.0.cross_attn.k_norm._extra_state": None,
            }

        def named_modules(self):
            return super().named_modules() + [
                (
                    "net.blocks.0._checkpoint_wrapped_module.cross_attn.k_norm",
                    TELinear(),
                )
            ]

    model = Model()
    model.net.result = LoadResult(
        ["blocks.0._checkpoint_wrapped_module.cross_attn.k_norm._extra_state"], []
    )
    report = run_guard(tmp_path, model, {"net.weight": Tensor()})
    assert report["status"] == "passed"
    assert report["missing_keys"] == ["net.blocks.0.cross_attn.k_norm._extra_state"]


def test_module_import_does_not_load_optional_dependencies():
    source = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import robotics_bench.engines.cosmos; "
        "assert not any(k in sys.modules for k in "
        "('torch', 'numpy', 'cosmos_policy', 'libero'))"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def engine_api():
    module = importlib.import_module("robotics_bench.engines.cosmos")
    assert hasattr(module, "CosmosEngine"), "CosmosEngine is not implemented"
    return module


def test_libero_import_context_restores_process_state_on_exception(tmp_path):
    module = engine_api()
    argv, path = sys.argv, sys.path[:]
    cwd = Path.cwd()
    with pytest.raises(RuntimeError, match="import failure"):
        with module._libero_import_context(tmp_path):
            assert sys.argv == ["robotics_bench_libero"]
            assert sys.path[0] == str(tmp_path)
            assert Path.cwd() == tmp_path
            raise RuntimeError("import failure")
    assert sys.argv is argv
    assert sys.path == path
    assert Path.cwd() == cwd


def test_libero_import_context_restores_cwd_after_success(tmp_path):
    module = engine_api()
    cwd = Path.cwd()
    with module._libero_import_context(tmp_path):
        assert Path.cwd() == tmp_path
    assert Path.cwd() == cwd


def test_preimported_cosmos_from_another_source_is_rejected(tmp_path, monkeypatch):
    module = engine_api()
    monkeypatch.setitem(
        sys.modules,
        "cosmos_policy",
        SimpleNamespace(__file__=str(tmp_path / "other/cosmos_policy/__init__.py")),
    )
    with pytest.raises(RuntimeError, match="outside the requested source"):
        module._check_import_sources(tmp_path / "requested")


@pytest.fixture
def native_boundary(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    module = engine_api()
    source = tmp_path / "source"
    (source / "cosmos_policy/config").mkdir(parents=True)
    (source / "cosmos_policy/config/config.py").write_text("")
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    vae = tmp_path / "vae.pt"
    vae.touch()
    stats = tmp_path / "statistics.json"
    stats.write_text(
        json.dumps(
            {
                "actions_min": [-1] * 7,
                "actions_max": [1] * 7,
                "proprio_min": [-1] * 9,
                "proprio_max": [1] * 9,
            }
        )
    )
    embeddings = tmp_path / "embeddings.pkl"
    embeddings.write_bytes(
        pickle.dumps({"pick up cup": np.zeros((1, 512, 1024), dtype=np.float32)})
    )
    model = FakeModel()
    model.eval = lambda: model
    model.to = lambda device: model
    calls = []
    loader = SimpleNamespace(instantiate=lambda config: model, SMOKE=False)
    resolved_config = SimpleNamespace(
        checkpoint=SimpleNamespace(load_path=str(checkpoint)),
        model=SimpleNamespace(
            config=SimpleNamespace(tokenizer=SimpleNamespace(vae_pth=str(vae)))
        ),
        dataloader_train=SimpleNamespace(dataset=SimpleNamespace(chunk_size=16)),
    )

    def load_model(**kwargs):
        calls.append(("load", kwargs, sys.argv[:], Path.cwd()))
        loaded = loader.instantiate(None)
        loaded.load_state_dict({"net.weight": Tensor()}, strict=False)
        return loaded, resolved_config

    loader.load_model_from_checkpoint = load_model

    def get_action(cfg, loaded, dataset_stats, obs, embedding, **kwargs):
        calls.append(("action", cfg, obs, embedding, kwargs))
        return {"actions": np.ones((16, 7), dtype=np.float64)}

    backend = SimpleNamespace(
        checkpoint_registry=SimpleNamespace(get_checkpoint_path=lambda path: path),
        np=np,
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        utils=SimpleNamespace(get_action=get_action, DEVICE="cuda:0"),
        loader=loader,
        constants=SimpleNamespace(
            ROBOT_PLATFORM="LIBERO", NUM_ACTIONS_CHUNK=16, ACTION_DIM=7, PROPRIO_DIM=9
        ),
    )
    monkeypatch.setattr(module, "_import_native", lambda source: backend)
    engine = module.CosmosEngine(source, checkpoint, stats, embeddings, vae)
    return SimpleNamespace(
        engine=engine,
        np=np,
        module=module,
        backend=backend,
        calls=calls,
        audit=tmp_path / "load.json",
        embeddings=embeddings,
        resolved_config=resolved_config,
    )


def load_engine(boundary):
    boundary.engine.load(["pick up cup"], boundary.audit)
    boundary.engine.reset(("libero_object", 0, 0))
    return boundary.engine


def observation(np):
    return {
        "primary_image": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        "wrist_image": np.arange(18, 36, dtype=np.uint8).reshape(2, 3, 3),
        "proprio": np.arange(9, dtype=np.float64),
    }


def test_load_uses_native_config_vae_override_and_restores_argv(native_boundary):
    b = native_boundary
    before = sys.argv[:]
    engine = load_engine(b)
    load = b.calls[0]
    assert load[1]["config_file"] == "cosmos_policy/config/config.py"
    assert load[1]["experiment_opts"] == [
        "model.config.tokenizer.vae_pth=" + str(engine.vae_checkpoint)
    ]
    assert "libero" in " ".join(load[2])
    assert load[3] == engine.source
    assert sys.argv == before
    assert json.loads(b.audit.read_text())["status"] == "passed"
    assert engine.metadata["batch_size"] == 1
    assert engine.metadata["resolved_config"] == {
        "checkpoint_load_path": str(engine.checkpoint),
        "tokenizer_vae_pth": str(engine.vae_checkpoint),
        "dataset_chunk_size": 16,
    }


@pytest.mark.parametrize("fail", [False, True])
def test_relative_audit_remains_in_callers_directory(
    native_boundary, tmp_path, monkeypatch, fail
):
    b = native_boundary
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    if fail:
        b.resolved_config.dataloader_train.dataset.chunk_size = 32
        with pytest.raises(RuntimeError, match="chunk_size"):
            b.engine.load(["pick up cup"], Path("artifacts/load.json"))
    else:
        b.engine.load(["pick up cup"], Path("artifacts/load.json"))
    assert Path.cwd() == caller
    audit = json.loads((caller / "artifacts/load.json").read_text())
    assert audit["status"] == ("failed" if fail else "passed")
    assert not (b.engine.source / "artifacts").exists()


@pytest.mark.parametrize("field", ["vae_pth", "chunk_size", "checkpoint_load_path"])
def test_resolved_configuration_mismatch_rejects_loaded_model(native_boundary, field):
    b = native_boundary
    if field == "vae_pth":
        b.resolved_config.model.config.tokenizer.vae_pth = "/ignored/override.pth"
    elif field == "chunk_size":
        b.resolved_config.dataloader_train.dataset.chunk_size = 32
    else:
        b.resolved_config.checkpoint.load_path = "/wrong/model.pt"
    with pytest.raises(RuntimeError, match=field):
        b.engine.load(["pick up cup"], b.audit)
    assert json.loads(b.audit.read_text())["status"] == "failed"
    with pytest.raises(RuntimeError, match="load"):
        b.engine.reset(("libero_object", 0, 0))


@pytest.mark.parametrize("fail", [False, True])
def test_registration_binds_only_unused_base_reference_and_restores(tmp_path, fail):
    module = importlib.import_module("robotics_bench.engines.cosmos")
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"real policy fixture")
    calls = []

    def original(path):
        calls.append(path)
        raise ValueError("unavailable original reference")

    registry = SimpleNamespace(get_checkpoint_path=original)
    try:
        with module._checkpoint_registration_context(registry, checkpoint) as bindings:
            for path in module.BASE_CHECKPOINT_REFERENCES:
                assert registry.get_checkpoint_path(path) == str(checkpoint)
            assert len(bindings) == 2
            with pytest.raises(ValueError, match="original"):
                registry.get_checkpoint_path("hf://other/model/model.pt")
            if fail:
                raise RuntimeError("config failed")
    except RuntimeError:
        assert fail
    assert calls == ["hf://other/model/model.pt"]
    assert registry.get_checkpoint_path is original


def test_infer_flips_once_preserves_dtype_and_native_preprocessing(native_boundary):
    b = native_boundary
    engine = load_engine(b)
    obs = observation(b.np)
    images_before = {key: value.copy() for key, value in obs.items()}
    actions = engine.infer_chunk(obs, "pick up cup", 37)
    _, cfg, native_obs, embedding, kwargs = b.calls[-1]
    for name in ("primary_image", "wrist_image"):
        b.np.testing.assert_array_equal(native_obs[name], images_before[name][::-1])
        b.np.testing.assert_array_equal(obs[name], images_before[name])
        assert native_obs[name].flags.c_contiguous
    b.np.testing.assert_array_equal(native_obs["proprio"], obs["proprio"])
    assert cfg.use_jpeg_compression and cfg.trained_with_image_aug
    assert cfg.normalize_proprio and cfg.unnormalize_actions
    assert cfg.chunk_size == 16
    assert isinstance(embedding, b.np.ndarray)
    assert kwargs == {
        "seed": 37,
        "randomize_seed": False,
        "num_denoising_steps_action": 5,
        "generate_future_state_and_value_in_parallel": False,
        "batch_size": 1,
    }
    assert actions.shape == (16, 7)
    assert actions.dtype == b.np.float64
    assert engine.metadata["action_dtype"] == "float64"


def test_infer_releases_native_latent_result(native_boundary):
    import weakref

    b = native_boundary
    engine = load_engine(b)
    refs = []

    class Latent:
        pass

    def get_action(*args, **kwargs):
        latent = Latent()
        refs.append(weakref.ref(latent))
        return {"actions": b.np.zeros((16, 7)), "generated_latent": latent}

    b.backend.utils.get_action = get_action
    engine.infer_chunk(observation(b.np), "pick up cup", 7)
    assert refs[0]() is None
    engine.reset(("libero_object", 0, 1))
    engine.infer_chunk(observation(b.np), "pick up cup", 7)
    assert all(ref() is None for ref in refs)
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="load"):
        engine.infer_chunk(observation(b.np), "pick up cup", 7)


def test_missing_task_embedding_fails_before_model_load(native_boundary):
    b = native_boundary
    with pytest.raises(ValueError, match="missing.*embedding"):
        b.engine.load(["unknown task"], b.audit)
    assert b.calls == []


@pytest.mark.parametrize("smoke", ["1", "true", "yes", "y"])
def test_smoke_env_is_rejected_before_model_load(native_boundary, monkeypatch, smoke):
    b = native_boundary
    monkeypatch.setenv("COSMOS_SMOKE", smoke)
    with pytest.raises(RuntimeError, match="COSMOS_SMOKE"):
        b.engine.load(["pick up cup"], b.audit)
    assert b.calls == []


def test_preimported_smoke_and_wrong_platform_are_rejected(native_boundary):
    b = native_boundary
    b.backend.loader.SMOKE = True
    with pytest.raises(RuntimeError, match="SMOKE"):
        b.engine.load(["pick up cup"], b.audit)
    b.backend.loader.SMOKE = False
    b.backend.constants.NUM_ACTIONS_CHUNK = 32
    with pytest.raises(RuntimeError, match="LIBERO"):
        b.engine.load(["pick up cup"], b.audit)


@pytest.mark.parametrize("case", ["bad_image", "bad_proprio", "bad_actions"])
def test_infer_rejects_malformed_inputs_or_outputs(native_boundary, case):
    b = native_boundary
    engine = load_engine(b)
    obs = observation(b.np)
    if case == "bad_image":
        obs["primary_image"] = obs["primary_image"].astype(b.np.float32)
    elif case == "bad_proprio":
        obs["proprio"][0] = b.np.nan
    else:
        b.backend.utils.get_action = lambda *args, **kwargs: {
            "actions": b.np.zeros((15, 7))
        }
    with pytest.raises(ValueError):
        engine.infer_chunk(obs, "pick up cup", 1)


def test_pickle_torch_storage_is_forced_to_cpu(monkeypatch):
    module = engine_api()
    import io

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            load=lambda file, **kwargs: calls.append((file.read(), kwargs))
        ),
    )
    unpickler = module._CPUUnpickler(io.BytesIO())
    unpickler.find_class("torch.storage", "_load_from_bytes")(b"serialized storage")
    assert calls == [
        (b"serialized storage", {"map_location": "cpu", "weights_only": False})
    ]
