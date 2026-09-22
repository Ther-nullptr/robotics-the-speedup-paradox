"""Offline single-client LingBot model worker, launched by the RoboTwin case."""

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import struct
import sys
import traceback


def write_json(path, data):
    def encode(value):
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"Unsupported audit value: {type(value).__name__}")

    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False, default=encode) + "\n")
    temp.replace(path)


def checkpoint_headers(directory):
    result = {}
    for path in sorted(directory.glob("*.safetensors")):
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 20_000_000:
                raise ValueError("Invalid safetensors header")
            header = json.loads(stream.read(length))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in result:
                raise ValueError(f"Duplicate checkpoint key: {key}")
            result[key] = value["shape"]
    return result


def restore_text_embedding_alias(model, keys):
    """Preserve the checkpoint's UMT5 shared input embedding across HF versions.

    Older UMT5EncoderModel constructs UMT5Stack(config, self.shared). Newer HF
    versions may create a separate encoder embedding when tie_word_embeddings
    is false, although this encoder checkpoint stores only shared.weight.
    """
    if "shared.weight" in keys and "encoder.embed_tokens.weight" not in keys:
        changed = model.encoder.embed_tokens is not model.shared
        model.set_input_embeddings(model.shared)
        if model.encoder.embed_tokens is not model.shared:
            raise RuntimeError("Failed to restore UMT5 input embedding alias")
        return changed
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cpu-offload", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    source, checkpoint, output = (
        p.expanduser().resolve()
        for p in (args.source, args.checkpoint, args.output_dir)
    )
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "wan_va"))
    audit_path = output / "checkpoint-load.json"
    audit = {
        "status": "loading",
        "checkpoint": str(checkpoint),
        "source": str(source),
        "attention_backend": "torch_sdpa",
        "cpu_offload": args.cpu_offload,
        "model_seed": args.seed,
        "debug_tensor_saving": False,
    }
    write_json(audit_path, audit)
    try:
        import numpy as np
        import torch

        native = importlib.import_module("wan_va.wan_va_server")
        if not Path(native.__file__).resolve().is_relative_to(source / "wan_va"):
            raise RuntimeError("LingBot was imported from a different source")
        config = deepcopy(native.VA_CONFIGS["robotwin"])
        config.wan22_pretrained_model_name_or_path = str(checkpoint)
        config.enable_offload = args.cpu_offload
        config.infer_mode = "server"
        config.save_root = str(output / "model")
        config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        config.rank = int(os.environ.get("RANK", 0))
        config.world_size = int(os.environ.get("WORLD_SIZE", 1))
        if config.world_size != 1:
            raise ValueError("This case uses one model worker")
        if (
            config.frame_chunk_size,
            config.action_per_frame,
            len(config.used_action_channel_ids),
        ) != (2, 16, 16):
            raise ValueError(
                "LingBot source does not match the RoboTwin checkpoint protocol"
            )
        native.init_distributed(config.world_size, config.local_rank, config.rank)
        # Debug writes are not part of policy execution or the recorded baseline.
        native.save_async = lambda *args, **kwargs: None
        model_class = native.load_transformer.__globals__["WanTransformer3DModel"]
        encoder_class = native.load_text_encoder.__globals__["UMT5EncoderModel"]

        def audited_encoder(path, torch_dtype, torch_device):
            encoder, loading = encoder_class.from_pretrained(
                path,
                torch_dtype=torch_dtype,
                output_loading_info=True,
                local_files_only=True,
            )
            shapes = checkpoint_headers(Path(path))
            restored = restore_text_embedding_alias(encoder, shapes)
            missing = [
                key for key, _ in encoder.named_parameters() if key not in shapes
            ]
            mismatch = [
                key
                for key, value in encoder.named_parameters()
                if key in shapes and list(value.shape) != shapes[key]
            ]
            if missing or mismatch:
                raise RuntimeError(
                    f"Incomplete text encoder load: missing={missing}, shape={mismatch}"
                )
            audit["text_encoder"] = {
                "learned_parameters_covered": True,
                "shared_embedding_alias_restored": restored,
                "loading_info": loading,
            }
            return encoder.to(torch_device)

        native.load_text_encoder = audited_encoder

        def audited_transformer(path, torch_dtype, torch_device, **kwargs):
            model, loading = model_class.from_pretrained(
                path,
                torch_dtype=torch_dtype,
                output_loading_info=True,
                local_files_only=True,
                **kwargs,
            )
            shapes = checkpoint_headers(Path(path))
            missing, mismatched = [], []
            for key, parameter in model.named_parameters():
                if key not in shapes:
                    missing.append(key)
                elif list(parameter.shape) != shapes[key]:
                    mismatched.append(key)
            if missing or mismatched or loading.get("error_msgs"):
                raise RuntimeError(
                    f"Incomplete LingBot transformer load: missing={missing}, shape={mismatched}, errors={loading.get('error_msgs')}"
                )
            audit.update(
                transformer_parameter_count=sum(p.numel() for p in model.parameters()),
                checkpoint_tensor_count=len(shapes),
                loading_info=loading,
                learned_parameters_covered=True,
            )
            return model.to(torch_device)

        native.load_transformer = audited_transformer
        model = native.VA_Server(config)
        original_infer = model.infer

        def seeded_infer(observation):
            if observation.get("reset"):
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
            return original_infer(observation)

        model.infer = seeded_infer
        audit.update(
            status="passed",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(0),
            dtype=str(config.param_dtype),
            frame_chunk_size=2,
            action_per_frame=16,
            video_denoising_steps=config.num_inference_steps,
            action_denoising_steps=config.action_num_inference_steps,
            attention_window=config.attn_window,
            source_sha256=hashlib.sha256(
                Path(native.__file__).read_bytes()
            ).hexdigest(),
        )
        write_json(audit_path, audit)
        write_json(
            output / "server-ready.json",
            {"port": args.port, "pid": os.getpid(), "status": "ready"},
        )
        native.run_async_server_mode(model, config.local_rank, "127.0.0.1", args.port)
    except BaseException as exc:
        audit.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        write_json(audit_path, audit)
        raise


if __name__ == "__main__":
    main()
