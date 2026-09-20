#!/usr/bin/env python3
"""Prepare explicit local resources for a static LIBERO case before evaluation."""

import argparse
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile


LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ASSET_DIRECTORIES = (
    "articulated_objects",
    "scenes",
    "stable_hope_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--case", choices=("cosmos_libero", "pi05_libero"), required=True
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("ROBOTICS_RESOURCE_ROOT", "~/.cache/robotics")),
        help="Model/dataset cache and generated configs; default ~/.cache/robotics",
    )
    parser.add_argument(
        "--with-dataset",
        action="store_true",
        help="Also download the case's training/calibration trajectories",
    )
    parser.add_argument(
        "--reuse",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Reuse a local resource directory without network access",
    )
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="NAME=REVISION",
        help="Override a resource's Hub revision (repeatable)",
    )
    parser.add_argument(
        "--source", type=Path, help="Compatible external case source checkout"
    )
    parser.add_argument("--python", type=Path, help="Existing case Python executable")
    parser.add_argument(
        "--libero-root",
        type=Path,
        help="Installed libero/libero directory containing bddl_files and init_files",
    )
    parser.add_argument("--env-file", type=Path, help="Default: ROOT/env/CASE.env")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the offline plan; do not download or write anything",
    )
    return parser


def _overrides(values, allowed):
    result = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or not setting or name not in allowed or name in result:
            raise ValueError(
                f"Expected one NAME=VALUE per resource; available: {sorted(allowed)}"
            )
        result[name] = setting
    return result


def _resource(name, repo, required, *, patterns=None, revision="main", kind="model"):
    return {
        "name": name,
        "repo_id": repo,
        "repo_type": kind,
        "revision": revision,
        "patterns": patterns or required,
        "required": required,
    }


def build_plan(args):
    root = args.root.expanduser().resolve()
    assets = Path.home() / ".cache/libero/assets"
    if args.case == "cosmos_libero":
        resources = [
            _resource(
                "policy",
                "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
                [
                    "Cosmos-Policy-LIBERO-Predict2-2B.pt",
                    "libero_dataset_statistics.json",
                    "libero_t5_embeddings.pkl",
                ],
                revision="cb689ec0e3347c13667d70a78a3447388f5c3bb8",
            ),
            _resource(
                "vae",
                "nvidia/Cosmos-Predict2-2B-Video2World",
                ["tokenizer/tokenizer.pth"],
                revision="f50c09f5d8ab133a90cac3f4886a6471e9ba3f18",
            ),
        ]
        dataset = _resource(
            "dataset",
            "nvidia/LIBERO-Cosmos-Policy",
            ["all_episodes/*.hdf5", "success_only/*.hdf5"],
            patterns=["*"],
            kind="dataset",
        )
    else:
        resources = [
            _resource(
                "policy",
                "lerobot/pi05_libero_finetuned_v044",
                [
                    "config.json",
                    "model.safetensors",
                    "policy_preprocessor.json",
                    "policy_postprocessor.json",
                ],
                patterns=["*.json", "*.safetensors"],
                revision="dbf8a3f794a9c4297b44f40b752712f50073d945",
            ),
            _resource(
                "tokenizer",
                "google/paligemma-3b-pt-224",
                ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
                patterns=[
                    "tokenizer*",
                    "special_tokens_map.json",
                    "config.json",
                    "preprocessor_config.json",
                ],
            ),
        ]
        dataset = _resource(
            "dataset",
            "lerobot/libero",
            ["meta/info.json", "data/*.parquet"],
            patterns=["*"],
            kind="dataset",
            revision="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
        )
    resources.append(
        _resource(
            "libero-assets",
            "jadechoghari/libero-assets",
            [f"{name}/*" for name in ASSET_DIRECTORIES],
            patterns=["*"],
        )
    )
    if args.with_dataset:
        resources.append(dataset)
    names = {item["name"] for item in resources}
    reuse = _overrides(args.reuse, names)
    revisions = _overrides(args.revision, names)
    if set(reuse) & set(revisions):
        raise ValueError("A reused resource cannot also request a Hub revision")
    for item in resources:
        name = item["name"]
        if name in revisions:
            item["revision"] = revisions[name]
        if name in reuse:
            item["reuse"] = str(Path(reuse[name]).expanduser().resolve())
        if name == "libero-assets":
            item["local_dir"] = str(assets)
            if name in reuse and Path(item["reuse"]).resolve() != assets.resolve():
                raise ValueError(
                    "This LIBERO wheel resolves assets at ~/.cache/libero/assets; "
                    "reuse that directory or use the existing manual case configuration"
                )
    return {
        "case": args.case,
        "root": str(root),
        "resources": resources,
        "env_file": str(
            (args.env_file or root / "env" / f"{args.case}.env").expanduser().absolute()
        ),
        "config_dir": str(root / "configs" / args.case),
        **{
            key: str(getattr(args, key).expanduser().resolve())
            if getattr(args, key)
            else None
            for key in ("source", "libero_root")
        },
        # Resolving a venv's Python symlink would silently select its base interpreter.
        "python": str(args.python.expanduser().absolute()) if args.python else None,
    }


def _required_names(resource, names):
    for pattern in resource["required"]:
        if not any(fnmatch(name, pattern) for name in names):
            raise ValueError(
                f"Missing required {resource['name']} file pattern: {pattern}"
            )


def _validate_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Required file is missing or empty: {path}")
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Git LFS pointer is not a downloaded resource: {path}")
    if path.suffix == ".json":
        json.loads(path.read_text())


def validate_local(resource, directory):
    directory = Path(directory)
    names = [
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(directory).parts
    ]
    _required_names(resource, names)
    selected = [n for n in names if any(fnmatch(n, p) for p in resource["patterns"])]
    for name in selected:
        _validate_file(directory / name)
    if resource["repo_id"] == "lerobot/pi05_libero_finetuned_v044":
        for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
            for step in json.loads((directory / filename).read_text()).get("steps", []):
                state = step.get("state_file")
                if state:
                    relative = Path(state)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(
                            "Processor state_file must stay inside checkpoint"
                        )
                    _validate_file(directory / relative)
    return selected


def download_resource(resource, root, hub):
    info = hub.HfApi().repo_info(
        repo_id=resource["repo_id"],
        repo_type=resource["repo_type"],
        revision=resource["revision"],
        files_metadata=True,
    )
    files = [
        f
        for f in info.siblings
        if any(fnmatch(f.rfilename, p) for p in resource["patterns"])
    ]
    _required_names(resource, [f.rfilename for f in files])
    for file in files:
        relative = Path(file.rfilename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Hub file path must stay inside the resource directory")
    options = {"cache_dir": str(Path(root) / "hub")}
    if "local_dir" in resource:
        options["local_dir"] = resource["local_dir"]
    directory = Path(
        hub.snapshot_download(
            repo_id=resource["repo_id"],
            repo_type=resource["repo_type"],
            revision=info.sha,
            allow_patterns=[f.rfilename for f in files],
            **options,
        )
    )
    for file in files:
        path = directory / file.rfilename
        if not path.is_file() or (
            file.size is not None and path.stat().st_size != file.size
        ):
            raise ValueError(
                f"Downloaded file size does not match Hub metadata: {path}"
            )
        _validate_file(path)
    validate_local(resource, directory)
    return {
        "name": resource["name"],
        "path": str(directory.resolve()),
        "origin": "huggingface",
        "repo_id": resource["repo_id"],
        "repo_type": resource["repo_type"],
        "requested_revision": resource["revision"],
        "resolved_revision": info.sha,
        "file_count": len(files),
        "bytes": sum((directory / f.rfilename).stat().st_size for f in files),
    }


def libero_paths(root, dataset, assets):
    root = Path(root)
    for suite in LIBERO_SUITES:
        bddl = list((root / "bddl_files" / suite).glob("*.bddl"))
        if len(bddl) != 10:
            raise ValueError(
                f"Expected 10 task BDDL files in {root / 'bddl_files' / suite}"
            )
        for task in bddl:
            state = root / "init_files" / suite / f"{task.stem}.pruned_init"
            if not state.is_file() or state.stat().st_size == 0:
                raise ValueError(f"Missing task initial state: {state}")
    return {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(dataset),
        "assets": str(assets),
    }


def _write_once(path, content):
    """Publish a complete file without replacing a different user's config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == content:
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def write_env(path, values):
    content = "# Generated resource paths. Source this file explicitly before running the case.\n"
    content += "".join(
        f"export {key}={shlex.quote(str(value))}\n"
        for key, value in sorted(values.items())
    )
    _write_once(path, content)


def prepare(plan, *, hub=None):
    root = Path(plan["root"])
    # Check caller-supplied runtime locations before starting large downloads.
    source = Path(plan["source"]) if plan["source"] else None
    if source:
        entry = (
            "cosmos_policy/config/config.py"
            if plan["case"] == "cosmos_libero"
            else "vlash/eval_libero.py"
        )
        if not (source / entry).is_file():
            raise ValueError(
                f"Compatible case source entry is missing: {source / entry}"
            )
    if plan["python"] and not os.access(plan["python"], os.X_OK):
        raise ValueError(
            "The case Python executable does not exist or is not executable"
        )
    config = None
    if plan["libero_root"]:
        config = libero_paths(
            Path(plan["libero_root"]),
            root / "datasets",
            Path.home() / ".cache/libero/assets",
        )
    results = []
    for resource in plan["resources"]:
        if "reuse" in resource:
            files = validate_local(resource, resource["reuse"])
            result = {
                "name": resource["name"],
                "path": resource["reuse"],
                "origin": "local",
                "resolved_revision": None,
                "file_count": len(files),
            }
        else:
            if hub is None:
                if os.environ.get("HF_HUB_OFFLINE", "").upper() in (
                    "1",
                    "ON",
                    "YES",
                    "TRUE",
                ):
                    raise ValueError(
                        "Online preparation requires HF_HUB_OFFLINE to be unset; use --reuse for offline resources"
                    )
                try:
                    import huggingface_hub as hub
                except ImportError as exc:
                    raise RuntimeError(
                        "Install download dependencies: python -m pip install -r tools/requirements-download.txt"
                    ) from exc
            print(
                f"Preparing {resource['name']}: {resource['repo_id']}", file=sys.stderr
            )
            result = download_resource(resource, root, hub)
        results.append(result)
    paths = {item["name"]: Path(item["path"]) for item in results}
    cosmos = plan["case"] == "cosmos_libero"
    prefix = "ROBOTICS_COSMOS_" if cosmos else "ROBOTICS_"
    values = {
        prefix + "CHECKPOINT": paths["policy"] / "Cosmos-Policy-LIBERO-Predict2-2B.pt"
        if cosmos
        else paths["policy"]
    }
    if cosmos:
        values.update(
            {
                prefix + "DATASET_STATS": paths["policy"]
                / "libero_dataset_statistics.json",
                prefix + "TEXT_EMBEDDINGS": paths["policy"]
                / "libero_t5_embeddings.pkl",
                prefix + "VAE_CHECKPOINT": paths["vae"] / "tokenizer/tokenizer.pth",
            }
        )
    else:
        values["ROBOTICS_TOKENIZER"] = paths["tokenizer"]
    if source:
        values["ROBOTICS_COSMOS_SOURCE" if cosmos else "ROBOTICS_SIM_SOURCE"] = source
    if plan["python"]:
        values["ROBOTICS_COSMOS_PYTHON" if cosmos else "ROBOTICS_PI05_PYTHON"] = plan[
            "python"
        ]
    if "dataset" in paths:
        values[prefix + "DATASET"] = paths["dataset"]
    if config:
        _write_once(
            Path(plan["config_dir"]) / "config.yaml",
            json.dumps(config, indent=2) + "\n",
        )
        values[prefix + "LIBERO_CONFIG_DIR"] = plan["config_dir"]
    report = {
        "status": "resources_prepared",
        "case": plan["case"],
        "resources": results,
        "env_file": plan["env_file"],
        "runtime_installed": False,
        "libero_config_generated": config is not None,
    }
    _write_once(
        Path(plan["env_file"] + ".resources.json"), json.dumps(report, indent=2) + "\n"
    )
    write_env(Path(plan["env_file"]), values)
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        else:
            report = prepare(plan)
            print(json.dumps(report, indent=2))
            print(
                f"Source the generated paths: source {shlex.quote(plan['env_file'])}",
                file=sys.stderr,
            )
            if not plan["libero_root"]:
                print(
                    "No LIBERO config generated; supply --libero-root or retain your existing case config.",
                    file=sys.stderr,
                )
    except Exception as exc:
        # Do not echo HTTP exception payloads, request headers, or credentials.
        if isinstance(exc, (ValueError, FileExistsError, RuntimeError, ImportError)):
            message = str(exc)
        else:
            message = f"{type(exc).__name__}: resource preparation failed; check Hub access, network, and disk space"
        print(f"Error: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
