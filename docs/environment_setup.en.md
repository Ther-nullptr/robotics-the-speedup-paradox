# Environment Setup and First Run

[中文](environment_setup.md) · [Project overview](../README.en.md) · [Resource options](../tools/RESOURCE_PREPARATION.md)

CPU tools, downloads and model/simulator execution use separate environments. `requirements-dev.txt` installs CPU development dependencies. `pip install -e .` installs this repository's package only. There is no complete fresh-machine GPU installer yet.

LingBot prioritizes a separate [RoboTwin case](../benchmarks/static/lingbot_robotwin/README.md), using existing local model and simulator environments. It is not a download-helper preset; its case guide documents paths, versions and the Transformers shared-embedding compatibility handling.

## 1. CPU environment

Python 3.11 or 3.12 is recommended; package metadata requires Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/validate_contracts.py --examples
python tools/compare_speedups.py --input examples/speedups/synthetic.json --markdown
python tools/embodied/trajectory_metrics.py --input tools/embodied/examples/smooth.csv
```

The dependencies are NumPy, jsonschema, pytest and Ruff. Plotting additionally requires:

```bash
python -m pip install -r tools/embodied/requirements-plot.txt
```

These commands do not require Torch, CUDA, a simulator or model weights. See [CONTRIBUTING](../CONTRIBUTING.md) for development checks.

## 2. Separate GPU environments

The following combinations were used in local experiments. They are compatibility evidence, not complete dependency lock files.

| Dependency | π0.5 + LIBERO | Cosmos + LIBERO | Cosmos + RoboCasa |
| --- | --- | --- | --- |
| Python | 3.10 | 3.10 | 3.10 |
| PyTorch | 2.7.1+cu126 | 2.7.0+cu128 | 2.7.0+cu128 |
| NumPy | 1.26.4 | 2.2.6 | 2.2.6 |
| MuJoCo | 3.8.0 | 3.2.6 | 3.2.6 |
| robosuite | 1.4.0 | 1.4.0 | 1.5.1 |
| Case-specific dependencies | LeRobot 0.4.1, Transformers 4.53.3, compatible evaluator changes | LIBERO 0.1.1, Transformer Engine, NATTEN | Compatible RoboCasa fork, Transformer Engine, NATTEN |

The local Cosmos stack uses Transformer Engine `2.2+cu128.torch27` and NATTEN `0.21.0+cu128.torch27`. Compiled extensions must match Torch/CUDA. Current model experiments used an RTX 6000 Ada.

Core model execution code now lives in `src/robotics_bench/models/`, selected by the default `--model-runtime owned`. Common frameworks, loaders and simulators remain external dependencies. Optional fusion requires Triton in the model environment. Integer backends additionally need an explicitly provided compatible CUTLASS checkout and CUDA development toolkit; the Ada build used CUDA 12.8. See the [operator build guide](../src/robotics_kernels/README.md) for paths and architecture selection. FP4/FP8 validation on Blackwell remains pending. Per-round figures use the external profile-visualizer skill with CairoSVG/Cairo in a separate rendering environment; see the [inference benchmark](../benchmarks/inference/README.md).

- **π0.5:** supply a VLASH sim evaluator compatible with this repository's arguments and observation-history protocol. Use a native LeRobot π0.5 LIBERO checkpoint, not VLASH-finetuned weights. The tested external source contains local changes; a package version alone does not establish compatibility. Preflight checks evaluator fields, and manifests record source identity.
- **Cosmos:** follow the [upstream setup](https://github.com/NVlabs/cosmos-policy/blob/main/SETUP.md) for model dependencies and configure LIBERO and RoboCasa separately. The local environments reused existing Cosmos packages; a complete installation from an empty machine has not been validated.
- **RoboCasa:** use the [compatible fork](https://github.com/moojink/robocasa-cosmos-policy); tested commit: `edd9a328b3ec98050f42d194c1419307a79c4d87`. Use the controller configuration supplied by Cosmos. See the [official RoboCasa instructions](https://github.com/NVlabs/cosmos-policy/blob/main/ROBOCASA.md).

Keep the two robosuite versions in separate environments. A case requires a working NVIDIA driver and headless EGL/OpenGL setup. Video output also needs imageio and an FFmpeg backend. Check the selected runtime:

```bash
nvidia-smi
/path/to/case-env/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Select an available GPU explicitly with `--gpu INDEX` or a full GPU UUID. Launchers set device and EGL options before loading runtime dependencies.

## 3. Models, tasks and simulator assets

The download client can be installed in the CPU environment:

```bash
python -m pip install -r tools/requirements-download.txt
python tools/prepare_resources.py --case cosmos_libero --dry-run
```

Supported cases are `pi05_libero`, `cosmos_libero` and `cosmos_robocasa`. Dry-run is offline and makes no filesystem changes. Remove that flag to download.

| Resource | Location and preparation |
| --- | --- |
| Checkpoints, VAE, tokenizer, statistics and T5 | Download helper; default `~/.cache/robotics/hub/` |
| Training/calibration trajectories | Downloaded only with `--with-dataset` |
| LIBERO task definitions and initial states | Supplied by the compatible package; pass its directory as `--libero-root` |
| LIBERO meshes and textures | Download helper; native location `~/.cache/libero/assets/` |
| RoboCasa kitchen assets | Compatible fork's downloader; its `robocasa/models/assets/` directory |
| Generated environment files | `~/.cache/robotics/env/<case>.env`; explicitly source the file |

Use `--root /data/robotics` or `ROBOTICS_RESOURCE_ROOT` to relocate model/dataset caches. Simulator asset paths follow their native packages. `HF_HOME` does not override the helper's explicit cache directory.

Closed-loop evaluation uses task initialization and the simulator, so full training trajectories are optional. Cosmos normalization statistics and precomputed T5 embeddings are required inference resources. Obtain any gated-resource access through the upstream provider; credentials are not written to generated env files.

Reuse existing directories with options such as `--reuse policy=DIR` and `--reuse vae=DIR`. Reusing every resource permits offline preparation. Changed preparation plans require a new `--env-file`; existing different configurations are not overwritten. See [resource preparation](../tools/RESOURCE_PREPARATION.md) for all options.

## 4. Bind runtime paths

Copy a case template, fill in existing paths and source it:

```bash
mkdir -p .local
cp -n benchmarks/static/cosmos_robocasa/paths.env.example .local/cosmos-robocasa.env
# Replace PATH/TO values with your runtime and resource paths.
source .local/cosmos-robocasa.env
```

Equivalent templates exist under `benchmarks/static/pi05_libero/` and `benchmarks/static/cosmos_libero/`. Keep the virtual environment's `bin/python` path; resolving that symlink to its base interpreter can select the wrong environment.

Alternatively, generate the paths during resource preparation:

```bash
python tools/prepare_resources.py --case cosmos_robocasa \
  --source /path/to/cosmos-policy \
  --robocasa-source /path/to/robocasa-cosmos-policy \
  --python /path/to/robocasa-env/bin/python
source ~/.cache/robotics/env/cosmos_robocasa.env
```

For Cosmos + LIBERO, select `--case cosmos_libero` and replace `--robocasa-source` with `--libero-root /path/to/site-packages/libero/libero`. For π0.5, select `--case pi05_libero`, point `--source` at the compatible VLASH sim checkout, and provide its LIBERO package directory. RoboCasa kitchen assets are still prepared separately.

CLI paths take precedence over environment variables. Launchers never source arbitrary env files automatically. The download client and GPU case may use different Python executables.

## 5. Preflight and launch

Source the relevant case env before each command. From the repository root:

```bash
# Cosmos + RoboCasa
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --layout-id 1 --style-id 1 --env-seed 0 \
  --episodes 1 --schedule sync \
  --output-dir runs/static/cosmos_robocasa/sync-001 --dry-run

# Cosmos + LIBERO
bash benchmarks/static/cosmos_libero/run.sh \
  --suite libero_object --task-ids 0 --episodes 1 --schedule sync \
  --output-dir runs/static/cosmos_libero/sync-001 --dry-run

# pi0.5 + LIBERO
bash benchmarks/static/pi05_libero/run.sh \
  --suite libero_object --episodes 1 --batch-size 1 --schedule sync --quant none \
  --output-dir runs/static/pi05_libero/sync-001 --dry-run
```

Dry-run imports no model/simulator code, loads no checkpoint tensors and creates no output directory. File hashing can still take time. After it passes, remove `--dry-run` and add `--gpu 3`. Add `--record-video` if needed. Paper asynchrony uses `--schedule paper_async --overlap-actions 2`.

`--episodes` is the total run budget. RoboCasa currently selects one task and one layout/style pair per command. Full π0.5 `libero_object` evaluation uses `--episodes 500 --batch-size 10`. RoboCasa can verify matching initialization with `--reference-run`.

## 6. Logs and results

Every real run requires a new `--output-dir`. Cosmos automatically writes `run.log`. To capture π0.5 console output:

```bash
mkdir -p logs
set -o pipefail
bash benchmarks/static/pi05_libero/run.sh \
  --schedule sync --quant none --gpu 3 \
  --output-dir runs/static/pi05_libero/trial-001 \
  2>&1 | tee logs/pi05-trial-001.log
```

Completed runs write `episode-summary.md` and `.json`. Videos are disabled by default; enabled recordings are stored under `videos/<task>/`. See the [project overview](../README.en.md#protocol-and-outputs) and case guides for the failure-budget metric and additional artifacts.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing weights, tokenizer or T5 | Prepare resources before evaluation, or supply explicit reuse paths |
| `SingleArmEnv` import failure | LIBERO may be using robosuite 1.5.x; select its separate 1.4.0 environment |
| RoboCasa assets missing or sampling errors | Verify fixtures, textures and Objaverse extraction in the compatible fork |
| Old container source paths | Pass a valid `--robocasa-source`; inspect the runtime and editable installation |
| Initialization differs from reference | Match task, layout/style, environment seed and asset version; inspect `initializations/` |
| Rendering or recording fails | Check EGL, MuJoCo, imageio, FFmpeg and device selection in the case environment |
| Conda XML/Matplotlib reports `XML_SetReparseDeferralEnabled` | Check that Python and libexpat match; a local verification workaround is not a project-wide launcher requirement |

Dependency checks, resource preparation and a successful GPU rollout establish different levels of readiness. Cross-machine installation still requires verifying upstream dependencies.

## KINETIX dynamic case

KINETIX uses an isolated Python 3.11/JAX runtime. The flow model, environment, Jax2D physics, twelve levels and small render textures are maintained in this repository; only checkpoint parameters and framework dependencies are external. Configure `ROBOTICS_KINETIX_PYTHON` and `ROBOTICS_KINETIX_POLICY_DIR`, then follow the [case guide](../benchmarks/dynamic/kinetix/README.md). The runtime requirement file records the tested library versions; a fresh-machine GPU installation has not yet been independently validated. No automatic model/data download occurs during evaluation.
