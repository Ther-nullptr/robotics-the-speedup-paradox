# Robotics: The Speedup Paradox

[中文](README.md) · [Environment setup](docs/environment_setup.en.md) · [Architecture](docs/architecture.md) · [Contributing](CONTRIBUTING.md)

Infrastructure for embodied-model inference and closed-loop experiments. Bind a model to a compatible task and simulator, compare synchronous execution and paper-style static asynchrony, and collect consistent success-rate and control-step results.

**Development preview.** Three static cases are runnable with compatible external model and simulator source. CPU analysis tools work independently. A license for this repository's own code has not yet been selected; see [third-party and release status](THIRD_PARTY_NOTICES.md).

## Features and validation

| Case or tool | Available functionality | Validation scope |
| --- | --- | --- |
| [π0.5 + LIBERO](benchmarks/static/pi05_libero/README.md) | External LeRobot/VLASH evaluator bridge; original-precision sync and `paper_async` | 500 `libero_object` episodes per mode: 494/500 synchronous, 464/500 with n′=2 |
| [Cosmos + LIBERO](benchmarks/static/cosmos_libero/README.md) | Separate engine, native simulator, single-environment runner; H=16 | Native action alignment; one object task succeeded in 137 sync / 154 async control steps |
| [Cosmos + RoboCasa](benchmarks/static/cosmos_robocasa/README.md) | Three cameras, H=32, initialization matching, single-environment execution | Fixed `TurnOffMicrowave` scene succeeded in 277 sync / 292 async steps |
| [Resource preparation](tools/RESOURCE_PREPARATION.md) | Download or reuse model resources; opt-in trajectory datasets | Local resource reuse, a real small-file download and launcher preflight |
| [Experiment summaries](tools/summarize_experiment.py) | Success rate, failure-budget totals, success-only means and per-task summaries | Shared CLI and automatic end-of-run reports |
| [Trajectory tools](tools/embodied/README.md) | Plot trajectories; compute velocity, acceleration and jerk | CPU tools with synthetic examples |
| [Contracts and speedups](docs/protocols/speedup-metrics.md) | Validate artifacts and compute ratios against an explicit baseline | CPU validation and calculations from declared inputs |

The Cosmos results are individual episodes, not full-suite success rates. Case guides describe the π0.5 evaluation settings. No paper speedup is reported without matching inference-time measurements. There is currently no built-in full-model quantization preset.

LingBot-VA, DynamicVLA + DOM and Kinetix remain planned. Static and dynamic experiments use separate case protocols; model/simulator compatibility is established per case.

## Quick start: CPU tools

Use Python 3.11 or 3.12. These commands require no GPU or model download:

```bash
git clone https://github.com/Ther-nullptr/robotics-the-speedup-paradox.git
cd robotics-the-speedup-paradox
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

python tools/validate_contracts.py --examples
python tools/compare_speedups.py --input examples/speedups/synthetic.json --markdown
python tools/embodied/trajectory_metrics.py --input tools/embodied/examples/smooth.csv
```

Examples are synthetic. Plotting additionally requires `tools/embodied/requirements-plot.txt`. Optionally install the package with `python -m pip install -e .`; this does not install the model or simulator runtime.

## Prepare resources and run a case

Follow the [environment guide](docs/environment_setup.en.md) to provide a separate GPU environment and compatible source for the selected case. The download helper can run in the CPU environment:

```bash
python -m pip install -r tools/requirements-download.txt
python tools/prepare_resources.py --case cosmos_robocasa --dry-run
```

Remove `--dry-run` to download. Use `--source`, `--python` and the case-specific source arguments to generate launcher paths, or fill in the case's `paths.env.example` for existing resources.

| Resource | Default location |
| --- | --- |
| Models and optional trajectory datasets | `~/.cache/robotics/hub/`; override with `--root` or `ROBOTICS_RESOURCE_ROOT` |
| Generated path configuration | `~/.cache/robotics/env/<case>.env` |
| LIBERO assets | `~/.cache/libero/assets/` |
| RoboCasa kitchen assets | `robocasa/models/assets/` within the compatible fork; prepared separately |
| Experiment outputs | Explicit `--output-dir` |

Once RoboCasa resources and runtime paths are configured:

```bash
source ~/.cache/robotics/env/cosmos_robocasa.env
bash benchmarks/static/cosmos_robocasa/run.sh \
  --task TurnOffMicrowave --episodes 1 \
  --schedule paper_async --overlap-actions 2 \
  --output-dir runs/static/cosmos_robocasa/demo-001 --dry-run
```

After preflight, remove `--dry-run` and select a GPU with `--gpu 3`. Add `--record-video` for a complete rollout video. Every real run needs a new output directory. Evaluation stays offline and reports missing resources explicitly.

Full case instructions: [π0.5 + LIBERO](benchmarks/static/pi05_libero/README.md), [Cosmos + LIBERO](benchmarks/static/cosmos_libero/README.md), [Cosmos + RoboCasa](benchmarks/static/cosmos_robocasa/README.md). Detailed case and protocol documents currently use Chinese, with English CLI/report fields.

## Protocol and outputs

`paper_async` selects the observation from primitive control step `t−n′`. Images and proprio use the same snapshot; insufficient history falls back to the current observation. The current runner advances the environment serially. Inference time, simulation time and host wall time have separate meanings; see the [simulation protocol](docs/protocols/simulation.md).

Runs save configuration and source identities, checkpoint audits, episode records and summaries. Cosmos also writes per-request observation offsets and an automatic `run.log`; capture π0.5 console output with redirection or `tee` when needed. Videos are disabled by default and include initial and terminal frames when enabled. RoboCasa videos contain all three camera views.

Automatic reports include success rate, the overall mean with failed episodes charged their declared step budget, and the mean over successful episodes. Recompute reports independently:

```bash
python tools/summarize_experiment.py \
  --input runs/static/cosmos_robocasa/demo-001 --per-task
```

RoboCasa's `--reference-run` verifies initialization against a completed baseline before the first inference. An initialization mismatch is an infrastructure error, not a failed policy episode.

## Repository layout

```text
benchmarks/static/             CLI, configuration and lifecycle for three cases
benchmarks/dynamic/            Planned dynamic cases
src/robotics_bench/engines/     Cosmos loading, preprocessing and action inference
src/robotics_bench/simulators/  Native LIBERO and RoboCasa adapters
src/robotics_bench/protocols/   Static action execution and observation history
tools/                        Resources, validation, summaries and trajectories
schemas/                      Shared contracts; case formats are declared separately
tests/                        CPU behavior and contract checks
docs/                         Setup, architecture and stable protocols
.github/                      Issue/PR templates, CPU CI and contribution settings
```

The [architecture guide](docs/architecture.md) describes actual call relationships, reset behavior and extension boundaries.

## Contributing and licensing

Work on topic branches and submit pull requests. Use bilingual commit titles and PR descriptions covering purpose, changes, validation, impact and rollback. Small changes can use one sentence per language in each section. Maintainers review and merge through PR merge commits; see [CONTRIBUTING](CONTRIBUTING.md).

Track source, necessary configuration, tests, stable documentation and small synthetic examples. Keep models, datasets, videos, logs, scratch notes and generated figures local. Public CPU CI does not run GPU experiments. Server-side branch protection must be verified separately from these repository files.

Background: [The Speedup Paradox](https://arxiv.org/abs/2606.28529). Model and simulator sources retain their own terms, documented in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). The repository-owned license remains undecided; this is a pre-release project.
