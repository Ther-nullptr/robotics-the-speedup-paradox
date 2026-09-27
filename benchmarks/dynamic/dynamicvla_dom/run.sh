#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROBOTICS_DRIVER_PYTHON:-python}" "$script_dir/run.py" "$@"
