#!/usr/bin/env bash
# Use any Python >=3.10 for orchestration; each case selects its own environment.
set -euo pipefail
script_path="${BASH_SOURCE[0]}"
if [[ "$script_path" != /* ]]; then
  script_path="$PWD/$script_path"
fi
exec "${ROBOTICS_DRIVER_PYTHON:-python}" "${script_path%/*}/full_evaluation.py" "$@"
