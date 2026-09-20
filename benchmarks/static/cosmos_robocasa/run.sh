#!/usr/bin/env bash
set -euo pipefail
script_path="${BASH_SOURCE[0]}"
if [[ "$script_path" != /* ]]; then
  script_path="$PWD/$script_path"
fi
exec "${ROBOTICS_COSMOS_ROBOCASA_PYTHON:-python}" "${script_path%/*}/run.py" "$@"
