#!/usr/bin/env bash
# Forward to the adjacent Python entry without changing the caller's directory.
set -euo pipefail

script_path="${BASH_SOURCE[0]}"
if [[ "$script_path" != /* ]]; then
  script_path="$PWD/$script_path"
fi
script_dir="${script_path%/*}"

exec "${ROBOTICS_PI05_PYTHON:-python}" "$script_dir/run.py" "$@"
