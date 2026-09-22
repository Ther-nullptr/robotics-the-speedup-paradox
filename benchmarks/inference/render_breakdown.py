#!/usr/bin/env python3
"""Render a selected same-cohort GPU breakdown as SVG and PNG, without a GPU."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--render-python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= len(args.profiles) <= 4 or len(set(args.profiles)) != len(
        args.profiles
    ):
        parser.error("Select one to four unique profile names from one cohort")
    source = args.input.expanduser().resolve()
    data = json.loads((source / "profile-breakdown.json").read_text())
    available = {row["name"]: row for row in data["profiles"]}
    if set(args.profiles) - set(available):
        parser.error("Requested profiles are absent from this measurement cohort")
    data["profiles"] = [available[name] for name in args.profiles]
    data["notes"].append(
        "CPU-only visualization of existing evidence. Complete policy wall medians are stored separately in measurements.json."
    )
    output = (args.output_dir or source / "figures").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data["sources"] = [
        os.path.relpath(source / name, output) for name in data["sources"]
    ]
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    selected = output / f"{stamp}_selected_breakdown.json"
    selected.write_text(json.dumps(data, indent=2) + "\n")
    result = subprocess.run(
        [
            args.render_python,
            str(args.skill.resolve() / "scripts/render_profile_breakdown.py"),
            str(selected),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    artifacts = json.loads(result.stdout)
    svg = Path(artifacts["svg"])
    png = svg.with_suffix(".png")
    # CairoSVG does not parse CSS Color 4 space-separated HSL. Convert colors
    # in the rasterization input only; retain the renderer's original SVG.
    conversion = """
import colorsys, re, sys
from pathlib import Path
import cairosvg
def rgb(match):
    h, s, l = map(float, match.groups())
    return '#' + ''.join(f'{round(v * 255):02x}' for v in colorsys.hls_to_rgb(h / 360, l / 100, s / 100))
source = Path(sys.argv[1]).read_text()
source = re.sub(r'hsl\\(([\\d.]+) ([\\d.]+)% ([\\d.]+)%\\)', rgb, source)
cairosvg.svg2png(bytestring=source.encode(), write_to=sys.argv[2])
"""
    subprocess.run(
        [args.render_python, "-c", conversion, str(svg), str(png)], check=True
    )
    artifacts.update(png=str(png), input=str(selected), source_cohort=str(source))
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
