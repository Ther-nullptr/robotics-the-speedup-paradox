"""Inspect experimental quantization recipes without loading model frameworks."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from robotics_bench.kinetix.quantization_recipes import (  # noqa: E402
    FAMILIES,
    describe_recipe,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--family", choices=FAMILIES, default="ptmix")
    parser.add_argument("--layers", help="Comma-separated channel Linear counts")
    parser.add_argument("--flow-steps", type=int, default=5)
    args = parser.parse_args()
    if args.layers is None:
        layers = (
            [0]
            if args.family == "fp16"
            else list(range(0, 9, 2 if args.family == "a4mix" else 1))
        )
    else:
        try:
            layers = [int(value) for value in args.layers.split(",")]
        except ValueError:
            parser.error("layers must contain comma-separated integers")
    if len(layers) != len(set(layers)):
        parser.error("layers must be distinct")
    try:
        recipes = [
            describe_recipe(args.family, count, flow_steps=args.flow_steps)
            for count in layers
        ]
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "format": "kinetix-quantization-recipes-v1",
                "status": "recipe_only",
                "recipes": recipes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
