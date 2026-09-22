"""CPU reference contract for LingBot source-aware KV read-window pruning."""

import argparse
import json
from pathlib import Path


INPUT_FORMAT = "lingbot-kv-read-window-input-v1"
OUTPUT_FORMAT = "lingbot-kv-read-window-selection-v1"
SOURCES = {"latent", "action", "unknown"}
NATIVE_BASELINE = {
    "attention_window": 72,
    "latent_tokens": 8640,
    "action_tokens": 1152,
    "latent_tokens_per_chunk": 240,
    "action_tokens_per_chunk": 32,
}
PAPER_PRESETS = {
    "base": {
        "enabled": False,
        "max_latent_tokens": 8640,
        "max_action_tokens": 1152,
    },
    "p1": {
        "enabled": True,
        "max_latent_tokens": 1024,
        "max_action_tokens": 256,
    },
    "p2": {
        "enabled": True,
        "max_latent_tokens": 512,
        "max_action_tokens": 128,
    },
    "p3": {
        "enabled": True,
        "max_latent_tokens": 256,
        "max_action_tokens": 64,
    },
    "p4": {
        "enabled": True,
        "max_latent_tokens": 128,
        "max_action_tokens": 32,
    },
}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _records(slots):
    if not isinstance(slots, list):
        raise ValueError("slots must be a list")
    result, seen = [], set()
    for index, item in enumerate(slots):
        if not isinstance(item, dict):
            raise ValueError(f"slots[{index}] must be an object")
        slot = _integer(item.get("slot"), f"slots[{index}].slot")
        if slot in seen:
            raise ValueError(f"duplicate cache slot: {slot}")
        seen.add(slot)
        source = item.get("source")
        if source not in SOURCES:
            raise ValueError(f"slots[{index}].source must be latent, action or unknown")
        generation = _integer(item.get("generation"), f"slots[{index}].generation", -1)
        if source != "unknown" and generation < 0:
            raise ValueError(f"slots[{index}] known source requires generation >= 0")
        current = item.get("current", False)
        if type(current) is not bool:
            raise ValueError(f"slots[{index}].current must be a boolean")
        result.append(
            {
                "slot": slot,
                "source": source,
                "generation": generation,
                "current": current,
            }
        )
    return result


def paper_preset(name):
    """Return one paper operating point plus auditable derived metadata."""
    if name not in PAPER_PRESETS:
        raise ValueError(f"preset must be one of: {', '.join(PAPER_PRESETS)}")
    config = dict(PAPER_PRESETS[name])
    readable = config["max_latent_tokens"] + config["max_action_tokens"]
    native = NATIVE_BASELINE["latent_tokens"] + NATIVE_BASELINE["action_tokens"]
    return {
        "preset": name,
        **config,
        "readable_kv_ratio_vs_native": readable / native,
    }


def resolve_config(request):
    preset = request.get("preset")
    explicit = {"enabled", "max_latent_tokens", "max_action_tokens"}
    if preset is not None:
        if explicit & request.keys():
            raise ValueError("preset cannot be combined with explicit window fields")
        if not isinstance(preset, str):
            raise ValueError("preset must be a string")
        return paper_preset(preset)
    return {
        "preset": None,
        "enabled": request.get("enabled"),
        "max_latent_tokens": request.get("max_latent_tokens"),
        "max_action_tokens": request.get("max_action_tokens"),
        "readable_kv_ratio_vs_native": None,
    }


def select_read_window(slots, *, enabled, max_latent_tokens, max_action_tokens):
    """Select historical cache slots without mutating or evicting the cache.

    A zero budget means unbounded for that source. Unknown-source and current
    slots are always preserved, even if that makes the selected count exceed a
    source budget. Recency is ordered by generation, then lower slot index.
    """
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    max_latent_tokens = _integer(max_latent_tokens, "max_latent_tokens")
    max_action_tokens = _integer(max_action_tokens, "max_action_tokens")
    records = _records(slots)

    if not enabled or (max_latent_tokens == 0 and max_action_tokens == 0):
        selected = {item["slot"] for item in records}
    else:
        selected = {
            item["slot"]
            for item in records
            if item["source"] == "unknown" or item["current"]
        }
        for source, budget in (
            ("latent", max_latent_tokens),
            ("action", max_action_tokens),
        ):
            candidates = [item for item in records if item["source"] == source]
            candidates.sort(key=lambda item: (-item["generation"], item["slot"]))
            kept = candidates if budget == 0 else candidates[:budget]
            selected.update(item["slot"] for item in kept)

    selected_slots = sorted(selected)
    dropped_slots = sorted(
        item["slot"] for item in records if item["slot"] not in selected
    )
    selected_records = [item for item in records if item["slot"] in selected]
    return {
        "selected_slots": selected_slots,
        "dropped_slots": dropped_slots,
        "counts": {
            "input": len(records),
            "selected": len(selected_slots),
            "dropped": len(dropped_slots),
            "latent": sum(item["source"] == "latent" for item in selected_records),
            "action": sum(item["source"] == "action" for item in selected_records),
            "unknown": sum(item["source"] == "unknown" for item in selected_records),
            "current": sum(item["current"] for item in selected_records),
        },
    }


def load_request(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != INPUT_FORMAT:
        raise ValueError(f"input format must be {INPUT_FORMAT}")
    if type(data.get("synthetic")) is not bool:
        raise ValueError("synthetic must be a boolean")
    return data


def build_result(request):
    config = resolve_config(request)
    selection = select_read_window(
        request.get("slots"),
        enabled=config["enabled"],
        max_latent_tokens=config["max_latent_tokens"],
        max_action_tokens=config["max_action_tokens"],
    )
    return {
        "format": OUTPUT_FORMAT,
        "synthetic": request["synthetic"],
        "selection_scope": "action_denoise_self_attention_historical_kv",
        "mutates_cache": False,
        "changes_cache_allocation": False,
        "budget_zero_means": "unbounded_for_that_source",
        "unknown_and_current_preserved": True,
        "configuration": config,
        **selection,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_result(load_request(args.input))
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(text, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
