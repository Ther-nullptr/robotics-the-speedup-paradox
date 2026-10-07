"""CPU contract tests for LingBot-VA KV read-window pruning."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "benchmarks/static/lingbot_robotwin/kv_read_window.py"
EXAMPLE = ROOT / "benchmarks/static/lingbot_robotwin/kv_read_window.synthetic.json"


def load_selector():
    spec = importlib.util.spec_from_file_location("lingbot_kv_window", SELECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = load_selector()


def select(request):
    config = selector.resolve_config(request)
    return selector.select_read_window(
        request["slots"],
        enabled=config["enabled"],
        max_latent_tokens=config["max_latent_tokens"],
        max_action_tokens=config["max_action_tokens"],
    )


def test_source_budgets_are_independent_and_safety_slots_survive():
    request = selector.load_request(EXAMPLE)
    before = deepcopy(request)
    result = select(request)
    assert result["selected_slots"] == [2, 4, 5, 6, 7]
    assert result["dropped_slots"] == [0, 1, 3]
    assert result["counts"] == {
        "input": 8,
        "selected": 5,
        "dropped": 3,
        "latent": 2,
        "action": 2,
        "unknown": 1,
        "current": 1,
    }
    assert request == before


def test_disabled_and_zero_budgets_do_not_prune():
    slots = selector.load_request(EXAMPLE)["slots"]
    for enabled, latent, action in ((False, 1, 1), (True, 0, 0)):
        result = selector.select_read_window(
            slots,
            enabled=enabled,
            max_latent_tokens=latent,
            max_action_tokens=action,
        )
        assert result["selected_slots"] == list(range(8))
        assert result["dropped_slots"] == []


def test_one_zero_budget_is_unbounded_only_for_its_source():
    slots = selector.load_request(EXAMPLE)["slots"]
    result = selector.select_read_window(
        slots,
        enabled=True,
        max_latent_tokens=0,
        max_action_tokens=1,
    )
    assert result["selected_slots"] == [0, 2, 3, 4, 5, 6, 7]
    assert result["dropped_slots"] == [1]


def test_equal_generation_uses_lower_slot_as_deterministic_tie_break():
    result = selector.select_read_window(
        [
            {"slot": 6, "source": "latent", "generation": 2},
            {"slot": 4, "source": "latent", "generation": 2},
        ],
        enabled=True,
        max_latent_tokens=1,
        max_action_tokens=0,
    )
    assert result["selected_slots"] == [4]
    assert result["dropped_slots"] == [6]


def test_paper_presets_match_table_five_and_have_monotone_ratios():
    assert [
        (
            selector.paper_preset(name)["enabled"],
            selector.paper_preset(name)["max_latent_tokens"],
            selector.paper_preset(name)["max_action_tokens"],
        )
        for name in selector.PAPER_PRESETS
    ] == [
        (False, 8640, 1152),
        (True, 1024, 256),
        (True, 512, 128),
        (True, 256, 64),
        (True, 128, 32),
    ]
    ratios = [
        selector.paper_preset(name)["readable_kv_ratio_vs_native"]
        for name in selector.PAPER_PRESETS
    ]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[0] == 1
    assert ratios[-1] == pytest.approx(160 / 9792)


@pytest.mark.parametrize(
    "update,message",
    [
        ({"max_latent_tokens": -1}, "max_latent_tokens"),
        ({"max_action_tokens": True}, "max_action_tokens"),
        ({"slots": [{"slot": 0, "source": "latent", "generation": -1}]}, "generation"),
        (
            {
                "slots": [
                    {"slot": 0, "source": "latent", "generation": 0},
                    {"slot": 0, "source": "action", "generation": 0},
                ]
            },
            "duplicate",
        ),
    ],
)
def test_invalid_contract_is_rejected(update, message):
    request = selector.load_request(EXAMPLE)
    request.update(update)
    with pytest.raises(ValueError, match=message):
        select(request)


def test_preset_rejects_unknown_or_conflicting_explicit_fields():
    with pytest.raises(ValueError, match="one of"):
        selector.resolve_config({"preset": "p5"})
    with pytest.raises(ValueError, match="cannot be combined"):
        selector.resolve_config({"preset": "p1", "enabled": True})


def test_cli_marks_synthetic_scope_and_does_not_claim_cache_mutation():
    result = subprocess.run(
        [sys.executable, str(SELECTOR_PATH), "--input", str(EXAMPLE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["synthetic"] is True
    assert payload["mutates_cache"] is False
    assert payload["changes_cache_allocation"] is False
    assert payload["selected_slots"] == [2, 4, 5, 6, 7]
