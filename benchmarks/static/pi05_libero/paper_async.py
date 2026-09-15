"""Describe the paper's overlap abstraction without claiming real concurrency.

The history-observation implementation selects an older observation by primitive
control steps. Its scalar timing estimate belongs to the paper model; it is not
host latency or observation age. No model/simulator code is imported here.
"""

import importlib.util
from pathlib import Path


def _load_timing_tools():
    # Resolve from this file, not cwd/sys.path: an external VLASH checkout must
    # not replace the repository's timing formula or its input validation.
    source = Path(__file__).resolve().parents[3] / "tools/compare_speedups.py"
    spec = importlib.util.spec_from_file_location("_robotics_paper_timing", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load repository timing helper: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TIMING = _load_timing_tools()


def paper_async_contract(options: dict) -> dict:
    """Validate explicit case options and return metadata plus scalar timing.

    Extra case fields are allowed but never used to infer inference latency,
    successful chunk counts, or task speedups. When inference time is absent,
    timing remains unknown until a caller provides a profile and its source.
    The source string is a declaration; this helper does not verify that source.
    """
    if not isinstance(options, dict):
        raise ValueError("paper async options must be a dictionary")

    def required(name):
        return _TIMING._required(options, name, "paper_async")

    schedule = _TIMING._text(required("schedule"), "schedule")
    if schedule not in {"sync", "paper_async"}:
        raise ValueError("schedule must be sync or paper_async")
    n_actions = _TIMING._integer(required("n_action_steps"), "n_action_steps", 1)
    overlap = _TIMING._integer(required("overlap_actions"), "overlap_actions")
    if overlap > n_actions:
        raise ValueError("overlap_actions must be <= n_action_steps")
    if schedule == "sync" and overlap != 0:
        raise ValueError("sync requires overlap_actions=0")
    action_ms = _TIMING._positive(
        required("paper_action_time_ms"), "paper_action_time_ms"
    )
    _TIMING._product(n_actions, action_ms, "action execution time")
    same_snapshot = required("delay_state_with_observation")
    if type(same_snapshot) is not bool:
        raise ValueError("delay_state_with_observation must be a boolean")

    inference_ms = required("paper_inference_time_ms")
    source = required("paper_inference_time_source")
    if source is not None:
        source = _TIMING._text(source, "paper_inference_time_source")
    if inference_ms is None:
        status = "requires_inference_profile"
        timing = {
            "hidden_inference_time_ms": None,
            "residual_inference_time_ms": None,
            "cycle_time_ms": None,
        }
    else:
        inference_ms = _TIMING._positive(inference_ms, "paper_inference_time_ms")
        if source is None:
            raise ValueError(
                "paper_inference_time_source is required with inference time"
            )
        status = "estimated"
        timing = _TIMING.chunk_timing(inference_ms, n_actions, action_ms, overlap)

    return {
        "schedule": schedule,
        "n_prime": overlap,
        "n_actions": n_actions,
        "implementation": "history_observation"
        if schedule == "paper_async"
        else "sync",
        "history_unit": "primitive_control_steps",
        "state_policy": "same_snapshot" if same_snapshot else "current_state",
        "history_fallback": "current_until_available",
        "paper_reference": "https://arxiv.org/abs/2606.28529v2",
        "timing": {
            "status": status,
            "clock_domain": "paper_model",
            "estimate_model": "paper_steady_state",
            "inference_time_ms": inference_ms,
            "inference_time_source": source,
            "action_time_ms": action_ms,
            **timing,
            "interpretation": (
                "representative_scalar_estimate; not_host_duration_or_observation_age"
            ),
        },
    }
