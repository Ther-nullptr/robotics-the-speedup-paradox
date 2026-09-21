"""CPU argument definitions and optional policy factory integration."""

from contextlib import contextmanager, ExitStack
from types import FunctionType
from .config import OptimizationConfig, SWITCHES


def add_arguments(parser):
    parser.add_argument("--model-runtime", choices=("owned", "native"), default="owned")
    parser.add_argument(
        "--enable",
        action="append",
        choices=SWITCHES,
        default=[],
        help="Enable one independent optimization (repeatable)",
    )
    parser.add_argument(
        "--precision", choices=("bf16", "int8", "int4", "fp8", "fp4"), default="bf16"
    )
    parser.add_argument(
        "--quant-scope",
        action="append",
        choices=("text", "expert", "vision", "projector", "dit"),
    )
    parser.add_argument("--integer-tactic", type=int, choices=(0, 1), default=0)


def configuration(args, model="pi05"):
    config = OptimizationConfig(
        switches=tuple(args.enable),
        precision=args.precision,
        scopes=tuple(args.quant_scope or (("text",) if model == "pi05" else ("dit",))),
        tactic=args.integer_tactic,
    )
    if (
        "shared_quant" in config.enabled or "activation_quant_fusion" in config.enabled
    ) and not config.precision.startswith("int"):
        raise ValueError("Integer preparation switches require int8 or int4 precision")
    allowed = (
        set(SWITCHES) - {"modulation"}
        if model == "pi05"
        else {
            "modulation",
            "gated_residual",
            "cuda_graph",
            "shared_quant",
            "activation_quant_fusion",
        }
    )
    if set(config.enabled) - allowed:
        raise ValueError(f"Unsupported {model} optimization switch")
    if (
        model == "pi05"
        and "condition_cache" in config.enabled
        and "flow_loop" not in config.enabled
    ):
        raise ValueError("condition_cache requires flow_loop")
    if "empty_image_cache" in config.enabled and "cuda_graph" not in config.enabled:
        raise ValueError("empty_image_cache requires cuda_graph")
    if args.model_runtime == "native" and (
        config.enabled or config.precision != "bf16"
    ):
        raise ValueError("Local optimizations require --model-runtime owned")
    if (
        model == "pi05"
        and "cuda_graph" in config.enabled
        and "flow_loop" not in config.enabled
    ):
        raise ValueError("PI0.5 cuda_graph requires flow_loop")
    if model == "cosmos" and config.scopes != ("dit",):
        raise ValueError("Cosmos quantization scope must be dit")
    if model == "pi05" and "dit" in config.scopes:
        raise ValueError("PI0.5 does not support the dit quantization scope")
    return config


def from_dict(data):
    return OptimizationConfig(
        switches=tuple(data["switches"]),
        precision=data["precision"],
        scopes=tuple(data["scopes"]),
        tactic=data["tactic"],
    )


@contextmanager
def owned_pi05_factory(evaluator, config):
    """Retain upstream environment/processor setup and own the executed policy.

    Clone the factory function's globals rather than modifying the installed
    LeRobot module or its registry. Changes to this evaluator are restored.
    """
    from robotics_bench.models.pi05.modeling_pi05 import PI05Policy
    from .pi05 import optimize_pi05

    original = evaluator.make_lerobot_policy
    namespace = dict(original.__globals__)
    original_class = namespace["get_policy_class"]
    namespace["get_policy_class"] = lambda name: (
        PI05Policy if name == "pi05" else original_class(name)
    )
    factory = FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    factory.__kwdefaults__ = original.__kwdefaults__
    reports = []
    with ExitStack() as stack:

        def make(*args, **kwargs):
            policy = factory(*args, **kwargs)
            policy.eval()
            reports.append(stack.enter_context(optimize_pi05(policy.model, config)))
            return policy

        evaluator.make_lerobot_policy = make
        try:
            yield reports
        finally:
            evaluator.make_lerobot_policy = original
