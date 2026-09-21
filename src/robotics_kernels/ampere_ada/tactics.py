"""Explicit CUTLASS configurations, without model-size dispatch thresholds."""

# M, N, INT8 K (INT4 doubles K), warp M, warp N, pipeline stages.
TACTICS = {
    0: (64, 128, 64, 32, 64, 3),
    1: (128, 128, 64, 32, 64, 3),
    2: (64, 64, 64, 32, 32, 3),
    3: (128, 64, 64, 32, 32, 3),
    4: (64, 128, 128, 32, 64, 3),
    5: (128, 128, 64, 32, 64, 4),
    6: (64, 256, 64, 32, 64, 3),
    7: (32, 128, 64, 32, 64, 3),
}
TACTIC_IDS = tuple(TACTICS)


def describe(tactic, bits):
    if type(tactic) is not int or tactic not in TACTICS or bits not in (4, 8):
        raise ValueError("Unknown integer tactic or bit width")
    m, n, k, wm, wn, stages = TACTICS[tactic]
    k *= 2 if bits == 4 else 1
    return {"tile": [m, n, k], "warp": [wm, wn, k], "stages": stages}
