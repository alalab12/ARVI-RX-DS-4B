from __future__ import annotations


DEFINITIVE_CLASSES = {"normal", "suspected_opacity"}
ALLOWED_CLASSES = DEFINITIVE_CLASSES | {"uncertain"}


def agreement_or_abstain(primary: str, secondary: str | None) -> str:
    """Fuse two predictions without directly flipping a definitive class."""

    if primary not in ALLOWED_CLASSES:
        raise ValueError(f"Unknown primary class: {primary}")
    if secondary is None:
        return primary
    if secondary not in ALLOWED_CLASSES:
        raise ValueError(f"Unknown secondary class: {secondary}")
    if primary == "uncertain":
        return secondary if secondary in DEFINITIVE_CLASSES else "uncertain"
    if secondary == primary:
        return primary
    return "uncertain"


def routing_bounds(
    low_threshold: float,
    high_threshold: float,
    margin: float,
) -> tuple[float, float]:
    """Expand a score interval by a non-negative margin inside [0, 1]."""

    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= low < high <= 1")
    if margin < 0:
        raise ValueError("Routing margin must be non-negative")
    return max(0.0, low_threshold - margin), min(1.0, high_threshold + margin)
