from dataclasses import dataclass, asdict

from pipeline.config import RoutingThresholds

FAST = "FAST"
FULL = "FULL"


@dataclass(frozen=True)
class RouteFeatures:
    token_count: int
    confidence: float
    char_count: int
    line_count: int
    word_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_features(text: str, confidence: float, word_count: int, token_count: int) -> RouteFeatures:
    lines = [line for line in text.splitlines() if line.strip()]
    return RouteFeatures(token_count, confidence, len(text), len(lines), word_count)


def decide_route(features: RouteFeatures, thresholds: RoutingThresholds, policy: str = "auto") -> str:
    if policy == "all_fast":
        return FAST
    if policy == "all_full":
        return FULL
    if features.token_count <= thresholds.max_tokens_fast and features.confidence >= thresholds.min_confidence_fast:
        return FAST
    return FULL
