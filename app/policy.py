
"""Review flags for uncertain service-routing predictions."""

REVIEW_CONFIDENCE_THRESHOLD = 0.60
REVIEW_TOP_TWO_MARGIN_THRESHOLD = 0.10


def assess_review(
    confidence: float,
    probabilities: dict,
) -> tuple[bool, list[str]]:
    """Flag uncertain predictions for human review.

    This policy does not change the model's predicted team.
    Thresholds are provisional and require operational validation.
    """

    reasons: list[str] = []

    if confidence < REVIEW_CONFIDENCE_THRESHOLD:
        reasons.append(
            "The prediction is uncertain because the model's confidence "
            "is below the provisional 60% threshold. A support agent "
            "should review the request before routing."
        )

    ranked = sorted(
        (float(value) for value in probabilities.values()),
        reverse=True,
    )

    if (
        len(ranked) >= 2
        and ranked[0] - ranked[1] < REVIEW_TOP_TWO_MARGIN_THRESHOLD
    ):
        reasons.append(
            "Two or more teams have similar model scores, so the intended "
            "destination is ambiguous. A support agent should confirm "
            "the correct team."
        )

    return bool(reasons), reasons
