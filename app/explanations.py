
"""Format employee-readable routing explanations."""


def build_reasons(
    model_reasons: list[str],
    predicted_team: str,
    channel: str,
    product_family: str,
    warranty_status: str,
) -> list[str]:
    """Build transparent explanations for a routing recommendation."""

    reasons: list[str] = []

    # Keep only the first model-generated explanation.
    # Additional context is added below in a consistent format.
    if model_reasons:
        reasons.append(model_reasons[0])

    reasons.append(
        f"The request was submitted through {channel}."
    )
    reasons.append(
        f"The product recorded on the request is {product_family}."
    )
    reasons.append(
        f"The supplied warranty status is {warranty_status}."
    )
    reasons.append(
        f"The model recommends {predicted_team} as the likely "
        "final resolution team. Confirm this against the request "
        "details and applicable routing policy."
    )

    return reasons
