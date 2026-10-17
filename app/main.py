
"""FastAPI application for Kestrel service-request routing."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from app.explanations import build_reasons
from app.policy import assess_review
from app.predictor import RoutingPredictor
from app.schemas import RoutingRequest, RoutingResponse

app = FastAPI(
    title="Kestrel Home AI Routing",
    description=(
        "Local service-request routing with model explanations "
        "and human-review flags."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_predictor() -> RoutingPredictor:
    """Load the trained model once and reuse it across requests."""
    return RoutingPredictor()


@app.get("/health")
def health():
    """Return the application health status."""
    return {"status": "ok", "service": "kestrel-routing"}


@app.post("/predict", response_model=RoutingResponse)
def predict(request: RoutingRequest):
    """Predict the most likely final resolution team for a request."""

    try:
        predictor = get_predictor()
        record = request.model_dump()
        prediction = predictor.predict(record)

        review_required, review_reasons = assess_review(
            prediction["confidence"],
            prediction["probabilities"],
        )

        reasons = build_reasons(
            model_reasons=prediction["reasons"],
            predicted_team=prediction["predicted_team"],
            channel=request.channel,
            product_family=request.product_family,
            warranty_status=request.warranty_status,
        )

        return RoutingResponse(
            request_id=request.request_id,
            predicted_team=prediction["predicted_team"],
            confidence=prediction["confidence"],
            reasons=reasons,
            review_required=review_required,
            review_reasons=review_reasons,
            model_name=prediction["model_name"],
            model_validation_accuracy=prediction[
                "model_validation_accuracy"
            ],
            disclaimer=(
                "This model predicts the team that historically resolved "
                "similar service requests. Confidence is a model score, "
                "not a guarantee of correctness. Validate recommendations "
                "against operational policy and actual resolution outcomes. "
                "Human review is recommended for uncertain or ambiguous "
                "requests."
            ),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The routing model is unavailable. "
                "Run `python -m src.train` and restart the service."
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The routing service could not process this request.",
        ) from exc
