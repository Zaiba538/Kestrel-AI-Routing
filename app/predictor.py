
"""Load the trained routing model and generate predictions."""

from pathlib import Path
import re

import joblib
import pandas as pd

from src.data import normalize_request_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "router.joblib"

# Remove weak, generic, and uninformative terms from explanations.
FILLER_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "for",
    "from", "get", "got", "has", "have", "he", "her", "here",
    "him", "his", "i", "if", "in", "into", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "please", "she",
    "so", "that", "the", "their", "them", "there", "they",
    "this", "to", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "need", "want", "help", "issue", "problem",
    "request", "customer",
}

# Generic terms that are rarely useful by themselves.
GENERIC_TERMS = {
    "product", "service", "support", "team", "item", "thing",
    "power", "working", "work", "use", "used", "using",
}


def clean_feature_name(feature_name: str) -> str:
    """Convert an internal TF-IDF feature into readable text."""
    name = feature_name.replace("text__", "")
    name = name.replace("categorical__", "")
    name = name.replace("_", " ").strip().lower()
    name = re.sub(r"\s+", " ", name)

    # Avoid explanations that contain only generic or filler terms.
    words = name.split()
    meaningful_words = [
        word for word in words
        if word not in FILLER_WORDS and word not in GENERIC_TERMS
    ]

    if not meaningful_words:
        return ""

    cleaned = " ".join(meaningful_words)

    # Ignore very short fragments and numeric-only features.
    if len(cleaned) < 3 or not re.search(r"[a-z]", cleaned):
        return ""

    return cleaned


class RoutingPredictor:
    """Local inference service using the saved scikit-learn pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Run `python -m src.train` first."
            )

        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.model_name = bundle["model_name"]
        self.feature_columns = bundle["feature_columns"]
        self.validation_metrics = bundle["validation_metrics"]
        self.target_column = bundle.get("target_column", "unknown")

    def predict(self, record: dict) -> dict:
        """Predict the most likely final resolution team."""

        values = {
            column: record.get(column, "")
            for column in self.feature_columns
        }
        values["request_text"] = normalize_request_text(
            values.get("request_text", "")
        )

        row = pd.DataFrame([values], columns=self.feature_columns)

        probabilities = self.pipeline.predict_proba(row)[0]
        classifier = self.pipeline.named_steps["classifier"]
        classes = list(classifier.classes_)

        best_index = int(probabilities.argmax())
        predicted_team = str(classes[best_index])
        confidence = float(probabilities[best_index])

        ordered_probabilities = sorted(
            (float(p) for p in probabilities),
            reverse=True,
        )
        top_two_margin = (
            ordered_probabilities[0] - ordered_probabilities[1]
            if len(ordered_probabilities) > 1
            else ordered_probabilities[0]
        )

        return {
            "predicted_team": predicted_team,
            "confidence": confidence,
            "top_two_margin": float(top_two_margin),
            "probabilities": {
                str(team): float(probability)
                for team, probability in zip(classes, probabilities)
            },
            "reasons": self.explain(row, predicted_team),
            "model_name": self.model_name,
            "model_validation_accuracy": float(
                self.validation_metrics["accuracy"]
            ),
        }

    def explain(self, row: pd.DataFrame, predicted_team: str) -> list[str]:
        """Return readable feature signals, not definitive causal reasons."""

        try:
            features = self.pipeline.named_steps["features"]
            classifier = self.pipeline.named_steps["classifier"]

            transformed = features.transform(row)
            feature_names = features.get_feature_names_out()
            class_index = list(classifier.classes_).index(predicted_team)

            contributions = transformed.multiply(
                classifier.coef_[class_index]
            ).toarray()[0]

            positive_indices = contributions.argsort()[::-1]
            reasons = []

            for index in positive_indices:
                if float(contributions[index]) <= 0:
                    break

                feature_name = clean_feature_name(str(feature_names[index]))
                if not feature_name:
                    continue

                if feature_name not in reasons:
                    reasons.append(feature_name)

                if len(reasons) >= 3:
                    break

            if reasons:
                return [
                    "Relevant learned text signals: "
                    + ", ".join(reasons)
                    + ". These are model signals, not proof of the issue."
                ]

        except (ValueError, AttributeError, IndexError, TypeError):
            pass

        return [
            "The model selected this team from the request and its learned "
            "routing patterns. No reliable readable feature signals were "
            "available, so human review may be appropriate."
        ]
