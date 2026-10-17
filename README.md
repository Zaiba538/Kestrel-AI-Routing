# Kestrel Home Appliances — AI Service Request Routing

## Overview

This project builds a machine-learning system that recommends a support team for incoming Kestrel Home Appliances service requests.

The selected model predicts the team that ultimately resolved each request (`final_team`) rather than simply reproducing the initial queue assignment. This target is intended to better reflect the business outcome, but it is distinct from the historical-label acceptance criterion.

The solution includes batch prediction, a FastAPI prediction service, a Streamlit interface, a confidence-based human-review policy, and automated tests.

## Model and Validation

- **Selected model:** TF-IDF text features with logistic regression.
- **Training records:** 10,822.
- **Chronological validation records:** 2,168.
- **Final-outcome accuracy:** 83.86%.
- **Final-outcome macro-F1:** 84.06%.
- **Existing-bot final-outcome accuracy:** 76.61% on the same validation set.
- **Improvement over the existing bot:** 7.24 percentage points.
- **Final-outcome majority-class baseline:** 23.25%.

The chronological split uses earlier records for training and later records for validation. This better represents a time-ordered deployment scenario than a random split, although performance may change as request patterns and operations evolve.

A separate model achieved **94.37% accuracy against historical team labels**. This measures agreement with historical queue assignments; it is not the accuracy of the selected final-outcome model.

The selected model has not demonstrated 90% overall accuracy against final resolution teams. The historical-label acceptance criterion must be evaluated separately.

## Confidence-Based Human Review

The provisional review policy uses:

- Minimum prediction confidence: **0.60**.
- Minimum top-two prediction-score margin: **0.10**.

On the chronological validation set:

- **Automatically routed:** 1,764 of 2,168 requests.
- **Coverage:** 81.37%.
- **Accuracy among automatically routed requests:** 95.80%.
- **Referred for human review:** 404 requests, or 18.63%.

These are retrospective validation results, not guarantees of production performance. The thresholds were evaluated on the same validation set used to assess the policy, so results may be optimistic. Model confidence has not been formally calibrated.

Requests flagged for review should be checked by an appropriate support employee before routing.

## Supported Teams

The model predicts one of seven support teams:

- Repairs
- Billing
- Filters & Consumables
- Installs & Demo
- Product Advice
- Returns & Replacement
- Warranty Claims

Historical team names are normalized to current team names.

## Project Structure

- `app/` — FastAPI service, prediction logic, explanations, and review policy.
- `src/` — data loading, training, evaluation, error analysis, and batch prediction.
- `ui/` — Streamlit interface.
- `tests/` — automated tests.
- `data/` — local assessment datasets; excluded from Git.
- `models/` — trained model artifact; excluded from Git by default.
- `reports/` — evaluation and operational analysis; excluded from Git by default.
- `predictions.csv` — batch predictions for unlabelled assessment records. Handle as restricted operational data.
- `memo.md` — business decision memo.

## Requirements and Setup

Requires Python 3.11 and a virtual environment.

Open PowerShell in the project root.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the virtual environment already exists, activate it without recreating it.

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Prepare the assessment data

Obtain the assessment data through the approved channel. Place the required input files in the local `data/` directory using the filenames expected by the source code.

Do not commit or publish restricted source data.

### 4. Train the model

Run the training pipeline to generate the model artifact and evaluation outputs locally:

```powershell
python -m src.train
```

Confirm that training completes successfully and creates the model artifact expected by the API configuration.

## Run the Pipeline

### Generate assessment predictions

After training the model, run:

```powershell
python -m src.predict_batch
```

This generates `predictions.csv` in the project root with the required `request_id` and `team` columns. The pipeline validates the output schema, record count, request ID uniqueness, and predicted team names.

Treat this file as restricted operational data unless an authorized submission channel explicitly permits sharing it.

### Run automated tests

```powershell
python -m pytest tests -v
```

The test suite checks data preparation, team-name normalization, model loading, prediction probabilities, and API behaviour.

## Run the API

After generating the model artifact, run:

```powershell
python -m uvicorn app.main:app --reload
```

Endpoints:

- Health check: <http://127.0.0.1:8000/health>
- Interactive API documentation: <http://127.0.0.1:8000/docs>
- Prediction endpoint: `POST /predict`

The prediction endpoint returns a suggested team, confidence score, explanation, human-review indicators, and model validation information.

Keep the API terminal running while testing requests or using the Streamlit interface.

## Run the Streamlit Interface

Open a second terminal in the project root and activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run ui/streamlit_app.py
```

Open <http://localhost:8501>.

The interface sends requests to the FastAPI service and displays the suggested team, confidence, explanation, and review recommendation. The API must remain running.

## Business Evaluation and Limitations

The selected model achieved 83.86% accuracy against final resolution teams, compared with 76.61% for the existing bot on the same chronological validation set.

This is evidence of improved historical outcome prediction, not proof that the model will reduce transfers, additional customer contacts, technician visits, or total operating costs.

The available dataset covers approximately 14.9 months rather than the 18 months described in the business request. Performance on legacy Zoho records has not been independently validated.

The recommended next step is a controlled shadow-mode pilot. Compare model recommendations with actual resolutions, transfers, additional customer contacts, manual overrides, and resolution times. Report the 90% historical-label criterion separately from final-outcome accuracy.

## Cost Considerations

The stated annual bot licence cost is **₹3.2 lakh**.

The historical cost analysis estimated annualized transfer costs of approximately **₹9.55 lakh**. A broader scenario that includes assumed additional-contact costs was approximately **₹14.71 lakh annually**.

These figures estimate current-process costs; they are not proven avoidable costs or guaranteed savings. The additional-contact scenario applies ₹260 to every historical-label/final-team disagreement, which is an assumption rather than a measured invoice. Hosting, maintenance, human review, and other operating costs must also be assessed.

No net savings claim should be made until a monitored pilot measures actual operational impact.

## Data Handling

Customer and operational data must be handled according to the applicable business policy.

Do not publish source datasets, request-level validation outputs, restricted model outputs, credentials, or other protected information in a public repository without authorization.

The local `.gitignore` excludes data CSVs, reports, trained model artifacts, and local diagnostic files. Review Git status and repository history before publishing.

The assessment requests a public repository containing `predictions.csv`, while the supplied operations policy prohibits public sharing of customer and operational data. Obtain written clarification or an approved submission method before publishing that file or other restricted artifacts.

## Deployment Recommendation

Proceed with a controlled shadow-mode pilot with human review. Evaluate data governance, performance by team and channel, ambiguous requests, legacy-source performance, and measurable operational outcomes before considering live routing.

**The current evidence does not justify immediately switching off the existing bot.**
