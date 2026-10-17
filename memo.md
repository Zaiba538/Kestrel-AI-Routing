# Decision Memo: Kestrel AI Service-Request Routing

**To:** Ritu Deshpande, Head of D2C Operations  
**Subject:** Recommendation on the AI Service-Request Routing Model

## Decision

Proceed with a **controlled shadow-mode pilot with human review**. Do not switch off the existing routing bot yet.

The selected text-based logistic regression model achieved **83.86% accuracy** and **84.06% macro-F1** against final resolution teams on a chronological validation set of 2,168 requests. On the same validation set, the existing bot achieved 76.61% outcome accuracy, giving the model a **7.24-percentage-point improvement**.

A separate historical-label model achieved **94.37% accuracy** against historical queue assignments. This result does not establish 90% accuracy against final resolution teams. The selected outcome-based model has not demonstrated the requested 90% historical-label agreement criterion either; that criterion must be evaluated separately against the specified target.

The business should confirm which target determines acceptance, and both metrics should be reported separately.

## Routing Policy and Human Review

Under the current provisional confidence policy, 1,764 of 2,168 validation requests would be automatically routed, representing **81.37% coverage**. Accuracy among these automatically routed requests was **95.80%**. The remaining 404 requests, or 18.63%, would be referred for human review.

These are retrospective validation results, not guarantees of production performance. The confidence thresholds have not been formally calibrated and should be treated as initial pilot settings rather than established production thresholds.

## Business and Cost Implications

The existing bot licence costs **₹3.2 lakh per year**.

The historical cost analysis identified:

- **3,902 recorded transfers** over approximately 14.9 months.
- An estimated annualized transfer cost of approximately **₹9.55 lakh**, using ₹305 per transfer.
- An annualized combined scenario of approximately **₹14.71 lakh**, including an assumed ₹260 additional-contact cost applied to each of 2,471 historical-label/final-team disagreements.

The combined scenario is not a verified savings opportunity. The additional-contact amount is an assumption rather than a measured invoice, and the model has not demonstrated how many transfers or customer contacts it would prevent.

Hosting, maintenance, human-review, and other operational costs must be included in any financial comparison. **No realized or net savings are claimed at this stage.**

## Recommended Actions for the Next Seven Days

1. **Validate the data scope.** The available history covers approximately 14.9 months, shorter than the requested 18 months. Confirm whether additional records are available and separately assess legacy Zoho performance.
2. **Run in shadow mode.** Generate recommendations without changing live routing decisions.
3. **Review high-risk errors.** Examine payment-related requests, purifier consumables, vague callback requests, ambiguous descriptions, and low-confidence predictions.
4. **Measure actual outcomes.** Track final-team accuracy, transfers per request, additional customer contacts, technician visits, manual overrides, and time to resolution.
5. **Evaluate the review policy.** Monitor coverage, reviewed-request workload, error rates, and confidence reliability. Adjust thresholds only through a defensible evaluation process.
6. **Confirm governance and deployment costs.** Obtain approval for data handling, access, hosting, maintenance, and the approved submission method before sharing operational outputs.

## Go/No-Go Criteria

Consider a limited live pilot only after data-governance checks and the legacy-data evaluation are complete. Expand deployment only when monitored evidence demonstrates reliable routing and measurable net operational benefit without unacceptable customer impact.

The 90% acceptance criterion must be evaluated against the target specified by the business. Historical-label accuracy must not be substituted for final-outcome accuracy.

## Final Recommendation

The current model shows a promising improvement over the existing bot on the available chronological outcome-validation set. The evidence supports further controlled evaluation, not an unconditional production rollout or immediate cancellation of the existing contract.
