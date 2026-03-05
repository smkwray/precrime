# Policy Curves (NIJ Y1 Static)

Generated: 2026-03-02 15:56 EST

Artifacts:
- JSON plot spec: `reports/plots/nij_y1_policy_curves.json`
- PNG figure (rendered from JSON): `docs/figures/nij_y1_policy_tradeoffs.png`

Configuration:
- Model: XGBoost best config for Y1 static (`calibration=platt`)
- Split seed: `42` (same fit/cal/test protocol as other NIJ reports)
- Group counts in test split: BLACK=2045, WHITE=1561

Caution:
- Curves summarize threshold-policy tradeoffs (budget vs error rates), not a recommended operational policy.
- Subgroup curves can vary with sample size and split randomness; treat small differences as uncertain.