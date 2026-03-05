# NIJ-Style Score Report

Generated: 2026-03-02 08:20 EST

Split policy: same seeded split as other NIJ reports (`fit/cal/test` from random seed 42).

Scoring views:
- NIJ fairness component at `t=0.5`: `FP_component = 1 - |FPR_black - FPR_white|`.
- NIJ fair+accurate index: `(1 - avg_sex_brier) * FP_component` (higher is better).
- Additive proxy requested for this task: `avg_sex_brier + lambda * |FPR_black - FPR_white|` (lower is better).
- Additive `lambda` used here: `1.000`.

| Horizon | Dataset | Variant | Brier(M) | Brier(F) | Avg-Sex Brier | FPR Black | FPR White | ΔFPR | NIJ FP Component | NIJ Fair+Accurate | Additive Score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| y1 | static | with_race | 0.19316 | 0.14355 | 0.16835 | 0.05308 | 0.05370 | 0.00061 | 0.99939 | 0.83114 | 0.16896 |
| y1 | static | without_race | 0.19349 | 0.14352 | 0.16851 | 0.05165 | 0.05282 | 0.00117 | 0.99883 | 0.83052 | 0.16967 |
| y2 | dynamic | with_race | 0.17790 | 0.14526 | 0.16158 | 0.05212 | 0.05450 | 0.00238 | 0.99762 | 0.83643 | 0.16396 |
| y2 | dynamic | without_race | 0.17754 | 0.14501 | 0.16127 | 0.01931 | 0.02014 | 0.00084 | 0.99916 | 0.83802 | 0.16211 |
| y3 | dynamic | with_race | 0.15015 | 0.10352 | 0.12683 | 0.00575 | 0.00767 | 0.00192 | 0.99808 | 0.87149 | 0.12875 |
| y3 | dynamic | without_race | 0.15007 | 0.10311 | 0.12659 | 0.00575 | 0.00460 | 0.00115 | 0.99885 | 0.87241 | 0.12773 |

Notes:
- NIJ formula source: NIJ Challenge judging criteria (`FP=1-|FPR_black-FPR_white|`, `Fair+Accurate=(1-BS)*FP`).
- This report uses sex-averaged Brier (`BS`) to align with the average male/female framing in the challenge.
- `Additive Score` is a local diagnostic view requested for this project; it is not the official NIJ index.
