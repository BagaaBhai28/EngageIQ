# API

Responses use `{ "data": ... }`; errors use `{ "error": "message" }` with appropriate HTTP status. Upload limit: 8 MB and 100,000 input rows.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/analytics` | Summary, format groups, monthly trend, reaction mix, timing cells, clusters and top posts |
| GET | `/api/model` | Validation comparisons, test metrics, test predictions and training provenance |
| POST | `/api/predict` | Seven pre-publication fields; inference and persistent browser-specific history |
| GET | `/api/history` | Most recent 100 predictions for this browser |
| POST | `/api/upload` | Multipart field `file`; CSV validation, cleaning, preview and local storage |
| GET | `/api/export` | Download the filtered analytics population as CSV |

Analytics/export query parameters: `dataset=builtin` or `uploaded`, `type=photo|video|status|link`, `start=YYYY-MM-DD`, `end=YYYY-MM-DD` (inclusive). Omit type/dates for all records. Invalid dates return 400. No matching records returns a valid empty summary.

Prediction JSON example:

```json
{"Page total likes":120000,"Type":"Photo","Category":2,"Post Month":10,"Post Weekday":6,"Post Hour":18,"Paid":0}
```

Weekday coding is Sunday=1 through Saturday=7. Category codes are 1 action, 2 product, 3 inspiration. Numeric inputs must be integers with finite supported values. The response includes interactions, training-relative level, extrapolation warning, selected model and test MAE. MAE is not a confidence interval. The API loads only bundled trusted local model artifacts.
