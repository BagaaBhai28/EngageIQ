# EngageIQ

A local Flask application for historical Facebook engagement analytics, pre-publication prediction and engagement clustering. The interface uses a quiet light palette, teal emphasis, persistent sidebar navigation, sample-size context and separate analytics and prediction workflows.

## Run on Windows

For automatic setup, dependency checks, data/model recovery and browser launch:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

Or double-click `START_ENGAGEIQ.cmd`. No administrator privileges or permanent policy changes are required. The launcher finds/installs Python 3.13 via winget, uses a local virtual environment, verifies all pinned packages, checks real data and models, then selects a free port from 5055 to 5065. It opens the browser when ready. Keep the terminal open; Ctrl+C stops the site. Use `-SetupOnly` to only install/check, or `-NoBrowser` to suppress opening a browser. Internet is needed for first-time dependency installation or missing raw data. The bundled PDF guide is `output/pdf/EngageIQ_Setup_and_Project_Guide.pdf`.

Manual setup is also supported:

Python 3.13 was used for this build. From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python app.py
```

Open http://127.0.0.1:5055. The CSV datasets, trained model artifacts and Chart.js library are bundled, so normal use needs no internet. Stop the server with Ctrl+C. No cloud deployment is configured. This is a local research workspace, not a multi-user hosted service.

## Reproduce the data and models

```powershell
.\.venv\Scripts\python scripts/download_datasets.py
.\.venv\Scripts\python scripts/prepare_datasets.py
.\.venv\Scripts\python scripts/train_models.py
.\.venv\Scripts\python -m pytest -q
```

Download requires internet. The downloader preserves TLS verification and uses Windows curl with the OS trust store if Python's trust store fails. Exact raw source URLs and SHA-256 checksums are in `data/raw/sources.json`. Keep the pinned dependencies when loading the bundled trusted Joblib models. Restart Flask after retraining.

## Implemented stages

1. Flask architecture, dependencies, JSON errors and local SQLite.
2. Real UCI dataset download and cleaning, with row-level quality counts.
3. Descriptive analytics: reactions, comments, shares, mean, median, formats and time.
4. Leakage-safe preprocessing and expanding-month model validation.
5. Nine model/target candidates including a median baseline; saved evaluation artifact.
6. Log-scaled K-Means, selecting k from 2–5 by sampled silhouette.
7–8. Analytics, prediction, upload, export and history APIs with SQLite persistence.
9–10. Responsive interface, local interactive charts, filters and scenario comparison.
11. Automated functional/integrity tests and documentation; desktop and narrow viewport review.

## Important interpretation

Analytics: reactions + comments + shares. Likes are already included in reactions and are not added again. No reach-based engagement rate is shown. Analytics and prediction datasets are never pooled. Recorded historical timestamps have no confirmed timezone.

Prediction: Facebook Metrics from 2014, a single cosmetics brand, 500 posts. Train months 1–9 (345 posts), test months 10–12 (155 posts). Ridge with log target was selected on validation MAE. Test MAE **135.86**, RMSE **252.03**, R² **−0.011**; median baseline MAE **136.71**. The gain is very small and predictive utility is limited. Predictions are demonstrations, not contemporary marketing recommendations. The saved model retains the untouched-test evaluation setup and is not refitted on all 500 rows.

CSV upload supports `post_type, post_date, comments, shares, reactions` or `likes`, plus optional `post_id`; native UCI names work too. It reports rejected/duplicate rows, keeps real viral outliers and enables descriptive analysis without retraining. Uploaded files and history stay under `instance/`, outside public assets. This browser's signed session cookie selects its history and upload. Data survives a server restart while the cookie and instance files remain available. Clearing cookies loses access to that browser's records.

See [API documentation](docs/API_DOCUMENTATION.md), [model report](docs/MODEL_REPORT.md), [dataset sources](docs/DATASET_SOURCES.md), and the supplied [implementation plan](IMPLEMENTATION_PLAN.md).
