# Forecasting the Effective Federal Funds Rate

A reproducible study of **one-month-ahead EFFR forecasts**, comparing a persistence baseline, Ridge regression and Random Forest on historical monthly data. The project emphasizes chronological evaluation, past-only features and honest baseline comparisons.

**Main finding:** repeating the previous month's rate wins on training cross-validation. Ridge nearly matches persistence on holdout RMSE but has higher MAE; Random Forest performs substantially worse. Model complexity does not guarantee better forecasts.

![Monthly EFFR forecasts on the holdout](reports/forecast.png)

## Reproduce the results

Use Python **3.12** for the recorded environment. From a fresh checkout:

```bash
git clone https://github.com/ageefox/effr-forecasting-ml.git
cd effr-forecasting-ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
effr-train --data data/effr_monthly.csv --output reports
effr-verify --data data/effr_monthly.csv --expected reports
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. The official-source CSV snapshot is included: no API keys or downloads are needed for training. The run is headless and saves figures rather than opening plot windows. A full run typically takes under a few minutes, depending on hardware.

The package also supports `python -m effr_forecasting.train`. Input and output paths are explicit, while `--test-fraction` and `--seed` default to `0.2` and `42`. Direct dependencies live in `pyproject.toml`; `requirements-lock.txt` records the tested environment.

## Recorded results

The CSV contains **752 monthly observations**, July 1954–February 2017. After a fixed 12-month feature warm-up, training uses **592 months** (July 1955–October 2004) and the chronological holdout uses **148 months** (November 2004–February 2017).

Metrics below come from the checked-in [metrics.csv](reports/metrics.csv). RMSE and MAE are **percentage points**, not relative percentages; 0.15 percentage points equals 15 basis points.

- **Persistence:** CV RMSE **0.4968**; holdout RMSE **0.1526**, MAE **0.0700**, R² **0.9934**.
- **Ridge:** CV RMSE **0.5632**; holdout RMSE **0.1535**, MAE **0.1080**, R² **0.9933**.
- **Random Forest:** CV RMSE **1.2579**; holdout RMSE **0.7514**, MAE **0.6569**, R² **0.8396**.

Persistence is selected using training CV and also has slightly lower holdout RMSE and MAE than Ridge. The difference is small, and the high R² values largely reflect persistent rate levels. The original Random Forest result does not hold under chronological evaluation.

Random Forest also illustrates a regime-change limitation: its predictions are averages of training targets and cannot extrapolate below the training response range. The training target never falls below 0.63%, while the holdout reaches 0.07% during the post-2008 near-zero-rate regime.

## Forecasting method

The prediction for month t uses only EFFR observations through t−1, assuming the prior monthly observation has been released early in month t. Features include lags at 1, 2, 3, 6 and 12 months; trailing 3-, 6- and 12-month means and standard deviations; and the preceding monthly change.

Five expanding time-series folds tune Ridge and Random Forest on the training period. Ridge scaling is fitted within each fold. The final models are then fitted once on the training block. During the holdout, each one-step forecast uses previous observed rates, including earlier holdout observations, so the result is a **rolling one-step evaluation**.

The earlier project's macroeconomic columns are excluded because their original preparation includes interpolation and lacks release/vintage records. The current dataset contains only the official H.15 EFFR series used by the benchmark. See the [data source record](data/README.md) and [methodology and leakage audit](docs/methodology.md).

### Why an autoregressive benchmark fits EFFR

EFFR is the transaction-based overnight rate that the Federal Reserve steers toward the FOMC's target rate or range. It tends to remain close to its recent level between policy changes and move in steps when the policy stance changes. The previous month's rate is therefore a meaningful baseline for testing whether historical patterns add predictive value. See the [New York Fed's EFFR definition](https://www.newyorkfed.org/markets/reference-rates/effr) and [monetary-policy implementation overview](https://www.newyorkfed.org/markets/domestic-market-operations/monetary-policy-implementation).

Because monthly averages can combine days before and after an FOMC decision, a future policy model should use meeting dates, real-time economic data, the prevailing target range and market expectations. This project stays focused on short-run persistence in the realized rate.

## Saved artifacts

- [Dated predictions](reports/predictions.csv) for every model and the baseline.
- [Run metadata](reports/run.json): data SHA-256, package versions, seed, split dates, feature list, CV boundaries and selected parameters.
- [Ridge CV results](reports/ridge_cv.csv) and [Random Forest CV results](reports/randomforest_cv.csv), including every searched configuration.
- [Feature importance](reports/feature_importance.csv) and the figure below. Impurity importance is descriptive, not causal.

![Training feature importance](reports/feature_importance.png)

## Project layout

```text
data/effr_monthly.csv       Official H.15 historical data snapshot
src/effr_forecasting/       Import-safe data validation, features and training CLI
tests/                     Temporal integrity and pipeline regression checks
reports/                   Reproduced metrics, predictions, metadata and figures
docs/methodology.md        Forecast design, leakage audit and data limitations
.github/workflows/ci.yml   Tests and full training smoke run
```

Tests cover past-only features, exclusion of unrelated inputs, invalid data handling, sorting and invariance of tuning to altered holdout labels. CI also reruns the benchmark and verifies the published artifacts.

## Limitations and next research steps

The benchmark dataset is the Federal Reserve Board's H.15 monthly EFFR series, documented in [data/README.md](data/README.md). Data end in 2017, and the holdout was examined during earlier project work, so it should be treated as a historical portfolio benchmark rather than fresh external validation.

A meaningful extension would obtain documented newer observations for external validation, then add macroeconomic features using historical release dates and vintages. Simply lagging the existing interpolated macro columns would not establish real-time validity.

## Contributors and license

Original project: **Anastasia Galkova and Bakr Bouhaya**. See [LICENSE](LICENSE) for code licensing and [data/README.md](data/README.md) for data attribution and terms.
