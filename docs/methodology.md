# Forecast design and audit

## Information available at forecast time

The target is the CSV's monthly EFFR observation for month t. The forecast origin is the first date after the completed month-t−1 observation is available. This is not a forecast made before month t begins. Inputs are EFFR lags 1, 2, 3, 6 and 12, trailing means and standard deviations over 3, 6 and 12 months ending at t-1, and the change from t-2 to t-1. The first 12 rows are excluded as a fixed warm-up.

The final 20% of usable dates are held out. Five expanding training folds select hyperparameters using mean fold RMSE. Standardization is fitted separately inside each Ridge training fold. Random Forest needs no scaling. There is no fitted imputation, clipping or feature selection. The lowest training-CV RMSE selects among Ridge, Random Forest and persistence; holdout scores do not determine this choice.

Models are fitted once on the training block. Each later one-month prediction receives the preceding observed EFFR, including earlier holdout observations. This is rolling one-step evaluation rather than a recursive forecast of the full holdout from one historical date. The same information rule applies within validation folds.

## External validation

The development dataset ends in February 2017. Before loading later observations, the feature list, Ridge penalty, Random Forest settings, and training-CV model choice were fixed in `reports/run.json`. The external run reads those settings, checks that the feature list has not changed, and refits the models on all usable development observations. It then forecasts each month from March 2017 through August 2026 under the same rolling one-step information rule.

Uncertainty estimates compare Ridge and Random Forest with persistence on the same months. The reported difference is the persistence metric minus the candidate metric, so a positive value favors the candidate. A paired circular moving-block bootstrap resamples 12-month blocks 10,000 times and reports the 2.5th and 97.5th percentiles. The blocks preserve short-range dependence better than resampling individual months. These intervals describe variation within this historical sequence; they do not turn one external period into evidence about every possible future policy regime.

RMSE and MAE are percentage points (multiply by 100 for basis points). R² is relative to the holdout mean and is not evidence of improvement over persistence. CV spans historical regimes unlike the holdout; compare models on the same dates. Impurity feature importance describes this fitted forest, not causal effects.

Random Forest predictions average training responses within terminal leaves and therefore cannot extrapolate beyond the training target range. The training target has a 0.63% floor, while the holdout reaches 0.07% in the post-2008 near-zero-rate regime. This mechanism explains the model's visibly elevated predictions during that period and is a central reason for its poor holdout performance.

## Economic rationale and scope

EFFR is a volume-weighted measure of overnight federal funds transactions, while the FOMC establishes the target rate or range and the Federal Reserve implements policy to keep overnight rates near it. EFFR consequently has a policy-guided, persistent structure: it is often stable between policy actions and changes in steps when the policy stance changes. A persistence forecast is therefore a substantive economic benchmark. An autoregressive model earns its complexity only if rate-history patterns improve on that benchmark across unseen periods.

This experiment measures short-run predictability in the realized monthly rate. A model of policy changes would need meeting-level timing, the prevailing target range, real-time macroeconomic vintages and market expectations available on the forecast date.

The available macro columns cannot establish what a forecaster knew in real time, while lagged EFFR has a clear timing rule. Ridge's mixed result across the two evaluation periods and Random Forest's consistent weakness both reinforce the value of a strong persistence baseline.

## Why the original pipeline was replaced

The earlier feature pipeline retained contemporaneous macro columns and ranked features using the full target series. Clipping, imputation and variance filtering also happened before the split. Those steps allowed information from the holdout period into training. The data-preparation scripts depended on source files that were not included and interpolated macro values across time.

The current package replaces that path with explicit inputs, past-only features and transformations fitted inside each training fold. Earlier results are not carried forward because they are not comparable to the new benchmark.

## Data provenance and limits

The development CSV has 752 consecutive monthly observations from July 1954 through February 2017. The external CSV adds 114 observations from March 2017 through August 2026. Both are EFFR-only snapshots of the Federal Reserve Board's H.15 monthly series `H15/H15/RIFSPFF_N.M`; `data/README.md` records the source, unit, frequency, retrieval dates, transformations, file hashes, usage terms, and official links. January and February 2017 were downloaded with the later package and matched against the end of the development file before being removed from the external period.

The earlier combined dataset attributed its inputs broadly to FRED, BLS and Kaggle without exact series identifiers, extraction dates, source URLs or vintage records. Those unused macro columns are no longer redistributed. Adding macroeconomic predictors in future work requires fresh, documented, vintage-aware sources rather than the earlier interpolated values.

The study remains univariate and monthly. It measures how fixed autoregressive rules behave across the observed rate cycles, not whether they can anticipate policy decisions. A policy-forecasting study would need a new design and separately documented, vintage-aware inputs.
