# Forecast design and audit

## Information available at forecast time

The target is the CSV's monthly EFFR observation for month t. The forecast origin is the first date after the completed month-t−1 observation is available. This is not a forecast made before month t begins. Inputs are EFFR lags 1, 2, 3, 6 and 12, trailing means and standard deviations over 3, 6 and 12 months ending at t-1, and the change from t-2 to t-1. The first 12 rows are excluded as a fixed warm-up.

The final 20% of usable dates are held out. Five expanding training folds select hyperparameters using mean fold RMSE. Standardization is fitted separately inside each Ridge training fold. Random Forest needs no scaling. There is no fitted imputation, clipping or feature selection. The lowest training-CV RMSE selects among Ridge, Random Forest and persistence; holdout scores do not determine this choice.

Models are fitted once on the training block. Each later one-month prediction receives the preceding observed EFFR, including earlier holdout observations. That is valid for rolling one-step evaluation, but it is not a recursive forecast of the whole holdout from a single historical date. The same information rule applies within validation folds. A gap is unnecessary for this one-step target because the last training label is already observed when the first validation prediction is made.

RMSE and MAE are percentage points (multiply by 100 for basis points). R² is relative to the holdout mean and is not evidence of improvement over persistence. CV spans historical regimes unlike the holdout; compare models on the same dates. Impurity feature importance describes this fitted forest, not causal effects.

Random Forest predictions average training responses within terminal leaves and therefore cannot extrapolate beyond the training target range. The training target has a 0.63% floor, while the holdout reaches 0.07% in the post-2008 near-zero-rate regime. This mechanism explains the model's visibly elevated predictions during that period and is a central reason for its poor holdout performance.

## Economic rationale and scope

EFFR is a volume-weighted measure of overnight federal funds transactions, while the FOMC establishes the target rate or range and the Federal Reserve implements policy to keep overnight rates near it. EFFR consequently has a policy-guided, persistent structure: it is often stable between policy actions and changes in steps when the policy stance changes. A persistence forecast is therefore a substantive economic benchmark. An autoregressive model earns its complexity only if rate-history patterns improve on that benchmark across unseen periods.

This experiment measures short-run predictability in the realized monthly rate. It does not model the FOMC's reaction to inflation, employment or growth, and it does not claim to forecast policy decisions. A monthly average may include observations from both sides of an FOMC decision, so a model of policy changes should use meeting-level timing instead. A credible extension would predict the next target-range decision or future target midpoint using the prevailing range, meeting calendar, real-time macroeconomic vintages and market expectations available on the forecast date.

This narrower scope is deliberate. The available macro columns cannot establish what a forecaster knew in real time, while lagged EFFR has an explicit publication and timing rule. The resulting negative finding—that Ridge and Random Forest do not clearly beat persistence—is economically plausible and is the primary benchmark result, not a failed attempt to obtain a more complex winner.

## Findings in the original implementation

At base commit `45f9f94`, feature engineering retained unshifted raw macro columns. Feature ranking used all labels, including future test labels. Clipping quantiles, imputation and variance filtering were fitted before splitting. Chronological splitting later in training did not undo those leaks. Multiple training programs ran sequentially on import; file locations depended on the current directory. Plotting referenced undefined global variables. Data preparation required raw CSVs absent from the repository and linearly interpolated macro values using future endpoints.

The replacement package avoids those production paths. The original notebook and removed scripts remain available in Git history at `45f9f94`. Previous results are not comparable to the new benchmark and are not carried forward as validated findings.

## Data provenance and limits

The checked-in CSV has 752 consecutive monthly observations from July 1954 through February 2017. It is an EFFR-only snapshot of the Federal Reserve Board's H.15 monthly series `H15/H15/RIFSPFF_N.M`; `data/README.md` records the source, unit, frequency, retrieval date, transformation, usage terms and official links. The data are monthly averages of daily figures and are capped at February 2017 to preserve the study period.

The earlier combined dataset attributed its inputs broadly to FRED, BLS and Kaggle without exact series identifiers, extraction dates, source URLs or vintage records. Those unused macro columns are no longer redistributed. Adding macroeconomic predictors in future work requires fresh, documented, vintage-aware sources rather than the earlier interpolated values.

This is a reproducible historical portfolio study, not a validated live forecasting service. The holdout predates recent rate cycles and has been inspected in prior project work, so it is not a pristine never-seen research test set. Stronger external validation would require newly sourced observations, documented release timing and vintage-aware macro inputs before expanding the model scope. No such data is silently substituted here.
