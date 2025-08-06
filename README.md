# Forecasting the Effective Federal Funds Rate (EFFR)

This project explores the use of machine learning models to forecast the Effective Federal Funds Rate using a range of macroeconomic indicators. The aim is to capture interest rate trends, improve predictive accuracy, and evaluate model performance in both stable and extreme economic conditions.

## Dataset

The dataset combines historical data from:
- U.S. Bureau of Labor Statistics
- Federal Reserve Economic Data (FRED)
- Kaggle data sources

**Date range:** July 1954 – February 2017  
**Features include:**
- Federal Funds Target Rate
- Effective Federal Funds Rate
- Real GDP Percent Change
- Unemployment Rate
- Inflation Rate
- Consumer Price Index (CPI)

## Feature Engineering

We created lagged features, rolling averages, interaction terms, and technical indicators to enrich the dataset. These included:
- Lag values of EFFR and macro indicators
- Rolling mean and volatility
- Month and quarter encoding
- Derived macro features like inflation gaps

## Models Used

- Linear Regression
- Polynomial Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Each model was tuned and evaluated using RMSE and visualized through actual vs. predicted plots.

## Key Insights

- Random Forest performed best overall, capturing mid-range EFFR values well.
- The model struggled with predicting extreme rate changes due to limited historical precedent.
- Richer features improved performance, but historical uniqueness of macro trends remains a challenge.

## Project Structure
effr-forecasting-ml/
├── data/                  # Cleaned dataset(s)
├── notebooks/             # Jupyter notebooks for EDA & modeling
├── src/                   # Python scripts for preprocessing and modeling
├── results/               # Visualizations and output graphs
├── requirements.txt       # Python package dependencies
├── .gitignore             # Ignored files
└── README.md              # Project summary and instructions
