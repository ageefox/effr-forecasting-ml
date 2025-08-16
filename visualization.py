# for plotting and visualizing trends
import matplotlib.pyplot as plt

# === Print metrics table ===
print("\n=== Model Performance Metrics ===")
print(metrics_df.sort_values("RMSE"))

# training the model
# Best model
# training the model
best_model = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
# training the model
print(f"\nBest model by RMSE: {best_model}")

# making predictions
# Find prediction column
pred_col = None
for col in preds_frame.columns:
# training the model
    if col.lower().startswith(best_model.lower().split("(")[0].lower()):
        pred_col = col
        break

# making predictions
# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(preds_frame["Actual"], preds_frame[pred_col], alpha=0.6)
# visualizing results
plt.plot([preds_frame["Actual"].min(), preds_frame["Actual"].max()],
         [preds_frame["Actual"].min(), preds_frame["Actual"].max()],
# making predictions
         'r--', label="Perfect prediction")
plt.xlabel("Actual EFFR")
# training the model
plt.ylabel(f"Predicted EFFR ({best_model})")
# training the model
plt.title(f"Actual vs Predicted EFFR — {best_model}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# working with dataframes
import pandas as pd
# for plotting and visualizing trends
import matplotlib.pyplot as plt

# Load feature importances
file_path = "rf_tuned_feature_importance.csv"
# loading the dataset
importances = pd.read_csv(file_path, index_col=0).squeeze()  # converts to Series

# Plot top 25
top_n = 25
top_features = importances.sort_values(ascending=True).tail(top_n)  # ascending for horizontal bar

# Plot
plt.figure(figsize=(8, 10))
# visualizing results
top_features.plot(kind='barh', color='skyblue')
plt.title(f"Top {top_n} Feature Importances (Random Forest Tuned)", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

