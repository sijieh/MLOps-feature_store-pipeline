import pandas as pd
import numpy as np
from feast import FeatureStore
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

# === CONFIG ===
feature_services = ["athletes_service_v1", "athletes_service_v2"]
target_column = "total_lift"
store = FeatureStore(repo_path=".")

# === HYPERPARAMETER SETTINGS ===
hyperparams_list = [
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
]

# === COLLECT RESULTS ===
all_results = []

for feature_service in feature_services:
    entity_df = pd.read_parquet("data/v1_features.parquet")[["athlete_id", "event_timestamp", target_column]].copy()
    df = store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service(feature_service),
    ).to_df()
    df = df.dropna()
    X = df.drop(columns=["athlete_id", "event_timestamp", target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for hp in hyperparams_list:
        label = f"{feature_service}\n{hp['n_estimators']} trees"
        tracker = EmissionsTracker(project_name=f"{label}")
        tracker.start()

        model = RandomForestRegressor(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        emissions = tracker.stop()

        all_results.append({
            "label": label,
            "feature_service": feature_service,
            "n_estimators": hp["n_estimators"],
            "max_depth": hp["max_depth"],
            "mse": mse,
            "r2": r2,
            "emissions_kg": emissions,
            "y_test": y_test,
            "y_pred": y_pred,
        })

# === SUMMARY DATAFRAME ===
results_df = pd.DataFrame([{
    "label": r["label"],
    "mse": r["mse"],
    "r2": r["r2"],
    "emissions_kg": r["emissions_kg"]
} for r in all_results])

print("\n=== Experiment Summary ===")
print(results_df)

# === PLOT 1: 4 SUBPLOTS OF PREDICTIONS ===
fig1, axs1 = plt.subplots(2, 2, figsize=(12, 10))
axs1 = axs1.flatten()

for i, res in enumerate(all_results):
    ax = axs1[i]
    ax.scatter(res["y_test"], res["y_pred"], alpha=0.5)
    ax.plot([res["y_test"].min(), res["y_test"].max()],
            [res["y_test"].min(), res["y_test"].max()],
            "--r")
    ax.set_xlabel("Actual Total Lift")
    ax.set_ylabel("Predicted Total Lift")
    ax.set_title(f"{res['label']}\nMSE: {res['mse']:.2f}, R²: {res['r2']:.2f}")

plt.tight_layout()
plt.show()

# === PLOT 2: BARPLOTS OF MSE & EMISSIONS ===
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))

# Barplot 1: MSE
axs2[0].bar(results_df["label"], results_df["mse"], color='skyblue')
axs2[0].set_title("MSE Comparison")
axs2[0].set_ylabel("Mean Squared Error")
axs2[0].tick_params(axis='x', rotation=45)

# Barplot 2: Emissions
axs2[1].bar(results_df["label"], results_df["emissions_kg"], color='salmon')
axs2[1].set_title("CO₂ Emissions Comparison")
axs2[1].set_ylabel("Emissions (kg CO₂)")
axs2[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()