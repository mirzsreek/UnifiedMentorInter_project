
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("ml_rolling_features.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

features = [
    "position",
    "previous_position",
    "two_days_ago_position",
    "three_days_ago_position",
    "recent_mean_position",
    "recent_position_volatility",
]

# Use the same chronological split
dates = sorted(df["date"].unique())
split_date = dates[int(len(dates) * 0.8)]

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(train[features], train["target_position"])

# Predictions on identical test rows
test["baseline_prediction"] = test["position"]
test["rf_prediction"] = model.predict(test[features])

# Absolute prediction errors
test["baseline_error"] = (
    test["target_position"] - test["baseline_prediction"]
).abs()

test["rf_error"] = (
    test["target_position"] - test["rf_prediction"]
).abs()

print("Baseline MAE:", round(test["baseline_error"].mean(), 3))
print("Random Forest MAE:", round(test["rf_error"].mean(), 3))

# Which model performed better?
test["winner"] = test.apply(
    lambda row: (
        "Random Forest"
        if row["rf_error"] < row["baseline_error"]
        else "Baseline"
        if row["baseline_error"] < row["rf_error"]
        else "Tie"
    ),
    axis=1,
)

print("\nWinner counts:")
print(test["winner"].value_counts())

# Compare performance by current chart position
test["rank_group"] = pd.cut(
    test["position"],
    bins=[0, 10, 20, 30, 40, 50],
    labels=["1-10", "11-20", "21-30", "31-40", "41-50"],
)

print("\nMAE by current chart position:")
print(
    test.groupby("rank_group", observed=False)[
        ["baseline_error", "rf_error"]
    ].mean().round(3)
)

# Show the 10 largest baseline errors
print("\nLargest baseline errors:")
print(
    test.nlargest(10, "baseline_error")[
        [
            "date", "song", "position", "target_position",
            "baseline_prediction", "rf_prediction",
            "baseline_error", "rf_error"
        ]
    ].to_string(index=False)
)

test.to_csv("model_error_comparison.csv", index=False)
print("\nSaved as model_error_comparison.csv")
