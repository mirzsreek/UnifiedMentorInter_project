
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load the new dataset
df = pd.read_csv("ml_rolling_features.csv")
df["date"] = pd.to_datetime(df["date"])

# 2. Define features and target
features = [
    "position",
    "previous_position",
    "two_days_ago_position",
    "three_days_ago_position",
    "recent_mean_position",
    "recent_position_volatility",
]

X = df[features]
y = df["target_position"]

# 3. Split chronologically
dates = sorted(df["date"].unique())
split_date = dates[int(len(dates) * 0.8)]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("Training ends:", df.loc[train_mask, "date"].max())
print("Testing starts:", df.loc[test_mask, "date"].min())
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 4. Baseline: predict tomorrow's rank using today's rank
baseline_predictions = df.loc[test_mask, "position"]
baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

# 5. Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
model_mae = mean_absolute_error(y_test, predictions)

print("\nBaseline MAE:", round(baseline_mae, 3))
print("Rolling-feature Random Forest MAE:", round(model_mae, 3))

# 7. Inspect feature importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance.round(3))
