
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load the prepared dataset
df = pd.read_csv("ml_prepared_data.csv")
df["date"] = pd.to_datetime(df["date"])

# 2. Remove rows with missing features
df = df.dropna(
    subset=[
        "position",
        "previous_position",
        "two_days_ago_position",
        "rank_movement",
        "target_position",
    ]
).copy()

# 3. Define features and target
features = [
    "position",
    "previous_position",
    "two_days_ago_position",
    "rank_movement",
]

X = df[features]
y = df["target_position"]

# 4. Split by date: earlier dates train, later dates test
dates = sorted(df["date"].unique())
split_index = int(len(dates) * 0.8)
split_date = dates[split_index]

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

# 5. Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# 6. Predict tomorrow's chart position
predictions = model.predict(X_test)

# 7. Evaluate against the actual positions
mae = mean_absolute_error(y_test, predictions)
baseline_mae = mean_absolute_error(
    y_test, df.loc[test_mask, "position"]
)

print("\nBaseline MAE:", round(baseline_mae, 3))
print("Random Forest MAE:", round(mae, 3))

# 8. See a few predictions
results = pd.DataFrame({
    "Actual position": y_test.iloc[:10].values,
    "Predicted position": predictions[:10].round(2),
})

print("\nSample predictions:")
print(results.to_string(index=False))

# 9. See which features the model relied on most
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance.round(3))
