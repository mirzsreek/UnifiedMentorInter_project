
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. Load data
df = pd.read_csv("ml_rolling_features.csv")
df["date"] = pd.to_datetime(df["date"])

# 2. Create movement labels
# Positive movement = rank improves (e.g. 20 -> 15)
df["rank_movement"] = df["position"] - df["target_position"]

df["movement_category"] = df["rank_movement"].apply(
    lambda x: "Rising" if x >= 3
    else "Falling" if x <= -3
    else "Stable"
)

# 3. Features available at prediction time
features = [
    "position",
    "previous_position",
    "two_days_ago_position",
    "three_days_ago_position",
    "recent_mean_position",
    "recent_position_volatility",
]

X = df[features]
y = df["movement_category"]

# 4. Chronological split: never shuffle time-series data
dates = sorted(df["date"].unique())
split_date = dates[int(len(dates) * 0.8)]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]

print("Training ends:", df.loc[train_mask, "date"].max())
print("Testing starts:", df.loc[test_mask, "date"].min())
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Baseline: always predict the most common training category
majority_class = y_train.mode()[0]
baseline_predictions = [majority_class] * len(y_test)

print("\nMajority-class baseline:", majority_class)
print("Baseline accuracy:", round(
    accuracy_score(y_test, baseline_predictions), 4
))
print("Baseline balanced accuracy:", round(
    balanced_accuracy_score(y_test, baseline_predictions), 4
))

# 6. Train classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 7. Evaluate
print("\nRandom Forest accuracy:", round(
    accuracy_score(y_test, predictions), 4
))
print("Random Forest balanced accuracy:", round(
    balanced_accuracy_score(y_test, predictions), 4
))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["Rising", "Stable", "Falling"],
    zero_division=0,
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(pd.DataFrame(
    confusion_matrix(
        y_test, predictions,
        labels=["Rising", "Stable", "Falling"]
    ),
    index=["Actual Rising", "Actual Stable", "Actual Falling"],
    columns=["Pred Rising", "Pred Stable", "Pred Falling"],
))

# 8. Feature importance
print("\nFeature importance:")
print(pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False).round(3))

# 9. Save predictions for further analysis
results = df.loc[test_mask, ["date", "song", "artist", "position"]].copy()
results["actual_movement"] = y_test
results["predicted_movement"] = predictions
results.to_csv("movement_predictions.csv", index=False)

print("\nSaved predictions to movement_predictions.csv")
