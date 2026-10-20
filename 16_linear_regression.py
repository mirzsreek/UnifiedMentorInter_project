
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Load prepared data
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

# 3. Select input features and target
features = [
    "position",
    "previous_position",
    "two_days_ago_position",
    "rank_movement",
]

X = df[features]
y = df["target_position"]


# 4. Split by calendar date, not by individual rows
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

# 5. Baseline prediction: tomorrow = today's position
baseline_predictions = df.loc[test_mask, "position"]

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

# 6. Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict and evaluate
predictions = model.predict(X_test)

linear_mae = mean_absolute_error(y_test, predictions)

print("\nBaseline MAE:", round(baseline_mae, 3))
print("Linear Regression MAE:", round(linear_mae, 3))

