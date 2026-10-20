
import pandas as pd
from sklearn.metrics import mean_absolute_error

# 1. Load the prepared data
df = pd.read_csv("ml_prepared_data.csv")
df["date"] = pd.to_datetime(df["date"])

# 2. Remove rows without complete historical features
df = df.dropna(
    subset=[
        "position",
        "previous_position",
        "two_days_ago_position",
        "rank_movement",
        "target_position",
    ]
).copy()

# 3. Sort by date before splitting
df = df.sort_values("date")

# 4. Split chronologically: first 80% for training,
#    last 20% for testing
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

# 5. Baseline: predict tomorrow's rank using today's rank
predictions = test["position"]

# 6. Evaluate using Mean Absolute Error (MAE)
mae = mean_absolute_error(
    test["target_position"],
    predictions
)

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Baseline MAE:", round(mae, 3), "chart positions")
