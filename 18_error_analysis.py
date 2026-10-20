
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("ml_prepared_data.csv")
df["date"] = pd.to_datetime(df["date"])

features = [
    "position",
    "previous_position",
    "two_days_ago_position",
    "rank_movement",
]

df = df.dropna(subset=features + ["target_position"]).copy()
df = df.sort_values("date")

dates = sorted(df["date"].unique())
split_date = dates[int(len(dates) * 0.8)]

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(train[features], train["target_position"])

test["prediction"] = model.predict(test[features])
test["absolute_error"] = (
    test["target_position"] - test["prediction"]
).abs()

print("Overall MAE:", round(test["absolute_error"].mean(), 3))

print("\nLargest prediction errors:")
print(
    test.nlargest(10, "absolute_error")[
        ["date", "song", "position", "target_position",
         "prediction", "absolute_error"]
    ].to_string(index=False)
)

print("\nMAE by today's chart position:")
test["position_group"] = pd.cut(
    test["position"],
    bins=[0, 10, 20, 30, 40, 50],
    labels=["1-10", "11-20", "21-30", "31-40", "41-50"],
)

print(
    test.groupby("position_group", observed=True)["absolute_error"]
    .mean()
    .round(3)
)
