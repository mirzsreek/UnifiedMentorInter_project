
import pandas as pd

df = pd.read_csv("clean_data.csv")
df["date"] = pd.to_datetime(df["date"])

df["song_id"] = (
    df["song"].str.strip().str.lower()
    + " | "
    + df["artist"].str.strip().str.lower()
)

df = df.sort_values(["song_id", "date"]).copy()
grouped = df.groupby("song_id", sort=False)

# Historical positions only: exclude today's position
df["previous_position"] = grouped["position"].shift(1)
df["two_days_ago_position"] = grouped["position"].shift(2)
df["three_days_ago_position"] = grouped["position"].shift(3)

# Recent mean and volatility use the three observations
# BEFORE today, avoiding use of future information.
df["recent_mean_position"] = (
    df.groupby("song_id")["position"]
    .transform(lambda s: s.shift(1).rolling(3).mean())
)

df["recent_position_volatility"] = (
    df.groupby("song_id")["position"]
    .transform(lambda s: s.shift(1).rolling(3).std())
)

# Define tomorrow's position and date
df["target_position"] = grouped["position"].shift(-1)
df["next_date"] = grouped["date"].shift(-1)

df["previous_date"] = grouped["date"].shift(1)

# Require consecutive dates from previous day through tomorrow
valid = (
    ((df["date"] - df["previous_date"]).dt.days == 1)
    & ((df["next_date"] - df["date"]).dt.days == 1)
)

df = df[valid].copy()

features = [
    "position",
    "previous_position",
    "two_days_ago_position",
    "three_days_ago_position",
    "recent_mean_position",
    "recent_position_volatility",
]

ml_df = df[
    ["date", "song", "artist"] + features + ["target_position"]
].dropna().copy()

ml_df.to_csv("ml_rolling_features.csv", index=False)

print("New dataset shape:", ml_df.shape)
print("\nNew features:")
print(ml_df[features].head())
print("\nMissing values:")
print(ml_df[features + ["target_position"]].isnull().sum())
print("\nSaved as ml_rolling_features.csv")
