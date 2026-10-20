
import pandas as pd

# 1. Load the cleaned dataset
df = pd.read_csv("clean_data.csv")
df["date"] = pd.to_datetime(df["date"])

# 2. Identify each song consistently
df["song_id"] = (
    df["song"].str.strip().str.lower()
    + " | "
    + df["artist"].str.strip().str.lower()
)

# 3. Sort each song's history chronologically
df = df.sort_values(["song_id", "date"]).copy()

# 4. Create features from past information
grouped = df.groupby("song_id", sort=False)

df["previous_position"] = grouped["position"].shift(1)
df["two_days_ago_position"] = grouped["position"].shift(2)

# Positive movement means the song improved its rank
df["rank_movement"] = (
    df["previous_position"] - df["two_days_ago_position"]
)

# 5. Define tomorrow's position as the target
df["target_position"] = grouped["position"].shift(-1)
df["next_date"] = grouped["date"].shift(-1)

# Keep only consecutive daily observations:
# yesterday -> today -> tomorrow
df["previous_date"] = grouped["date"].shift(1)

df = df[
    ((df["date"] - df["previous_date"]).dt.days == 1)
    & ((df["next_date"] - df["date"]).dt.days == 1)
].copy()

# 6. Select the features and target we need
columns = [
    "date",
    "song",
    "artist",
    "position",
    "previous_position",
    "two_days_ago_position",
    "rank_movement",
    "target_position",
]

ml_df = df[columns].copy()

# 7. Validate and save
print("ML dataset shape:", ml_df.shape)
print("\nSample rows:")
print(ml_df.head(10).to_string(index=False))

print("\nMissing values:")
print(ml_df.isnull().sum())

ml_df.to_csv("ml_prepared_data.csv", index=False)
print("\nSaved as ml_prepared_data.csv")
