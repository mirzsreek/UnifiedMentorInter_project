
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("clean_data.csv")

# Prepare columns
df["date"] = pd.to_datetime(df["date"])
df["song_clean"] = df["song"].str.strip().str.lower()
df["artist_clean"] = df["artist"].str.strip().str.lower()
df["track_id"] = df["song_clean"] + " | " + df["artist_clean"]

# Sort each song's chart history by date
df = df.sort_values(["track_id", "date"])

# Calculate movement between consecutive calendar days
df["previous_date"] = df.groupby("track_id")["date"].shift(1)
df["previous_position"] = df.groupby("track_id")["position"].shift(1)

df["days_between"] = (
    df["date"] - df["previous_date"]
).dt.days

# Compare rankings only when the song appeared on consecutive days
consecutive = df[df["days_between"] == 1].copy()

# Positive movement means the song improved its ranking
consecutive["rank_change"] = (
    consecutive["previous_position"] - consecutive["position"]
)

# Find the biggest one-day rises and falls
fastest_risers = consecutive.nlargest(10, "rank_change")
biggest_fallers = consecutive.nsmallest(10, "rank_change")

print("\n===== TOP 10 FASTEST ONE-DAY RISERS =====")
print(
    fastest_risers[
        ["date", "song", "artist", "previous_position",
         "position", "rank_change"]
    ].to_string(index=False)
)

print("\n===== TOP 10 BIGGEST ONE-DAY FALLERS =====")
print(
    biggest_fallers[
        ["date", "song", "artist", "previous_position",
         "position", "rank_change"]
    ].to_string(index=False)
)

# Measure ranking stability for songs with enough observations
stability = consecutive.groupby("track_id").agg(
    song=("song", "first"),
    artist=("artist", "first"),
    movement_observations=("rank_change", "count"),
    average_absolute_change=("rank_change", lambda x: x.abs().mean()),
    average_position=("position", "mean")
)

# Require at least 10 consecutive-day comparisons
stable_songs = (
    stability[stability["movement_observations"] >= 10]
    .sort_values("average_absolute_change")
    .head(10)
)

print("\n===== TOP 10 MOST STABLE SONGS =====")
print(stable_songs.to_string())

# Save results if storage permits
try:
    fastest_risers.to_csv("fastest_risers.csv", index=False)
    biggest_fallers.to_csv("biggest_fallers.csv", index=False)
    stable_songs.to_csv("most_stable_songs.csv")
    print("\nAnalysis complete. Result CSV files saved.")
except OSError as error:
    print("\nAnalysis completed, but files could not be saved.")
    print("Storage error:", error)

