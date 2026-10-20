
import pandas as pd

# Load dataset
df = pd.read_csv("clean_data.csv")

# Clean data
df["date"] = pd.to_datetime(df["date"])
df["song_clean"] = df["song"].str.strip().str.lower()
df["artist_clean"] = df["artist"].str.strip().str.lower()
df["track_id"] = df["song_clean"] + " | " + df["artist_clean"]

# Check required columns
required = {"popularity", "position", "track_id", "song", "artist"}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Convert numeric columns safely
df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
df["position"] = pd.to_numeric(df["position"], errors="coerce")
df = df.dropna(subset=["popularity", "position"])

# Summarize each song
song_summary = df.groupby("track_id").agg(
    song=("song", "first"),
    artist=("artist", "first"),
    average_popularity=("popularity", "mean"),
    best_chart_position=("position", "min"),
    average_chart_position=("position", "mean"),
    chart_appearances=("date", "count")
)

# Correlation across songs
correlation = song_summary[
    ["average_popularity", "best_chart_position",
     "average_chart_position", "chart_appearances"]
].corr()

print("\n===== SONG-LEVEL CORRELATION MATRIX =====")
print(correlation.round(3))

print("\n===== TOP 10 SONGS BY AVERAGE POPULARITY =====")
print(
    song_summary.nlargest(10, "average_popularity")[
        ["song", "artist", "average_popularity",
         "best_chart_position", "chart_appearances"]
    ].to_string(index=False)
)

# Save results
try:
    song_summary.to_csv("song_popularity_summary.csv")
    correlation.to_csv("popularity_chart_correlations.csv")
    print("\nAnalysis completed. Result CSV files saved.")
except OSError as error:
    print("\nAnalysis completed, but files could not be saved.")
    print("Storage error:", error)

