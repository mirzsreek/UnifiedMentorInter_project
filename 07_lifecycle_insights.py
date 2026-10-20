
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("clean_data.csv")
df["date"] = pd.to_datetime(df["date"])

# Normalize song and artist names
df["song_clean"] = df["song"].str.strip().str.lower()
df["artist_clean"] = df["artist"].str.strip().str.lower()

# Create unique track identifier
df["track_id"] = df["song_clean"] + " | " + df["artist_clean"]

# Calculate lifecycle metrics
lifecycle = df.groupby("track_id").agg(
    entry_date=("date", "min"),
    exit_date=("date", "max"),
    total_days=("date", lambda x: (x.max() - x.min()).days + 1),
    peak_position=("position", "min"),
    chart_appearances=("date", "count")
)

# Calculate time to peak
peak_dates = (
    df[df["position"] == df.groupby("track_id")["position"].transform("min")]
    .groupby("track_id")["date"]
    .min()
)

lifecycle["time_to_peak"] = (
    peak_dates - lifecycle["entry_date"]
).dt.days


# Display complete results without truncation

columns_to_show = [
    "entry_date",
    "exit_date",
    "total_days",
    "peak_position",
    "chart_appearances",
    "time_to_peak"
]

print("\nTOP 10 SONGS BY OBSERVED CHART SPAN:")
print(
    lifecycle.sort_values("total_days", ascending=False)
    [columns_to_show].head(10).to_string()
)

print("\nTOP 10 SONGS BY BEST PEAK POSITION:")
print(
    lifecycle.sort_values(
        ["peak_position", "total_days"],
        ascending=[True, False]
    )[columns_to_show].head(10).to_string()
)

print("\nTOP 10 SONGS BY CHART APPEARANCES:")
print(
    lifecycle.sort_values("chart_appearances", ascending=False)
    [columns_to_show].head(10).to_string()
)


# 1. Longest observed chart spans
print("\nTOP 10 SONGS BY OBSERVED CHART SPAN:")
print(
    lifecycle.sort_values("total_days", ascending=False)
    .head(10)
)

# 2. Best chart performers
print("\nTOP 10 SONGS BY PEAK CHART POSITION:")
print(
    lifecycle.sort_values(
        ["peak_position", "total_days"],
        ascending=[True, False]
    ).head(10)
)

# 3. Most chart appearances
print("\nTOP 10 SONGS BY CHART APPEARANCES:")
print(
    lifecycle.sort_values("chart_appearances", ascending=False)
    .head(10)
)

# Save results
lifecycle.to_csv("song_lifecycle.csv")

print("\nLifecycle insights calculated successfully!")
