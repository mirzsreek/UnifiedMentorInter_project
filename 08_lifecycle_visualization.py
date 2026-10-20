
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("clean_data.csv")
df["date"] = pd.to_datetime(df["date"])

# Normalize song and artist names
df["song_clean"] = df["song"].str.strip().str.lower()
df["artist_clean"] = df["artist"].str.strip().str.lower()

# Create unique track identifiers
df["track_id"] = df["song_clean"] + " | " + df["artist_clean"]

# Calculate lifecycle metrics
lifecycle = df.groupby("track_id").agg(
    entry_date=("date", "min"),
    exit_date=("date", "max"),
    peak_position=("position", "min"),
    chart_appearances=("date", "count")
)

lifecycle["total_days"] = (
    lifecycle["exit_date"] - lifecycle["entry_date"]
).dt.days + 1

# ------------------------------------------
# CHART 1: Top 10 songs by chart appearances
# ------------------------------------------

top_appearances = lifecycle.nlargest(10, "chart_appearances")
labels = top_appearances.index.str.slice(0, 35)

plt.figure(figsize=(12, 7))
plt.barh(labels, top_appearances["chart_appearances"])
plt.xlabel("Number of chart appearances")
plt.ylabel("Song and artist")
plt.title("Top 10 Songs by Chart Appearances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top_chart_appearances.png", dpi=300)
plt.show()

# ------------------------------------------
# CHART 2: Lifecycle span vs chart appearances
# ------------------------------------------

plt.figure(figsize=(10, 6))
plt.scatter(
    lifecycle["total_days"],
    lifecycle["chart_appearances"],
    alpha=0.7
)
plt.xlabel("Observed chart span (calendar days)")
plt.ylabel("Number of chart appearances")
plt.title("Observed Chart Span vs Chart Appearances")
plt.tight_layout()
plt.savefig("lifecycle_span_vs_appearances.png", dpi=300)
plt.show()

# ------------------------------------------
# SAVE THE LIFECYCLE TABLE
# ------------------------------------------

lifecycle.to_csv("song_lifecycle.csv")

print("Both visualizations generated successfully!")
print("Saved: top_chart_appearances.png")
print("Saved: lifecycle_span_vs_appearances.png")
