
import pandas as pd

# 1. Load the cleaned dataset
df = pd.read_csv("clean_data.csv")

# 2. Prepare the data
df["date"] = pd.to_datetime(df["date"])
df["artist_clean"] = (
    df["artist"].str.strip().str.lower()
)

# Extract month
df["month"] = df["date"].dt.to_period("M").astype(str)

# 3. Calculate overall artist dominance
artist_summary = df.groupby("artist_clean").agg(
    chart_appearances=("song", "count"),
    unique_songs=("song", "nunique"),
    first_chart_date=("date", "min"),
    last_chart_date=("date", "max")
).sort_values(
    "chart_appearances", ascending=False
)

print("\n===== TOP 10 ARTISTS OVERALL =====")
print(artist_summary.head(10).to_string())

# 4. Calculate monthly artist dominance
monthly_artist = (
    df.groupby(["month", "artist_clean"])
    .size()
    .reset_index(name="chart_appearances")
)

# Rank artists within each month
monthly_artist["monthly_rank"] = (
    monthly_artist.groupby("month")["chart_appearances"]
    .rank(method="dense", ascending=False)
)

monthly_top = monthly_artist[
    monthly_artist["monthly_rank"] == 1
].sort_values("month")

print("\n===== LEADING ARTISTS BY MONTH =====")
print(monthly_top.to_string(index=False))

# 5. Count distinct songs per artist and month
monthly_songs = (
    df.groupby(["month", "artist_clean"])["song"]
    .nunique()
    .reset_index(name="unique_songs")
)

# 6. Save the results
try:
    artist_summary.to_csv("artist_dominance_summary.csv")
    monthly_artist.to_csv("monthly_artist_dominance.csv", index=False)
    monthly_top.to_csv("monthly_leading_artists.csv", index=False)
    monthly_songs.to_csv("monthly_artist_songs.csv", index=False)

    print("\nAnalysis completed successfully!")
    print("Saved:")
    print("- artist_dominance_summary.csv")
    print("- monthly_artist_dominance.csv")
    print("- monthly_leading_artists.csv")
    print("- monthly_artist_songs.csv")

except OSError as error:
    print("\nAnalysis completed, but saving failed.")
    print("Storage error:", error)
