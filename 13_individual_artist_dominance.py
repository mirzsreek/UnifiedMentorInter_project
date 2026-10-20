
import pandas as pd

# 1. Load cleaned dataset
df = pd.read_csv("clean_data.csv")

# 2. Prepare artist and date columns
df["date"] = pd.to_datetime(df["date"])
df["artist"] = df["artist"].fillna("").str.strip()

# 3. Split collaborations into individual artists
df["individual_artist"] = df["artist"].str.split(r"\s*&\s*")

artist_df = df.explode("individual_artist")

artist_df["individual_artist"] = (
    artist_df["individual_artist"]
    .str.strip()
    .str.lower()
)

# Remove empty artist names
artist_df = artist_df[
    artist_df["individual_artist"] != ""
].copy()

# 4. Calculate individual artist appearances
artist_summary = artist_df.groupby(
    "individual_artist"
).agg(
    chart_appearances=("song", "size"),
    unique_songs=("song", "nunique"),
    first_chart_date=("date", "min"),
    last_chart_date=("date", "max")
).sort_values(
    "chart_appearances", ascending=False
)

print("\n===== TOP 15 INDIVIDUAL ARTISTS =====")
print(artist_summary.head(15).to_string())

# 5. Calculate monthly appearances for each artist
artist_df["month"] = (
    artist_df["date"].dt.to_period("M").astype(str)
)

monthly = (
    artist_df.groupby(["month", "individual_artist"])
    .size()
    .reset_index(name="chart_appearances")
)

# Save results
try:
    artist_summary.to_csv(
        "individual_artist_dominance.csv"
    )
    monthly.to_csv(
        "individual_artist_monthly_trends.csv",
        index=False
    )

    print("\nAnalysis completed successfully!")
    print("Saved individual_artist_dominance.csv")
    print("Saved individual_artist_monthly_trends.csv")

except OSError as error:
    print("\nAnalysis completed, but saving failed.")
    print("Storage error:", error)

