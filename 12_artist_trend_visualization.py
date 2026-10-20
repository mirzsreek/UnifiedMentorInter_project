
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load cleaned data
df = pd.read_csv("clean_data.csv")

# 2. Prepare columns
df["date"] = pd.to_datetime(df["date"])
df["artist_clean"] = (
    df["artist"].str.strip().str.lower()
)
df["month"] = df["date"].dt.to_period("M")

# 3. Select the five leading artists
top_artists = [
    "saiko",
    "myke towers",
    "quevedo",
    "karol g",
    "bad bunny"
]

# 4. Count chart appearances per artist per month
monthly = (
    df[df["artist_clean"].isin(top_artists)]
    .groupby(["month", "artist_clean"])
    .size()
    .unstack(fill_value=0)
)

# Include all months in the observed period
all_months = pd.period_range(
    df["date"].min().to_period("M"),
    df["date"].max().to_period("M"),
    freq="M"
)

monthly = monthly.reindex(all_months, fill_value=0)
monthly.index = monthly.index.astype(str)

# Ensure all five artists appear as columns
monthly = monthly.reindex(columns=top_artists, fill_value=0)

# 5. Print monthly results
print("\n===== MONTHLY ARTIST APPEARANCES =====")
print(monthly.to_string())

# 6. Plot monthly trends
plt.figure(figsize=(13, 7))

for artist in top_artists:
    plt.plot(
        monthly.index,
        monthly[artist],
        marker="o",
        linewidth=2,
        label=artist.title()
    )

plt.xlabel("Month")
plt.ylabel("Number of chart appearances")
plt.title("Monthly Chart Presence of the Top 5 Artists")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 7. Save the chart
try:
    plt.savefig(
        "artist_monthly_trends.png",
        dpi=300,
        bbox_inches="tight"
    )
    print("\nChart saved as artist_monthly_trends.png")
except OSError as error:
    print("\nCould not save the chart:", error)

plt.show()

# 8. Save the monthly data
try:
    monthly.to_csv("artist_monthly_trends.csv")
    print("Monthly data saved as artist_monthly_trends.csv")
except OSError as error:
    print("Could not save the monthly data:", error)

print("\nArtist trend analysis completed!")

