
import pandas as pd

# ==========================================
# 1. LOAD THE CLEANED DATASET
# ==========================================

df = pd.read_csv("clean_data.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

print("Dataset shape:")
print(df.shape)


# ==========================================
# 2. CLEAN SONG AND ARTIST NAMES
# ==========================================

df["song_clean"] = df["song"].str.strip().str.lower()
df["artist_clean"] = df["artist"].str.strip().str.lower()

# Create a unique identifier for each song and artist combination
df["track_id"] = df["song_clean"] + " | " + df["artist_clean"]


# ==========================================
# 3. SORT DATA BY DATE
# ==========================================

df = df.sort_values("date")


# ==========================================
# 4. CALCULATE ENTRY AND EXIT DATES
# ==========================================

entry_dates = df.groupby("track_id")["date"].min()
exit_dates = df.groupby("track_id")["date"].max()

lifecycle = pd.DataFrame({
    "entry_date": entry_dates,
    "exit_date": exit_dates
})

# Calculate inclusive calendar-day span
lifecycle["total_days"] = (
    lifecycle["exit_date"] - lifecycle["entry_date"]
).dt.days + 1


# ==========================================
# 5. CALCULATE PEAK CHART POSITION
# ==========================================

# Lower position numbers indicate better chart rankings
peak_positions = df.groupby("track_id")["position"].min()

lifecycle["peak_position"] = peak_positions


# ==========================================
# 6. CALCULATE TIME TO PEAK
# ==========================================

# Find the first date each track achieved its best position
peak_rows = df.merge(
    lifecycle[["peak_position"]],
    left_on="track_id",
    right_index=True
)

peak_rows = peak_rows[
    peak_rows["position"] == peak_rows["peak_position"]
]

first_peak_dates = peak_rows.groupby("track_id")["date"].min()

lifecycle["time_to_peak"] = (
    first_peak_dates - lifecycle["entry_date"]
).dt.days


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\nFirst 10 song entry dates:")
print(lifecycle["entry_date"].head(10))

print("\nFirst 10 song exit dates:")
print(lifecycle["exit_date"].head(10))

print("\nFirst 10 lifecycle records:")
print(lifecycle.head(10))

print("\nLifecycle table shape:")
print(lifecycle.shape)


# ==========================================
# 8. SAVE LIFECYCLE DATASET
# ==========================================

lifecycle.to_csv("song_lifecycle.csv")

print("\nLifecycle analysis saved successfully!")
print("Output file: song_lifecycle.csv")
