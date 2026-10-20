import pandas as pd

# Load cleaned dataset
df = pd.read_csv("clean_data.csv")

# Basic information
print("Dataset shape:")
print(df.shape)

print("\nBasic statistics:")
print(df.describe())
print("\nPopularity range:")
print("Minimum:", df["popularity"].min())
print("Maximum:", df["popularity"].max())
print("Average:", df["popularity"].mean())
print("\nTop 10 artists by number of chart appearances:")

artist_counts = df["artist"].value_counts()

print(artist_counts.head(10))
print("\nSaiko variations:")
print(df[df["artist"].str.lower() == "saiko"]["artist"].value_counts())
# Normalize artist names
df["artist_clean"] = df["artist"].str.strip().str.lower()

print("\nSaiko after normalization:")
print(df[df["artist_clean"] == "saiko"]["artist_clean"].value_counts())
print("\nExplicit vs Clean songs:")

print(df["is_explicit"].value_counts())
print("\nAlbum type distribution:")
print(df["album_type"].value_counts())
print("\nSong duration:")
print("Shortest:", df["duration_ms"].min(), "ms")
print("Longest:", df["duration_ms"].max(), "ms")
print("Average:", df["duration_ms"].mean(), "ms")
# Convert song duration from milliseconds to minutes
df["duration_minutes"] = df["duration_ms"] / 60000

print("\nAverage song duration:")
print(df["duration_minutes"].mean(), "minutes")
print("\nAlbum size:")
print("Smallest:", df["total_tracks"].min())
print("Largest:", df["total_tracks"].max())
print("Average:", df["total_tracks"].mean())
print("\nUnique songs:")
print(df["song"].nunique())

print("\nUnique artists:")
print(df["artist_clean"].nunique())