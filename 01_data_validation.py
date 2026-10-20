import pandas as pd

df = pd.read_csv("Atlantic_Spain.csv")

# Get only 01-03-2025
march1 = df[df["date"] == "01-03-2025"].copy()

print("TOTAL ROWS:", len(march1))

# Show first 50 rows
first_50 = march1.iloc[:50]

# Show second 50 rows
second_50 = march1.iloc[50:100]

print("\nFIRST 50 SONGS:")
print(first_50[["position", "song", "artist"]].to_string(index=False))

print("\nSECOND 50 SONGS:")
print(second_50[["position", "song", "artist"]].to_string(index=False))

# Compare songs
first_songs = set(first_50["song"])
second_songs = set(second_50["song"])

print("\nSONGS ONLY IN FIRST 50:")
print(first_songs - second_songs)

print("\nSONGS ONLY IN SECOND 50:")
print(second_songs - first_songs)

print("\nNUMBER OF COMMON SONGS:")
print(len(first_songs & second_songs))