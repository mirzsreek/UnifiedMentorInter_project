import pandas as pd

df = pd.read_csv("Atlantic_Spain.csv")

print("Original dataset shape:")
print(df.shape)

# Check the problematic date
march1 = df[df["date"] == "01-03-2025"]

print("\nRecords on 01-03-2025:")
print(len(march1))
# Keep only the second 50 records for 01-03-2025
march1_clean = march1.iloc[50:100]

print("\nRecords kept for 01-03-2025:")
print(len(march1_clean))
# Remove all records for 01-03-2025
df_without_march1 = df[df["date"] != "01-03-2025"]

# Add the clean 50 records back
clean_df = pd.concat([df_without_march1, march1_clean], ignore_index=True)

print("\nCleaned dataset shape:")
print(clean_df.shape)
# Convert date column to proper date format
clean_df["date"] = pd.to_datetime(clean_df["date"], format="%d-%m-%Y")

print("\nDate data type:")
print(clean_df["date"].dtype)
# Sort the data by date and chart position
clean_df = clean_df.sort_values(["date", "position"])

print("\nFirst 5 rows after sorting:")
print(clean_df.head())
# Check how many songs we have for each date
songs_per_day = clean_df.groupby("date").size()

print("\nSongs per day:")
print(songs_per_day.value_counts())

# Check number of unique dates
print("\nUnique dates:")
print(clean_df["date"].nunique())

# Check duplicate rows
print("\nDuplicate rows:")
print(clean_df.duplicated().sum())
# Save the cleaned dataset
clean_df.to_csv("clean_data.csv", index=False)

print("\nCleaned dataset saved successfully!")