
import pandas as pd

# Load the dataset
df = pd.read_csv("ml_rolling_features.csv")
df["date"] = pd.to_datetime(df["date"])

# Calculate next-day movement
# Positive = rising; negative = falling
df["rank_movement"] = df["position"] - df["target_position"]

# Categorize movement
df["movement_category"] = df["rank_movement"].apply(
    lambda movement:
        "Rising" if movement >= 3
        else "Falling" if movement <= -3
        else "Stable"
)

# Overall movement distribution
print("Movement category counts:")
print(df["movement_category"].value_counts())

print("\nMovement category percentages:")
print(
    (df["movement_category"].value_counts(normalize=True) * 100)
    .round(2)
)

# Average movement by current chart position
df["rank_group"] = pd.cut(
    df["position"],
    bins=[0, 10, 20, 30, 40, 50],
    labels=["1-10", "11-20", "21-30", "31-40", "41-50"]
)

print("\nMovement by current rank group:")
print(pd.crosstab(df["rank_group"], df["movement_category"]))

# Save results
df.to_csv("rank_movement_analysis.csv", index=False)
print("\nSaved as rank_movement_analysis.csv")
