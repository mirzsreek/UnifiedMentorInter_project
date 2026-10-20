import pandas as pd

df = pd.read_csv("Atlantic_Spain.csv")

# Find exact duplicate rows
duplicates = df[df.duplicated(keep=False)]

print("NUMBER OF DUPLICATE ROWS:")
print(len(duplicates))

print("\nDUPLICATE ROWS:")
print(duplicates.to_string(index=False))