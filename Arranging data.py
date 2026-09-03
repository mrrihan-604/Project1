import pandas as pd

# Load the dataset
df = pd.read_csv("house_price_prediction_dataset.csv")

# Rename the neighborhood column to 'Location'
df.rename(
    columns={"Neighborhood": "Location", "Neighbourhood": "Location"},
    inplace=True,
)

# Save the updated data
df.to_csv("house_price_prediction_dataset.csv", index=False)

print("Done! Renamed successfully.")

import pandas as pd

# 1. Load the dataset
df = pd.read_csv("house_price_prediction_dataset.csv")

# 2. Rename the neighborhood column to 'Location'
df.rename(
    columns={"Neighborhood": "Location", "Neighbourhood": "Location"},
    inplace=True,
)

# 3. Drop Has-garage and Has-Pool
# (Includes common spelling variations just in case of underscores or case differences)
columns_to_drop = [
    "Has-garage",
    "Has-Pool",
    "Has_Garage",
    "Has_Pool",
    "Has_garage",
    "Has_pool",
]
df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# 4. Save the updated data
df.to_csv("house_price_prediction_dataset.csv", index=False)

print("Remaining columns:")
print(df.columns.tolist())