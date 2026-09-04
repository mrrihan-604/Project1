# ============================================================
# HOUSE PRICE PREDICTION PROJECT
# Task: TA-008 - Exploratory Data Analysis (EDA)
# Objective:
# Analyze relationships between property features and House Price
# ============================================================


# ------------------------------------------------------------
# STEP 1: Import required libraries
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# STEP 2: Load the dataset
# ------------------------------------------------------------

# Read the CSV file
df = pd.read_csv("house_price_prediction_dataset.csv")


# ------------------------------------------------------------
# STEP 3: Display basic information about the dataset
# ------------------------------------------------------------

print("\n========== DATASET PREVIEW ==========")

# Display first 5 records
print(df.head())


print("\n========== DATASET SHAPE ==========")

# Display number of rows and columns
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


print("\n========== COLUMN NAMES ==========")

# Display all column names
print(df.columns.tolist())


print("\n========== DATA TYPES ==========")

# Display data type of each column
print(df.dtypes)


# ------------------------------------------------------------
# STEP 4: Statistical summary
# ------------------------------------------------------------

print("\n========== STATISTICAL SUMMARY ==========")

# Describe numerical columns
print(df.describe())


# ------------------------------------------------------------
# STEP 5: Check missing values
# ------------------------------------------------------------

print("\n========== MISSING VALUES ==========")

# Count missing values in each column
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 6: Check duplicate records
# ------------------------------------------------------------

print("\n========== DUPLICATE RECORDS ==========")

# Count duplicate rows
print("Number of duplicate rows:", df.duplicated().sum())


# ------------------------------------------------------------
# STEP 7: Analyze numerical features
# ------------------------------------------------------------

print("\n========== CORRELATION WITH PRICE ==========")

# Select numerical columns
numeric_columns = [
    "Square_Feet",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "Year_Built",
    "Price"
]

# Calculate correlation matrix
correlation_matrix = df[numeric_columns].corr()

print(correlation_matrix)


# ------------------------------------------------------------
# STEP 8: Display correlation of each numerical feature
#         with Price
# ------------------------------------------------------------

print("\n========== CORRELATION WITH HOUSE PRICE ==========")

price_correlation = correlation_matrix["Price"].sort_values(
    ascending=False
)

print(price_correlation)


# ------------------------------------------------------------
# STEP 9: Relationship between Square Feet and Price
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Square_Feet"],
    df["Price"],
    alpha=0.6
)

plt.xlabel("Square Feet")
plt.ylabel("House Price")
plt.title("Relationship Between Square Feet and House Price")

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# STEP 10: Relationship between Bedrooms and Price
# ------------------------------------------------------------

bedroom_price = df.groupby("Bedrooms")["Price"].mean()

print("\n========== AVERAGE PRICE BY NUMBER OF BEDROOMS ==========")

print(bedroom_price)


plt.figure(figsize=(8, 5))

plt.bar(
    bedroom_price.index,
    bedroom_price.values
)

plt.xlabel("Number of Bedrooms")
plt.ylabel("Average House Price")
plt.title("Average House Price by Number of Bedrooms")

plt.grid(axis="y")

plt.show()


# ------------------------------------------------------------
# STEP 11: Relationship between Bathrooms and Price
# ------------------------------------------------------------

bathroom_price = df.groupby("Bathrooms")["Price"].mean()

print("\n========== AVERAGE PRICE BY NUMBER OF BATHROOMS ==========")

print(bathroom_price)


plt.figure(figsize=(8, 5))

plt.bar(
    bathroom_price.index,
    bathroom_price.values
)

plt.xlabel("Number of Bathrooms")
plt.ylabel("Average House Price")
plt.title("Average House Price by Number of Bathrooms")

plt.grid(axis="y")

plt.show()


# ------------------------------------------------------------
# STEP 12: Relationship between Floors and Price
# ------------------------------------------------------------

floor_price = df.groupby("Floors")["Price"].mean()

print("\n========== AVERAGE PRICE BY NUMBER OF FLOORS ==========")

print(floor_price)


plt.figure(figsize=(8, 5))

plt.bar(
    floor_price.index,
    floor_price.values
)

plt.xlabel("Number of Floors")
plt.ylabel("Average House Price")
plt.title("Average House Price by Number of Floors")

plt.grid(axis="y")

plt.show()


# ------------------------------------------------------------
# STEP 13: Relationship between Year Built and Price
# ------------------------------------------------------------

year_price = df.groupby("Year_Built")["Price"].mean()

print("\n========== AVERAGE PRICE BY YEAR BUILT ==========")

print(year_price)


plt.figure(figsize=(10, 5))

plt.scatter(
    year_price.index,
    year_price.values,
    alpha=0.7
)

plt.xlabel("Year Built")
plt.ylabel("Average House Price")
plt.title("Relationship Between Year Built and Average House Price")

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# STEP 14: Relationship between Location and Price
# ------------------------------------------------------------

location_price = df.groupby("Location")["Price"].agg(
    ["mean", "min", "max", "count"]
)

print("\n========== PRICE ANALYSIS BY LOCATION ==========")

print(location_price)


plt.figure(figsize=(8, 5))

location_price["mean"].plot(
    kind="bar"
)

plt.xlabel("Location")
plt.ylabel("Average House Price")
plt.title("Average House Price by Location")

plt.xticks(rotation=0)

plt.grid(axis="y")

plt.show()


# ------------------------------------------------------------
# STEP 15: Relationship between Condition and Price
# ------------------------------------------------------------

condition_price = df.groupby("Condition")["Price"].agg(
    ["mean", "min", "max", "count"]
)

print("\n========== PRICE ANALYSIS BY CONDITION ==========")

print(condition_price)


plt.figure(figsize=(8, 5))

condition_price["mean"].plot(
    kind="bar"
)

plt.xlabel("House Condition")
plt.ylabel("Average House Price")
plt.title("Average House Price by House Condition")

plt.xticks(rotation=0)

plt.grid(axis="y")

plt.show()


# ------------------------------------------------------------
# STEP 16: Location + Condition relationship with Price
# ------------------------------------------------------------

location_condition_price = df.groupby(
    ["Location", "Condition"]
)["Price"].mean()

print("\n========== AVERAGE PRICE BY LOCATION AND CONDITION ==========")

print(location_condition_price)


# ------------------------------------------------------------
# STEP 17: Identify highest and lowest priced houses
# ------------------------------------------------------------

print("\n========== HIGHEST PRICED HOUSE ==========")

highest_price_house = df.loc[df["Price"].idxmax()]

print(highest_price_house)


print("\n========== LOWEST PRICED HOUSE ==========")

lowest_price_house = df.loc[df["Price"].idxmin()]

print(lowest_price_house)


# ------------------------------------------------------------
# STEP 18: Identify largest and smallest houses
# ------------------------------------------------------------

print("\n========== LARGEST HOUSE ==========")

largest_house = df.loc[df["Square_Feet"].idxmax()]

print(largest_house)


print("\n========== SMALLEST HOUSE ==========")

smallest_house = df.loc[df["Square_Feet"].idxmin()]

print(smallest_house)


# ------------------------------------------------------------
# STEP 19: Final EDA observations
# ------------------------------------------------------------

print("\n==============================================")
print("           EDA ANALYSIS COMPLETED")
print("==============================================")


print("\nImportant numerical features related to Price:")

# Remove Price itself from the list
print(
    price_correlation.drop("Price")
)


print("\nAverage Price by Location:")

print(location_price["mean"])


print("\nAverage Price by Condition:")

print(condition_price["mean"])


print("\n========== END OF TA-008 ==========")