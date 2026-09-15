import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("Housing.csv")


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print("Number of duplicate rows:", df.duplicated().sum())


# ============================================================
# BASIC DATA ANALYSIS
# ============================================================

print("\n========== MOST EXPENSIVE HOUSES ==========")
print(df.sort_values("price", ascending=False).head())

print("\n========== CHEAPEST HOUSES ==========")
print(df.sort_values("price", ascending=True).head())

print("\n========== MOST EXPENSIVE HOUSE ==========")
print(df.loc[df["price"].idxmax()])

print("\n========== CHEAPEST HOUSE ==========")
print(df.loc[df["price"].idxmin()])

print("\n========== LARGEST HOUSE ==========")
print(df.loc[df["area"].idxmax()])

print("\n========== SMALLEST HOUSE ==========")
print(df.loc[df["area"].idxmin()])

print("\n========== AVERAGE PRICE BY NUMBER OF BEDROOMS ==========")
print(df.groupby("bedrooms")["price"].mean())


# ============================================================
# FEATURE CREATION
# ============================================================

df["price_per_sqft"] = df["price"] / df["area"]

print("\n========== PRICE PER SQUARE FOOT ==========")
print(df[["area", "price", "price_per_sqft"]].head())


df["total_rooms"] = df["bedrooms"] + df["bathrooms"]

print("\n========== TOTAL ROOMS ==========")
print(df[["bedrooms", "bathrooms", "total_rooms"]].head())


# ============================================================
# UNIVARIATE ANALYSIS
# ============================================================

# Page 1

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(
    data=df,
    x="price",
    bins=30,
    kde=True,
    ax=axes[0, 0]
)

axes[0, 0].set_title("Distribution of House Prices")
axes[0, 0].set_xlabel("Price")
axes[0, 0].set_ylabel("Number of Houses")


sns.histplot(
    data=df,
    x="area",
    bins=30,
    kde=True,
    ax=axes[0, 1]
)

axes[0, 1].set_title("Distribution of House Areas")
axes[0, 1].set_xlabel("Area")
axes[0, 1].set_ylabel("Number of Houses")


sns.countplot(
    data=df,
    x="bedrooms",
    ax=axes[1, 0]
)

axes[1, 0].set_title("Number of Bedrooms")
axes[1, 0].set_xlabel("Bedrooms")
axes[1, 0].set_ylabel("Number of Houses")


sns.countplot(
    data=df,
    x="bathrooms",
    ax=axes[1, 1]
)

axes[1, 1].set_title("Number of Bathrooms")
axes[1, 1].set_xlabel("Bathrooms")
axes[1, 1].set_ylabel("Number of Houses")


fig.suptitle("Univariate Analysis - Page 1", fontsize=16)
plt.tight_layout()
plt.show()


# Page 2

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.countplot(
    data=df,
    x="parking",
    ax=axes[0]
)

axes[0].set_title("Parking Spaces")
axes[0].set_xlabel("Parking Spaces")
axes[0].set_ylabel("Number of Houses")


sns.countplot(
    data=df,
    x="furnishingstatus",
    ax=axes[1]
)

axes[1].set_title("Furnishing Status")
axes[1].set_xlabel("Furnishing Status")
axes[1].set_ylabel("Number of Houses")


fig.suptitle("Univariate Analysis - Page 2", fontsize=16)
plt.tight_layout()
plt.show()


# ============================================================
# BIVARIATE ANALYSIS
# ============================================================

# Page 3

fig, axes = plt.subplots(2, 2, figsize=(14, 10))


sns.scatterplot(
    data=df,
    x="area",
    y="price",
    ax=axes[0, 0]
)

axes[0, 0].set_title("Area vs Price")
axes[0, 0].set_xlabel("Area")
axes[0, 0].set_ylabel("Price")


sns.boxplot(
    data=df,
    x="bedrooms",
    y="price",
    ax=axes[0, 1]
)

axes[0, 1].set_title("Price by Number of Bedrooms")
axes[0, 1].set_xlabel("Bedrooms")
axes[0, 1].set_ylabel("Price")


sns.boxplot(
    data=df,
    x="bathrooms",
    y="price",
    ax=axes[1, 0]
)

axes[1, 0].set_title("Price by Number of Bathrooms")
axes[1, 0].set_xlabel("Bathrooms")
axes[1, 0].set_ylabel("Price")


sns.boxplot(
    data=df,
    x="parking",
    y="price",
    ax=axes[1, 1]
)

axes[1, 1].set_title("Price by Parking Spaces")
axes[1, 1].set_xlabel("Parking Spaces")
axes[1, 1].set_ylabel("Price")


fig.suptitle("Bivariate Analysis - Page 1", fontsize=16)
plt.tight_layout()
plt.show()


# Page 4

plt.figure(figsize=(10, 7))

sns.boxplot(
    data=df,
    x="furnishingstatus",
    y="price"
)

plt.title("Price by Furnishing Status")
plt.xlabel("Furnishing Status")
plt.ylabel("Price")

plt.tight_layout()
plt.show()


# ============================================================
# MULTIVARIATE ANALYSIS
# ============================================================

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="area",
    y="price",
    hue="bedrooms"
)

plt.title("Area vs Price by Number of Bedrooms")
plt.xlabel("Area")
plt.ylabel("Price")

plt.tight_layout()
plt.show()


# ============================================================
# CORRELATION
# ============================================================

numeric_columns = [
    "area",
    "bedrooms",
    "bathrooms",
    "parking",
    "price"
]

correlation = df[numeric_columns].corr()

print("\n========== CORRELATION ==========")
print(correlation)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# OUTLIER DETECTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="price"
)

plt.title("Price Outlier Detection")
plt.xlabel("Price")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="area"
)

plt.title("Area Outlier Detection")
plt.xlabel("Area")

plt.tight_layout()
plt.show()


# ============================================================
# IQR OUTLIER DETECTION - PRICE
# ============================================================

q1 = df["price"].quantile(0.25)
q3 = df["price"].quantile(0.75)

iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

print("\n========== PRICE IQR ==========")
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

price_outliers = df[
    (df["price"] < lower_limit) |
    (df["price"] > upper_limit)
]

print("\n========== PRICE OUTLIERS ==========")
print(price_outliers)


# ============================================================
# IQR OUTLIER DETECTION - AREA
# ============================================================

q1 = df["area"].quantile(0.25)
q3 = df["area"].quantile(0.75)

iqr = q3 - q1

lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

print("\n========== AREA IQR ==========")
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

area_outliers = df[
    (df["area"] < lower_limit) |
    (df["area"] > upper_limit)
]

print("\n========== AREA OUTLIERS ==========")
print(area_outliers)


# ============================================================
# BINNING
# ============================================================

df["area_category"] = pd.cut(
    df["area"],
    bins=[0, 1500, 3000, float("inf")],
    labels=["Small", "Medium", "Large"]
)

print("\n========== AREA CATEGORIES ==========")
print(df[["area", "area_category"]].head(10))


# ============================================================
# ENCODING CATEGORICAL VARIABLES
# ============================================================

print("\n========== CATEGORICAL COLUMNS ==========")
print(df.select_dtypes(include=["object"]).columns)


df["mainroad"] = df["mainroad"].map({
    "Yes": 1,
    "No": 0
})

df["guestroom"] = df["guestroom"].map({
    "Yes": 1,
    "No": 0
})

df["basement"] = df["basement"].map({
    "Yes": 1,
    "No": 0
})

df["hotwaterheating"] = df["hotwaterheating"].map({
    "Yes": 1,
    "No": 0
})

df["airconditioning"] = df["airconditioning"].map({
    "Yes": 1,
    "No": 0
})

df["furnishingstatus"] = df["furnishingstatus"].map({
    "Furnished": 2,
    "Semi-Furnished": 1,
    "Unfurnished": 0
})


print("\n========== DATA TYPES AFTER ENCODING ==========")
print(df.dtypes)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

# price_per_sqft is removed because it uses the target variable.
# area_category is removed because it is still categorical.

X = df.drop(
    columns=[
        "price",
        "price_per_sqft",
        "area_category"
    ]
)

y = df["price"]


print("\n========== FEATURES (X) ==========")
print(X.head())

print("\n========== TARGET (y) ==========")
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n========== TRAIN TEST SPLIT ==========")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("\n========== SCALING ==========")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

print("\n========== FIRST 5 SCALED TRAINING ROWS ==========")
print(X_train_scaled[:5])