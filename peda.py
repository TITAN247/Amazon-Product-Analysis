# ==========================================
# AMAZON PRODUCT ANALYSIS (COMPLETE EDA)
# ==========================================

# ================================
# 1. IMPORT LIBRARIES
# ================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (8,5)
sns.set_style("whitegrid")

# ================================
# 2. LOAD DATASET
# ================================

df = pd.read_csv("amazon.csv")

# ================================
# 3. BASIC INFORMATION
# ================================

print("Shape :",df.shape)

print("\nColumns")
print(df.columns)

print("\nFirst Five Rows")
print(df.head())

print("\nLast Five Rows")
print(df.tail())

print("\nRandom Sample")
print(df.sample(5))

# ================================
# 4. DATASET INFORMATION
# ================================

print(df.info())

print(df.describe())

print(df.describe(include="object"))

# ================================
# 5. CHECK MISSING VALUES
# ================================

print(df.isnull().sum())

print(df.isnull().sum()/len(df)*100)

# ================================
# 6. DUPLICATE VALUES
# ================================

print("Duplicate Rows :",df.duplicated().sum())

# ================================
# 7. DATA CLEANING
# ================================

# Remove ₹ and comma

df["discounted_price"] = df["discounted_price"].str.replace("₹","",regex=False)
df["discounted_price"] = df["discounted_price"].str.replace(",","",regex=False)

df["actual_price"] = df["actual_price"].str.replace("₹","",regex=False)
df["actual_price"] = df["actual_price"].str.replace(",","",regex=False)

df["discount_percentage"] = df["discount_percentage"].str.replace("%","",regex=False)

# Convert Datatype

df["discounted_price"] = pd.to_numeric(df["discounted_price"])

df["actual_price"] = pd.to_numeric(df["actual_price"])

df["discount_percentage"] = pd.to_numeric(df["discount_percentage"])

df["rating"] = pd.to_numeric(df["rating"],errors="coerce")

df["rating_count"] = df["rating_count"].str.replace(",","",regex=False)

df["rating_count"] = pd.to_numeric(df["rating_count"],errors="coerce")

print(df.dtypes)

# ================================
# 8. MISSING VALUES AFTER CLEANING
# ================================

print(df.isnull().sum())

# ================================
# 9. OUTLIERS
# ================================

plt.figure(figsize=(10,5))

sns.boxplot(x=df["actual_price"])

plt.title("Actual Price Outliers")

plt.show()

plt.figure(figsize=(10,5))

sns.boxplot(x=df["discounted_price"])

plt.title("Discounted Price Outliers")

plt.show()

# ================================
# 10. IQR METHOD
# ================================

Q1 = df["actual_price"].quantile(0.25)

Q3 = df["actual_price"].quantile(0.75)

IQR = Q3-Q1

Lower = Q1-1.5*IQR

Upper = Q3+1.5*IQR

outliers = df[(df["actual_price"]<Lower) | (df["actual_price"]>Upper)]

print(outliers.shape)

# ================================
# 11. FEATURE ENGINEERING
# ================================

df["Discount_Amount"] = df["actual_price"]-df["discounted_price"]

# ================================
# 12. BINNING
# ================================

price_bins = [0,500,2000,5000,10000,100000]

price_labels = ["Budget",
                "Affordable",
                "Premium",
                "Luxury",
                "Ultra Luxury"]

df["Price_Category"] = pd.cut(df["actual_price"],
                              bins=price_bins,
                              labels=price_labels)

rating_bins=[0,2,3,4,5]

rating_labels=["Poor",
               "Average",
               "Good",
               "Excellent"]

df["Rating_Category"] = pd.cut(df["rating"],
                               bins=rating_bins,
                               labels=rating_labels)

# ================================
# 13. UNIVARIATE ANALYSIS
# ================================

sns.histplot(df["actual_price"],bins=30,kde=True)

plt.title("Actual Price Distribution")

plt.show()

sns.histplot(df["rating"],bins=20,kde=True)

plt.title("Rating Distribution")

plt.show()

sns.countplot(y=df["Price_Category"])

plt.title("Price Categories")

plt.show()

sns.countplot(y=df["Rating_Category"])

plt.title("Rating Categories")

plt.show()

# ================================
# 14. BIVARIATE ANALYSIS
# ================================

sns.scatterplot(data=df,
                x="actual_price",
                y="rating")

plt.title("Price vs Rating")

plt.show()

sns.boxplot(data=df,
            x="Price_Category",
            y="rating")

plt.xticks(rotation=20)

plt.show()

sns.barplot(data=df,
            x="Price_Category",
            y="discount_percentage")

plt.xticks(rotation=20)

plt.show()

# ================================
# 15. TOP CATEGORIES
# ================================

top_category = df["category"].value_counts().head(10)

print(top_category)

top_category.plot(kind="bar")

plt.title("Top Product Categories")

plt.show()

# ================================
# 16. GROUPBY
# ================================

print(df.groupby("Price_Category")["rating"].mean())

print(df.groupby("Price_Category")["discount_percentage"].mean())

# ================================
# 17. CORRELATION
# ================================

num = df[["actual_price",
          "discounted_price",
          "discount_percentage",
          "rating",
          "rating_count",
          "Discount_Amount"]]

sns.heatmap(num.corr(),
            annot=True,
            cmap="coolwarm")

plt.show()

# ================================
# 18. TOP RATED PRODUCTS
# ================================

top = df.sort_values("rating",
                     ascending=False)

print(top[["product_name",
           "rating"]].head(10))

# ================================
# 19. MOST EXPENSIVE PRODUCTS
# ================================

exp = df.sort_values("actual_price",
                     ascending=False)

print(exp[["product_name",
           "actual_price"]].head(10))

# ================================
# 20. HIGHEST DISCOUNT
# ================================

high_discount = df.sort_values("discount_percentage",
                               ascending=False)

print(high_discount[["product_name",
                     "discount_percentage"]].head(10))

# ================================
# 21. BUSINESS INSIGHTS
# ================================

print("Average Rating :",df["rating"].mean())

print("Average Price :",df["actual_price"].mean())

print("Average Discount :",df["discount_percentage"].mean())

print("Most Common Category")

print(df["category"].mode())