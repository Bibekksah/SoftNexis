import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,5)

df = pd.read_csv(r"B:\internship offer letter\Task-2\olist_customers_dataset.csv")

print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
df.head()


print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isna().sum())

print("\nSummary Stats (numeric columns):")
print(df.describe())

print("\nUnique values per column:")
print(df.nunique())



# distribution of customer_state (if present)
if "customer_state" in df.columns:
    plt.figure(figsize=(12,6))
    sns.countplot(x="customer_state", data=df, order=df["customer_state"].value_counts().index)
    plt.title("Customer Count by State")
    plt.xticks(rotation=90)
    plt.show()

# Check unique customers
if "customer_unique_id" in df.columns:
    print("Unique Customers:", df["customer_unique_id"].nunique())
    print("Duplicate customer_id entries:", df.duplicated(subset="customer_unique_id").sum())



# Customer_id vs State
if "customer_state" in df.columns and "customer_id" in df.columns:
    state_counts = df.groupby("customer_state")["customer_id"].nunique().sort_values(ascending=False)
    state_counts.plot(kind="bar", figsize=(12,6), title="Unique Customers per State")
    plt.show()



# Plot distributions for numeric columns
num_cols = df.select_dtypes(include=["int64","float64"]).columns
for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()


if len(num_cols) > 1:
    plt.figure(figsize=(8,6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

# Faceted Analysis distribution by state vs unique_id existence)
if "customer_state" in df.columns:
    g = sns.FacetGrid(df, col="customer_state", col_wrap=4, height=3)
    g.map_dataframe(sns.histplot, x="customer_unique_id", bins=20)
    plt.show()

# Pairplot for multivariate analysis 
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
if len(num_cols) >= 2:
    sns.pairplot(df[num_cols].sample(min(500, len(df))), diag_kind="kde")
    plt.show()


outlier_report = {}
for col in num_cols:
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outlier_report[col] = (z_scores > 3).sum()

print("Outlier counts per numeric column (Z-score > 3):")
print(outlier_report)

# IQR method
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR))).sum()
    print(f"{col}: {outliers} outliers (IQR method)")
