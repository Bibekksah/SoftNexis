import pandas as pd
import numpy as np
import re


df = pd.read_csv("customers-100.csv")

print("Initial Shape:", df.shape)
df.info()
df.head()

print("Duplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("After Removing Duplicates:", df.shape)

for col in ["notes", "temp_id"]:
    if col in df.columns:
        df = df.drop(columns=[col])

if "cust_id" in df.columns:
    df = df.rename(columns={"cust_id": "customer_id"})

if "customer_id" in df.columns:
    cols = ["customer_id"] + [c for c in df.columns if c != "customer_id"]
    df = df[cols]

df.head()

if "customer_id" in df.columns:
    df = df.dropna(subset=["customer_id"])

if "age" in df.columns:
    df["age"].fillna(df["age"].median(), inplace=True)

if "email" in df.columns:
    df["email"].fillna("unknown@example.com", inplace=True)

df.isna().sum()


if "signup_date" in df.columns:
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

if "purchase_amount" in df.columns:
    df["purchase_amount"] = (
        df["purchase_amount"]
        .astype(str)
        .str.replace("[\$,]", "", regex=True)
    )
    df["purchase_amount"] = pd.to_numeric(df["purchase_amount"], errors="coerce")

if "age" in df.columns:
    df["age"] = df["age"].astype(int)

df.head()

if "name" in df.columns:
    df["name"] = df["name"].str.lower().str.strip().str.title()


if "region" in df.columns:
    df["region"] = df["region"].replace({
        "west": "Western",
        "W": "Western",
        "south": "Southern",
        "NORTH": "Northern"
    })

def clean_phone(phone):
    digits = re.sub(r"\D", "", str(phone))
    if len(digits) == 10:
        return "+91-" + digits
    elif len(digits) == 12 and digits.startswith("91"):
        return "+" + digits[:2] + "-" + digits[2:]
    return phone

if "phone" in df.columns:
    df["phone"] = df["phone"].apply(clean_phone)

df.head()

df.to_csv("customers_cleaned.csv", index=False)
print("Data Cleaning Completed! Saved as customers_cleaned.csv")
