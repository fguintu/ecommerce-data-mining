import pandas as pd
import numpy as np

# ---------- Load ----------
# Online Retail II ships as an .xlsx with two sheets (2009-2010 and 2010-2011)
RAW_PATH = "online_retail_II.xlsx"

df_2009 = pd.read_excel(RAW_PATH, sheet_name="Year 2009-2010")
df_2010 = pd.read_excel(RAW_PATH, sheet_name="Year 2010-2011")
df = pd.concat([df_2009, df_2010], ignore_index=True)

print(f"Raw rows: {len(df):,}")

# ---------- Standardize column names ----------
df.columns = [c.strip().replace(" ", "") for c in df.columns]
# Expected columns now: Invoice, StockCode, Description, Quantity,
# InvoiceDate, Price, CustomerID, Country

# ---------- Type coercion ----------
df["Invoice"] = df["Invoice"].astype(str)
df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# ---------- Filtering ----------
# 1. Drop cancellations (Invoice starts with "C")
df = df[~df["Invoice"].str.startswith("C")]

# 2. Drop non-product StockCodes
NON_PRODUCT_CODES = {"POST", "DOT", "M", "BANK CHARGES", "CRUK", "PADS", "D", "C2"}
df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]

# Also drop pure-letter codes (admin/test entries) — keep codes with at least one digit
df = df[df["StockCode"].str.contains(r"\d", regex=True, na=False)]

# 3. Drop null Customer IDs
df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

# 4. Drop non-positive quantity and price
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

# 5. Drop null invoice dates and descriptions
df = df.dropna(subset=["InvoiceDate", "Description"])
df["Description"] = df["Description"].str.strip()

# ---------- Derived columns ----------
df["LineRevenue"] = df["Quantity"] * df["Price"]

# ---------- Deduplicate ----------
df = df.drop_duplicates()

# ---------- Sanity checks ----------
print(f"Cleaned rows: {len(df):,}")
print(f"Unique invoices: {df['Invoice'].nunique():,}")
print(f"Unique customers: {df['CustomerID'].nunique():,}")
print(f"Unique products: {df['StockCode'].nunique():,}")
print(f"Date range: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
print(f"Countries: {df['Country'].nunique()}")

# ---------- Save ----------
df.to_parquet("online_retail_cleaned.parquet", index=False)
print("Saved → online_retail_cleaned.parquet")