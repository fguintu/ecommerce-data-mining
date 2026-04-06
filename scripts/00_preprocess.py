"""
00_preprocess.py
----------------
Cleans the raw Online Retail II xlsx and saves a parquet to data/.
Run from anywhere: paths are resolved relative to this file.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT     = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "online_retail_II.xlsx"
OUT_PATH = ROOT / "data" / "online_retail_cleaned.parquet"

# ---------- Load ----------
df_2009 = pd.read_excel(RAW_PATH, sheet_name="Year 2009-2010")
df_2010 = pd.read_excel(RAW_PATH, sheet_name="Year 2010-2011")
df = pd.concat([df_2009, df_2010], ignore_index=True)

print(f"Raw rows: {len(df):,}")

# ---------- Standardize column names ----------
df.columns = [c.strip().replace(" ", "") for c in df.columns]

# ---------- Type coercion ----------
df["Invoice"]     = df["Invoice"].astype(str)
df["StockCode"]   = df["StockCode"].astype(str).str.strip().str.upper()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"]    = pd.to_numeric(df["Quantity"], errors="coerce")
df["Price"]       = pd.to_numeric(df["Price"], errors="coerce")

# ---------- Filtering ----------
df = df[~df["Invoice"].str.startswith("C")]

NON_PRODUCT_CODES = {"POST", "DOT", "M", "BANK CHARGES", "CRUK", "PADS", "D", "C2"}
df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]
df = df[df["StockCode"].str.contains(r"\d", regex=True, na=False)]

df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
df = df.dropna(subset=["InvoiceDate", "Description"])
df["Description"] = df["Description"].str.strip()

# ---------- Derived columns ----------
df["LineRevenue"] = df["Quantity"] * df["Price"]

# ---------- Deduplicate ----------
df = df.drop_duplicates()

# ---------- Sanity checks ----------
print(f"Cleaned rows:      {len(df):,}")
print(f"Unique invoices:   {df['Invoice'].nunique():,}")
print(f"Unique customers:  {df['CustomerID'].nunique():,}")
print(f"Unique products:   {df['StockCode'].nunique():,}")
print(f"Date range:        {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
print(f"Countries:         {df['Country'].nunique()}")

# ---------- Save ----------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print(f"Saved -> {OUT_PATH.relative_to(ROOT)}")
