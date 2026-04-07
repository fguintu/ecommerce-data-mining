"""
02_fpgrowth.py  –  Segment-conditioned FP-Growth Pattern Mining
================================================================
CS 570 Data Mining  |  Online Retail II dataset

What this script does:
  1. Loads the cleaned dataset produced by 00_preprocess.py
  2. Builds a basket matrix: one row per invoice, one column per product,
     True/False for whether that product appeared in that invoice
  3. Runs FP-Growth on the FULL dataset to find frequent itemsets
  4. Derives association rules (confidence, lift) from those itemsets
  5. Saves results to outputs/

Run:
    python scripts/02_fpgrowth.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # no display needed – saves to file
import seaborn as sns
from pathlib import Path
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUT_DIR    = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

CLEANED    = DATA_DIR / "online_retail_cleaned.parquet"

# ── hyper-parameters (tune here if needed) ───────────────────────────────────
MIN_SUPPORT    = 0.02   # item-set must appear in ≥2 % of invoices
MIN_CONFIDENCE = 0.30   # rule A→B must be right ≥30 % of the time
MIN_LIFT       = 1.0    # lift > 1  means A and B co-occur more than by chance

# =============================================================================
# STEP 1  –  Load data
# =============================================================================
print("Loading cleaned data …")
df = pd.read_parquet(CLEANED)

# We need at minimum: Invoice (basket id) and Description (item name)
# Drop rows where either is missing
df = df.dropna(subset=["Invoice", "Description"])
df["Description"] = df["Description"].str.strip().str.upper()

print(f"  Rows loaded     : {len(df):,}")
print(f"  Unique invoices : {df['Invoice'].nunique():,}")
print(f"  Unique items    : {df['Description'].nunique():,}")

# =============================================================================
# STEP 2  –  Build basket (transaction) list
# =============================================================================
print("\nBuilding transaction baskets …")

# Each invoice → list of items bought in that order
baskets = (
    df.groupby("Invoice")["Description"]
    .apply(list)
    .tolist()
)

print(f"  Total baskets : {len(baskets):,}")

# TransactionEncoder turns the list-of-lists into a True/False matrix
te = TransactionEncoder()
te_array = te.fit_transform(baskets)
basket_df = pd.DataFrame(te_array, columns=te.columns_)

print(f"  Basket matrix : {basket_df.shape[0]:,} rows × {basket_df.shape[1]:,} columns")

# =============================================================================
# STEP 3  –  Run FP-Growth to find frequent itemsets
# =============================================================================
print(f"\nRunning FP-Growth  (min_support={MIN_SUPPORT}) …")

frequent_itemsets = fpgrowth(
    basket_df,
    min_support=MIN_SUPPORT,
    use_colnames=True,   # keep actual product names (not indices)
)

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

print(f"  Frequent itemsets found : {len(frequent_itemsets):,}")
print(frequent_itemsets["length"].value_counts().sort_index()
      .rename("count").rename_axis("itemset size").to_string())

# =============================================================================
# STEP 4  –  Derive association rules
# =============================================================================
print(f"\nDeriving association rules  (min_confidence={MIN_CONFIDENCE}) …")

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE,
)

# Add a readable columns
rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

# Keep only rules where lift > 1 (genuine co-occurrence, not coincidence)
rules = rules[rules["lift"] >= MIN_LIFT].copy()
rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

print(f"  Rules after lift filter  : {len(rules):,}")
print("\nTop 10 rules by lift:")
print(rules[["antecedents_str", "consequents_str",
             "support", "confidence", "lift"]].head(10).to_string(index=False))

# =============================================================================
# STEP 5  –  Save results
# =============================================================================
itemsets_path = OUT_DIR / "frequent_itemsets.csv"
rules_path    = OUT_DIR / "association_rules.csv"

frequent_itemsets_out = frequent_itemsets.copy()
frequent_itemsets_out["itemsets"] = frequent_itemsets_out["itemsets"].apply(
    lambda x: ", ".join(sorted(x))
)
frequent_itemsets_out.to_csv(itemsets_path, index=False)
print(f"\nFrequent itemsets saved → {itemsets_path}")

rules_out = rules[["antecedents_str", "consequents_str",
                    "support", "confidence", "lift",
                    "leverage", "conviction"]].copy()
rules_out.to_csv(rules_path, index=False)
print(f"Association rules saved  → {rules_path}")

# =============================================================================
# STEP 6  –  Visualisations
# =============================================================================
print("\nGenerating plots …")

# -- 6a: Support vs Confidence scatter coloured by lift ----------------------
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(
    rules["support"], rules["confidence"],
    c=rules["lift"], cmap="YlOrRd", alpha=0.7, edgecolors="none", s=40
)
plt.colorbar(sc, ax=ax, label="Lift")
ax.set_xlabel("Support")
ax.set_ylabel("Confidence")
ax.set_title("Association Rules – Support vs Confidence (coloured by Lift)")
plt.tight_layout()
plot1 = OUT_DIR / "rules_scatter.png"
fig.savefig(plot1, dpi=150)
plt.close(fig)
print(f"  Scatter plot saved → {plot1}")

# -- 6b: Distribution of itemset sizes ---------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
size_counts = frequent_itemsets["length"].value_counts().sort_index()
ax.bar(size_counts.index, size_counts.values, color="steelblue", edgecolor="white")
ax.set_xlabel("Itemset Size")
ax.set_ylabel("Count")
ax.set_title("Distribution of Frequent Itemset Sizes")
plt.tight_layout()
plot2 = OUT_DIR / "itemset_size_dist.png"
fig.savefig(plot2, dpi=150)
plt.close(fig)
print(f"  Itemset size plot saved → {plot2}")

# -- 6c: Top 15 most frequent single items -----------------------------------
single_items = (
    frequent_itemsets[frequent_itemsets["length"] == 1]
    .copy()
)
single_items["item"] = single_items["itemsets"].apply(lambda x: list(x)[0])
single_items = single_items.sort_values("support", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(single_items["item"][::-1], single_items["support"][::-1], color="coral")
ax.set_xlabel("Support")
ax.set_title("Top 15 Most Frequent Individual Items")
plt.tight_layout()
plot3 = OUT_DIR / "top_items.png"
fig.savefig(plot3, dpi=150)
plt.close(fig)
print(f"  Top items plot saved    → {plot3}")

print("\n✅  02_fpgrowth.py complete!")
print(f"   Outputs in: {OUT_DIR.resolve()}")
