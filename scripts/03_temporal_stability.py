"""
03_temporal_stability.py  –  Quarterly Temporal Stability of Association Rules
===============================================================================
CS 570 Data Mining  |  Online Retail II dataset

What this script does:
  1. Loads the cleaned dataset produced by 00_preprocess.py
  2. Splits all transactions into 4 quarters based on InvoiceDate
  3. Runs FP-Growth independently on each quarter
  4. Computes a "stability score" for every itemset:
        how many of the 4 quarters contain that itemset?
        4/4 = perfectly stable year-round
        1/4 = seasonal / trend item
  5. Saves per-quarter rules + a master stability summary
  6. Plots quarterly trends so you can see what shifts over time

Run:
    python scripts/03_temporal_stability.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

CLEANED  = DATA_DIR / "online_retail_cleaned.parquet"

# ── hyper-parameters ─────────────────────────────────────────────────────────
# Smaller segments have fewer transactions so we lower the support a little
MIN_SUPPORT_QUARTERLY = 0.015   # 1.5 % per quarter
MIN_CONFIDENCE        = 0.30
MIN_LIFT              = 1.0

QUARTER_NAMES = {1: "Q1 (Jan–Mar)", 2: "Q2 (Apr–Jun)",
                 3: "Q3 (Jul–Sep)", 4: "Q4 (Oct–Dec)"}

# =============================================================================
# STEP 1  –  Load & prepare data
# =============================================================================
print("Loading cleaned data …")
df = pd.read_parquet(CLEANED)
df = df.dropna(subset=["Invoice", "Description", "InvoiceDate"])
df["Description"]  = df["Description"].str.strip().str.upper()
df["InvoiceDate"]  = pd.to_datetime(df["InvoiceDate"])
df["Quarter"]      = df["InvoiceDate"].dt.quarter
df["Year"]         = df["InvoiceDate"].dt.year

print(f"  Rows loaded     : {len(df):,}")
print(f"  Date range      : {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}")
print(f"  Unique invoices : {df['Invoice'].nunique():,}")
print(f"  Quarters present: {sorted(df['Quarter'].unique())}")

# =============================================================================
# HELPER – build basket matrix from a dataframe slice
# =============================================================================
def build_basket_matrix(slice_df):
    """Given a subset of df, return the True/False basket DataFrame."""
    baskets = (
        slice_df.groupby("Invoice")["Description"]
        .apply(list)
        .tolist()
    )
    te = TransactionEncoder()
    te_array = te.fit_transform(baskets)
    return pd.DataFrame(te_array, columns=te.columns_), len(baskets)


# =============================================================================
# STEP 2  –  Run FP-Growth per quarter
# =============================================================================
quarter_itemsets = {}   # q → frequent_itemsets DataFrame
quarter_rules    = {}   # q → rules DataFrame
quarter_stats    = []   # summary row per quarter

print("\nRunning FP-Growth per quarter …")

for q in [1, 2, 3, 4]:
    q_df = df[df["Quarter"] == q]
    if q_df.empty:
        print(f"  {QUARTER_NAMES[q]}: NO DATA – skipping")
        continue

    basket_matrix, n_baskets = build_basket_matrix(q_df)

    print(f"\n  {QUARTER_NAMES[q]}:")
    print(f"    Invoices : {n_baskets:,}")
    print(f"    Items    : {basket_matrix.shape[1]:,}")

    fi = fpgrowth(
        basket_matrix,
        min_support=MIN_SUPPORT_QUARTERLY,
        use_colnames=True,
    )
    fi["length"] = fi["itemsets"].apply(len)
    fi["quarter"] = q

    if fi.empty:
        print(f"    ⚠ No frequent itemsets found (try lowering MIN_SUPPORT_QUARTERLY)")
        continue

    r = association_rules(fi, metric="confidence", min_threshold=MIN_CONFIDENCE)
    r  = r[r["lift"] >= MIN_LIFT].copy()
    r["antecedents_str"] = r["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    r["consequents_str"] = r["consequents"].apply(lambda x: ", ".join(sorted(x)))
    r["quarter"] = q

    quarter_itemsets[q] = fi
    quarter_rules[q]    = r

    quarter_stats.append({
        "Quarter"          : QUARTER_NAMES[q],
        "Invoices"         : n_baskets,
        "Frequent Itemsets": len(fi),
        "Rules"            : len(r),
        "Avg Lift"         : round(r["lift"].mean(), 3) if not r.empty else 0,
    })

    print(f"    Frequent itemsets : {len(fi):,}")
    print(f"    Rules             : {len(r):,}")

# =============================================================================
# STEP 3  –  Compute temporal stability scores
# =============================================================================
print("\nComputing stability scores …")

# Combine all itemsets; convert frozensets → sorted strings for easy comparison
all_itemsets = []
for q, fi in quarter_itemsets.items():
    fi_copy = fi.copy()
    fi_copy["itemset_str"] = fi_copy["itemsets"].apply(lambda x: " | ".join(sorted(x)))
    all_itemsets.append(fi_copy[["itemset_str", "support", "length", "quarter"]])

combined = pd.concat(all_itemsets, ignore_index=True)

# Stability = number of quarters in which this itemset appeared (1–4)
stability = (
    combined.groupby("itemset_str")
    .agg(
        quarters_present  = ("quarter", "nunique"),
        quarters_list     = ("quarter", lambda x: sorted(x.unique().tolist())),
        mean_support      = ("support", "mean"),
        itemset_length    = ("length", "first"),
    )
    .reset_index()
    .sort_values(["quarters_present", "mean_support"], ascending=[False, False])
)

stability["stability_label"] = stability["quarters_present"].map({
    4: "Stable (all 4 quarters)",
    3: "Mostly stable (3/4)",
    2: "Seasonal (2/4)",
    1: "Rare / trend (1/4)",
})

print(f"\n  Total unique itemsets across all quarters : {len(stability):,}")
print(stability["stability_label"].value_counts().to_string())

# Top stable itemsets (appear in all 4 quarters)
stable_all = stability[stability["quarters_present"] == 4].head(20)
print(f"\nTop itemsets stable across ALL 4 quarters (top 10):")
print(stable_all[["itemset_str", "mean_support", "itemset_length"]].head(10).to_string(index=False))

# =============================================================================
# STEP 4  –  Save results
# =============================================================================
# Per-quarter rules
for q, r in quarter_rules.items():
    path = OUT_DIR / f"rules_Q{q}.csv"
    r[["antecedents_str", "consequents_str", "support",
       "confidence", "lift", "quarter"]].to_csv(path, index=False)
    print(f"\nQ{q} rules saved → {path}")

# Stability summary
stab_path = OUT_DIR / "temporal_stability.csv"
stability_out = stability.copy()
stability_out["quarters_list"] = stability_out["quarters_list"].apply(str)
stability_out.to_csv(stab_path, index=False)
print(f"\nStability summary saved → {stab_path}")

# Quarter stats summary
stats_df = pd.DataFrame(quarter_stats)
stats_path = OUT_DIR / "quarterly_stats.csv"
stats_df.to_csv(stats_path, index=False)
print(f"Quarterly stats saved   → {stats_path}")

# =============================================================================
# STEP 5  –  Visualisations
# =============================================================================
print("\nGenerating plots …")

quarters_present = sorted(quarter_itemsets.keys())

# -- 5a: Itemset count & rule count per quarter --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

q_labels = [QUARTER_NAMES[q] for q in quarters_present]
itemset_counts = [len(quarter_itemsets[q]) for q in quarters_present]
rule_counts    = [len(quarter_rules[q])    for q in quarters_present]

axes[0].bar(q_labels, itemset_counts, color="steelblue", edgecolor="white")
axes[0].set_title("Frequent Itemsets per Quarter")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(q_labels, rule_counts, color="coral", edgecolor="white")
axes[1].set_title("Association Rules per Quarter")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=20)

plt.suptitle("Quarterly Pattern Mining Summary", fontsize=13, fontweight="bold")
plt.tight_layout()
plot1 = OUT_DIR / "quarterly_counts.png"
fig.savefig(plot1, dpi=150)
plt.close(fig)
print(f"  Quarterly counts plot → {plot1}")

# -- 5b: Average lift per quarter (trend line) ---------------------------------
avg_lifts = [quarter_rules[q]["lift"].mean() for q in quarters_present]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(q_labels, avg_lifts, marker="o", linewidth=2, color="darkorange")
ax.set_ylabel("Average Lift")
ax.set_title("Average Rule Lift by Quarter\n(higher = stronger item associations)")
ax.set_ylim(bottom=1)
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plot2 = OUT_DIR / "quarterly_avg_lift.png"
fig.savefig(plot2, dpi=150)
plt.close(fig)
print(f"  Lift trend plot       → {plot2}")

# -- 5c: Stability breakdown pie chart -----------------------------------------
stab_counts = stability["stability_label"].value_counts()
colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.pie(stab_counts.values, labels=stab_counts.index,
       autopct="%1.1f%%", colors=colors[:len(stab_counts)],
       startangle=140, pctdistance=0.8)
ax.set_title("Itemset Temporal Stability\n(across 4 quarters)")
plt.tight_layout()
plot3 = OUT_DIR / "stability_pie.png"
fig.savefig(plot3, dpi=150)
plt.close(fig)
print(f"  Stability pie chart   → {plot3}")

# -- 5d: Support heatmap for top stable itemsets across quarters ---------------
# Pivot: itemset (rows) × quarter (cols) → support value (0 if absent)
pivot_data = combined.pivot_table(
    index="itemset_str", columns="quarter", values="support", aggfunc="mean"
).fillna(0)

# Keep only the top 20 itemsets by mean support
top_items = combined.groupby("itemset_str")["support"].mean().nlargest(20).index
pivot_top = pivot_data.loc[pivot_data.index.isin(top_items)]
pivot_top.columns = [QUARTER_NAMES[c] for c in pivot_top.columns]

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    pivot_top, annot=True, fmt=".3f", cmap="YlOrRd",
    linewidths=0.5, ax=ax, cbar_kws={"label": "Support"}
)
ax.set_title("Top 20 Itemsets – Support Across Quarters", fontsize=12)
ax.set_xlabel("")
ax.set_ylabel("Itemset")
plt.tight_layout()
plot4 = OUT_DIR / "stability_heatmap.png"
fig.savefig(plot4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Stability heatmap     → {plot4}")

# =============================================================================
# DONE
# =============================================================================
print("\n✅  03_temporal_stability.py complete!")
print(f"   All outputs saved in: {OUT_DIR.resolve()}")
print("\nKey findings to look for in outputs/:")
print("  • temporal_stability.csv   – which itemsets are stable vs seasonal")
print("  • stability_heatmap.png    – visual of how support changes per quarter")
print("  • quarterly_avg_lift.png   – is Q4 (holiday) showing stronger bundling?")
