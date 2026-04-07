"""
04_anomaly_detection.py
-----------------------
Isolation Forest anomaly detection on per-invoice features.
  - Engineers per-invoice features from the cleaned transaction data
  - Fits Isolation Forest (scikit-learn) with contamination tuning
  - Produces diagnostic plots, summary tables, and flagged invoice list

Run from anywhere: paths are resolved relative to this file.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# -- Config -------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "online_retail_cleaned.parquet"
SEG_PATH   = ROOT / "outputs" / "Clustering" / "rfm_segments.csv"
OUTPUT_DIR = ROOT / "outputs" / "AnomalyDetection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTAMINATION = 0.02   # expect ~2% anomalous invoices
RANDOM_SEED   = 42

def p(name):
    """Return absolute path string for an output file."""
    return str(OUTPUT_DIR / name)

# -- 1. Load ------------------------------------------------------------------
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"  {len(df):,} rows | {df['Invoice'].nunique():,} invoices")

# -- 2. Engineer per-invoice features -----------------------------------------
print("Engineering per-invoice features...")
inv = (
    df.groupby("Invoice")
    .agg(
        TotalQuantity  =("Quantity",    "sum"),
        TotalRevenue   =("LineRevenue", "sum"),
        NumLineItems   =("StockCode",   "count"),
        NumDistinct    =("StockCode",   "nunique"),
        AvgUnitPrice   =("Price",       "mean"),
        MaxUnitPrice   =("Price",       "max"),
        CustomerID     =("CustomerID",  "first"),
        InvoiceDate    =("InvoiceDate", "first"),
        Country        =("Country",     "first"),
    )
    .reset_index()
)
inv["AvgQuantityPerLine"] = inv["TotalQuantity"] / inv["NumLineItems"]
inv["RevenuePerLine"]     = inv["TotalRevenue"]  / inv["NumLineItems"]

print(f"  {len(inv):,} invoices built")
print("\nPer-invoice feature summary:")
feat_cols = [
    "TotalQuantity", "TotalRevenue", "NumLineItems", "NumDistinct",
    "AvgUnitPrice", "MaxUnitPrice", "AvgQuantityPerLine", "RevenuePerLine",
]
print(inv[feat_cols].describe().round(2).to_string())

# -- 3. Scale features --------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(inv[feat_cols].values)

# -- 4. Fit Isolation Forest --------------------------------------------------
print(f"\nFitting Isolation Forest (contamination={CONTAMINATION})...")
iso = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
inv["AnomalyLabel"] = iso.fit_predict(X)          # 1 = normal, -1 = anomaly
inv["AnomalyScore"] = iso.decision_function(X)    # lower = more anomalous

n_anom = (inv["AnomalyLabel"] == -1).sum()
print(f"  Anomalies flagged: {n_anom:,} / {len(inv):,} "
      f"({100 * n_anom / len(inv):.1f}%)")

# -- 5. Anomaly score distribution -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(inv["AnomalyScore"], bins=80, edgecolor="black", alpha=0.7)
threshold = inv.loc[inv["AnomalyLabel"] == -1, "AnomalyScore"].max()
axes[0].axvline(x=threshold, color="red", linestyle="--",
                label=f"Threshold = {threshold:.4f}")
axes[0].set_title("Anomaly Score Distribution")
axes[0].set_xlabel("Anomaly Score (lower = more anomalous)")
axes[0].set_ylabel("Number of Invoices")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

inv["Label"] = inv["AnomalyLabel"].map({1: "Normal", -1: "Anomaly"})
for label, color in [("Normal", "#2ca02c"), ("Anomaly", "#d62728")]:
    subset = inv[inv["Label"] == label]
    axes[1].scatter(
        subset["TotalRevenue"], subset["TotalQuantity"],
        label=label, alpha=0.4, s=12, color=color,
    )
axes[1].set_xlabel("Total Revenue (£)")
axes[1].set_ylabel("Total Quantity")
axes[1].set_title("Anomalies: Revenue vs Quantity")
axes[1].legend(markerscale=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(p("04_score_distribution.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/AnomalyDetection/04_score_distribution.png")

# -- 6. Feature comparison: anomaly vs normal ---------------------------------
comparison = inv.groupby("Label")[feat_cols].mean().T.round(2)
comparison["Ratio"] = (comparison["Anomaly"] / comparison["Normal"]).round(2)

print("\nFeature means — Anomaly vs Normal:")
print(comparison.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
comparison[["Normal", "Anomaly"]].plot(kind="bar", ax=ax)
ax.set_title("Mean Feature Values: Anomaly vs Normal Invoices")
ax.set_ylabel("Mean Value")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
fig.savefig(p("04_feature_comparison.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/AnomalyDetection/04_feature_comparison.png")

# -- 7. Feature box plots for key dimensions ---------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
box_feats = ["TotalRevenue", "TotalQuantity", "NumLineItems", "MaxUnitPrice"]

for ax, feat in zip(axes.ravel(), box_feats):
    data_norm = inv.loc[inv["Label"] == "Normal", feat]
    data_anom = inv.loc[inv["Label"] == "Anomaly", feat]
    ax.boxplot(
        [data_norm, data_anom],
        tick_labels=["Normal", "Anomaly"],
        patch_artist=True,
        boxprops=dict(facecolor="#cce5ff"),
    )
    ax.set_title(feat)
    ax.grid(True, alpha=0.3)

plt.suptitle("Feature Distributions: Normal vs Anomaly", fontsize=13)
plt.tight_layout()
fig.savefig(p("04_box_plots.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/AnomalyDetection/04_box_plots.png")

# -- 8. Top anomalies table ---------------------------------------------------
top_anom = (
    inv[inv["AnomalyLabel"] == -1]
    .sort_values("AnomalyScore")
    .head(20)[
        ["Invoice", "CustomerID", "InvoiceDate", "Country",
         "TotalQuantity", "TotalRevenue", "NumLineItems",
         "AvgUnitPrice", "AnomalyScore"]
    ]
    .reset_index(drop=True)
)

# Add item descriptions: summarize what was bought per anomalous invoice
anomaly_invoices = set(inv.loc[inv["AnomalyLabel"] == -1, "Invoice"])
anom_lines = df[df["Invoice"].isin(anomaly_invoices)].copy()

# Build a summary string of top items per invoice (by revenue)
items_summary = (
    anom_lines.sort_values("LineRevenue", ascending=False)
    .groupby("Invoice")
    .apply(
        lambda g: " | ".join(
            f"{row['Description']} (x{int(row['Quantity'])}, £{row['LineRevenue']:.2f})"
            for _, row in g.head(5).iterrows()
        ),
        include_groups=False,
    )
    .rename("TopItems")
)
top_anom = top_anom.merge(items_summary, on="Invoice", how="left")

print("\nTop 20 most anomalous invoices:")
print(top_anom.to_string())

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")
table_cols = [c for c in top_anom.columns if c != "TopItems"]
col_labels = list(table_cols)
cell_text  = top_anom.head(10)[table_cols].values.tolist()
for i, row in enumerate(cell_text):
    cell_text[i] = [
        str(row[0]),                                # Invoice
        str(row[1]),                                # CustomerID
        str(row[2])[:10],                           # InvoiceDate
        str(row[3]),                                # Country
        f"{row[4]:,.0f}",                           # TotalQuantity
        f"£{row[5]:,.2f}",                          # TotalRevenue
        f"{row[6]:,}",                              # NumLineItems
        f"£{row[7]:.2f}",                           # AvgUnitPrice
        f"{row[8]:.4f}",                            # AnomalyScore
    ]

tbl = ax.table(cellText=cell_text, colLabels=col_labels,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.8)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2c2c2c")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title("Top 10 Most Anomalous Invoices", fontsize=12,
             fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(p("04_top_anomalies_table.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> outputs/AnomalyDetection/04_top_anomalies_table.png")

# -- 9. Anomaly rate by customer segment (if clustering output exists) --------
if SEG_PATH.exists():
    print("\nJoining with clustering segments...")
    seg = pd.read_csv(SEG_PATH, dtype={"CustomerID": str})
    inv_seg = inv.merge(seg[["CustomerID", "Segment"]], on="CustomerID", how="left")
    inv_seg["Segment"] = inv_seg["Segment"].fillna("Unknown")

    seg_rates = (
        inv_seg.groupby("Segment")
        .agg(
            Invoices   =("Invoice",      "count"),
            Anomalies  =("AnomalyLabel", lambda x: (x == -1).sum()),
        )
    )
    seg_rates["AnomalyRate"] = (100 * seg_rates["Anomalies"] / seg_rates["Invoices"]).round(2)
    seg_rates = seg_rates.sort_values("AnomalyRate", ascending=False)

    print("\nAnomaly rate by customer segment:")
    print(seg_rates.to_string())

    seg_colors = {
        "High Performance":    "#2ca02c",
        "Average Performance": "#ff7f0e",
        "Low Performance":     "#d62728",
        "Unknown":             "#999999",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        seg_rates.index,
        seg_rates["AnomalyRate"],
        color=[seg_colors.get(s, "#1f77b4") for s in seg_rates.index],
    )
    for bar, rate in zip(bars, seg_rates["AnomalyRate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{rate:.1f}%", ha="center", fontsize=10)
    ax.set_title("Anomaly Rate by Customer Segment")
    ax.set_ylabel("Anomaly Rate (%)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(p("04_anomaly_by_segment.png"), dpi=150)
    plt.close(fig)
    print(f"Saved -> outputs/AnomalyDetection/04_anomaly_by_segment.png")
else:
    print("\n  (Clustering output not found — skipping segment breakdown.)")

# -- 10. Anomaly count over time ----------------------------------------------
inv["YearMonth"] = inv["InvoiceDate"].dt.to_period("M")
monthly = (
    inv.groupby("YearMonth")
    .agg(
        TotalInvoices=("Invoice",      "count"),
        Anomalies    =("AnomalyLabel", lambda x: (x == -1).sum()),
    )
)
monthly["AnomalyRate"] = (100 * monthly["Anomalies"] / monthly["TotalInvoices"]).round(2)
monthly.index = monthly.index.astype(str)

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.bar(monthly.index, monthly["TotalInvoices"], alpha=0.4, label="Total Invoices")
ax1.bar(monthly.index, monthly["Anomalies"], color="red", alpha=0.7, label="Anomalies")
ax1.set_xlabel("Month")
ax1.set_ylabel("Count")
ax1.tick_params(axis="x", rotation=45)
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3, axis="y")

ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly["AnomalyRate"], color="orange",
         marker="o", linewidth=2, label="Anomaly Rate (%)")
ax2.set_ylabel("Anomaly Rate (%)")
ax2.legend(loc="upper right")

plt.title("Monthly Anomaly Trend")
plt.tight_layout()
fig.savefig(p("04_monthly_trend.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/AnomalyDetection/04_monthly_trend.png")

# -- 11. Anomaly by country (top 10) -----------------------------------------
country_anom = (
    inv[inv["AnomalyLabel"] == -1]
    .groupby("Country")["Invoice"]
    .count()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 5))
country_anom.plot(kind="bar", ax=ax, color="#d62728", alpha=0.8)
ax.set_title("Top 10 Countries by Anomalous Invoice Count")
ax.set_ylabel("Number of Anomalous Invoices")
ax.tick_params(axis="x", rotation=30)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(p("04_anomaly_by_country.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/AnomalyDetection/04_anomaly_by_country.png")

# -- 12. Correlation heatmap of invoice features ------------------------------
fig, ax = plt.subplots(figsize=(9, 7))
corr = inv[feat_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix (per-invoice)")
plt.tight_layout()
fig.savefig(p("04_feature_correlation.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/AnomalyDetection/04_feature_correlation.png")

# -- 13. Save outputs ---------------------------------------------------------
inv.to_csv(p("invoice_anomaly_scores.csv"), index=False)
print(f"Saved -> outputs/AnomalyDetection/invoice_anomaly_scores.csv")

top_anom.to_csv(p("top_anomalies.csv"), index=False)
print(f"Saved -> outputs/AnomalyDetection/top_anomalies.csv")

comparison.to_csv(p("feature_comparison.csv"))
print(f"Saved -> outputs/AnomalyDetection/feature_comparison.csv")

# Full line-item detail for all anomalous invoices
anom_detail = anom_lines.merge(
    inv[["Invoice", "AnomalyScore"]],
    on="Invoice",
    how="left",
).sort_values(["AnomalyScore", "Invoice", "LineRevenue"], ascending=[True, True, False])
anom_detail.to_csv(p("anomaly_line_items.csv"), index=False)
print(f"Saved -> outputs/AnomalyDetection/anomaly_line_items.csv")

print("\nDone.")
