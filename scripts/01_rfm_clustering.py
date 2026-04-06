"""
01_rfm_clustering.py
--------------------
RFM clustering following the article approach:
  - Log-transform RFM features (right-skew correction, per README)
  - Min-Max normalize (per article)
  - Elbow method to select k, defaulting to k=3 (per article reasoning)
  - Labels: High Performance / Average Performance / Low Performance

Run from anywhere: paths are resolved relative to this file.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# -- Config -------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "online_retail_cleaned.parquet"
OUTPUT_DIR = ROOT / "outputs" / "Clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_RANGE     = range(2, 10)
K_FINAL     = 3          # article: k=2 oversimplifies; k=3 balances granularity
RANDOM_SEED = 42

def p(name):
    """Return absolute path string for an output file."""
    return str(OUTPUT_DIR / name)

# -- 1. Load ------------------------------------------------------------------
print("Loading data...")
df = pd.read_parquet(DATA_PATH)
print(f"  {len(df):,} rows | {df['CustomerID'].nunique():,} customers")

# -- 2. Compute RFM -----------------------------------------------------------
snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"  Snapshot date: {snapshot.date()}")

rfm = (
    df.groupby("CustomerID")
    .agg(
        Recency  =("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("Invoice",     "nunique"),
        Monetary =("LineRevenue", "sum"),
    )
    .reset_index()
)

print("\nRFM summary (raw):")
print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2).to_string())

# -- 3. Log-transform then Min-Max scale --------------------------------------
rfm["R_log"] = np.log1p(rfm["Recency"])
rfm["F_log"] = np.log1p(rfm["Frequency"])
rfm["M_log"] = np.log1p(rfm["Monetary"])

features = rfm[["R_log", "F_log", "M_log"]].values
scaler   = MinMaxScaler()
X        = scaler.fit_transform(features)

# -- 4. Elbow method ----------------------------------------------------------
print("\nEvaluating k (elbow method)...")
inertias    = []
silhouettes = []

for k in K_RANGE:
    km     = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    print(f"  k={k}: inertia={km.inertia_:,.0f}  silhouette={silhouettes[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_RANGE), inertias, marker="o")
axes[0].axvline(x=K_FINAL, color="red", linestyle="--", label=f"Chosen k={K_FINAL}")
axes[0].set_title("Elbow - Inertia vs k")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia (WCSS)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_RANGE), silhouettes, marker="o", color="orange")
axes[1].axvline(x=K_FINAL, color="red", linestyle="--", label=f"Chosen k={K_FINAL}")
axes[1].set_title("Silhouette Score vs k")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(p("01_elbow_silhouette.png"), dpi=150)
plt.close(fig)
print(f"  Saved -> outputs/Clustering/01_elbow_silhouette.png")

# -- 5. Final model: k=3 ------------------------------------------------------
print(f"\nFitting k={K_FINAL} (k=2 oversimplifies; k=3 separates low/avg/high)...")
km_final       = KMeans(n_clusters=K_FINAL, n_init=20, random_state=RANDOM_SEED)
rfm["Cluster"] = km_final.fit_predict(X)

# -- 6. Profile clusters by raw RFM means -------------------------------------
profile = (
    rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
    .mean().round(1)
    .sort_values("Monetary", ascending=False)
)
print("\nCluster profiles (raw RFM means):")
print(profile.to_string())

# -- 7. Label clusters: High / Average / Low Performance ----------------------
rfm_stats = rfm.groupby("Cluster").agg(
    R_mean=("Recency",    "mean"),
    F_mean=("Frequency",  "mean"),
    M_mean=("Monetary",   "mean"),
    size  =("CustomerID", "count"),
).reset_index()

for col in ["R_mean", "F_mean", "M_mean"]:
    mn, mx = rfm_stats[col].min(), rfm_stats[col].max()
    rfm_stats[col + "_n"] = (rfm_stats[col] - mn) / (mx - mn + 1e-9)

rfm_stats["score"] = (
    (1 - rfm_stats["R_mean_n"]) * 0.30 +
    rfm_stats["F_mean_n"]       * 0.35 +
    rfm_stats["M_mean_n"]       * 0.35
)
rfm_stats = rfm_stats.sort_values("score", ascending=False).reset_index(drop=True)

LABELS    = ["High Performance", "Average Performance", "Low Performance"]
label_map = {int(row["Cluster"]): LABELS[i] for i, row in rfm_stats.iterrows()}
rfm["Segment"] = rfm["Cluster"].map(label_map)

seg_colors = {
    "High Performance":    "#2ca02c",
    "Average Performance": "#ff7f0e",
    "Low Performance":     "#d62728",
}

seg_summary = (
    rfm.groupby("Segment")
    .agg(
        Customers=("CustomerID", "count"),
        Recency  =("Recency",    "mean"),
        Frequency=("Frequency",  "mean"),
        Monetary =("Monetary",   "mean"),
    )
    .round(1)
    .loc[LABELS]
)
print("\nSegment summary:")
print(seg_summary.to_string())

# -- 8. Silhouette plot -------------------------------------------------------
sil_vals = silhouette_samples(X, rfm["Cluster"].values)
rfm["Silhouette"] = sil_vals
avg_sil  = silhouette_score(X, rfm["Cluster"].values)

fig, ax = plt.subplots(figsize=(8, 5))
y_lower = 10
colors  = cm.tab10(np.linspace(0, 1, K_FINAL))

for i, cluster_id in enumerate(sorted(rfm["Cluster"].unique())):
    cluster_sil = np.sort(sil_vals[rfm["Cluster"] == cluster_id])
    y_upper = y_lower + len(cluster_sil)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                     facecolor=colors[i], alpha=0.7,
                     label=f"{label_map[cluster_id]} (n={len(cluster_sil)})")
    y_lower = y_upper + 10

ax.axvline(x=avg_sil, color="red", linestyle="--",
           label=f"Avg silhouette = {avg_sil:.3f}")
ax.set_title(f"Silhouette Plot - k={K_FINAL}")
ax.set_xlabel("Silhouette coefficient"); ax.set_ylabel("Cluster")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
fig.savefig(p("01_silhouette_plot.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/Clustering/01_silhouette_plot.png")

# -- 9. Scatter: Recency vs log-Monetary --------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for seg in LABELS:
    grp = rfm[rfm["Segment"] == seg]
    ax.scatter(grp["Recency"], np.log1p(grp["Monetary"]),
               label=seg, alpha=0.5, s=15, color=seg_colors[seg])

ax.set_xlabel("Recency (days since last purchase)")
ax.set_ylabel("log(1 + Monetary)")
ax.set_title("Customer Segments - Recency vs log-Monetary")
ax.legend(markerscale=2, fontsize=9)
plt.tight_layout()
fig.savefig(p("01_scatter_recency_monetary.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/Clustering/01_scatter_recency_monetary.png")

# -- 10. RFM heatmap ----------------------------------------------------------
heatmap_data = (
    rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
    .mean()
    .apply(lambda col: (col - col.mean()) / col.std())
    .loc[LABELS]
)
heatmap_data["Recency"] = -heatmap_data["Recency"]

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, center=0)
ax.set_title("RFM Heatmap by Segment (z-scored; Recency inverted)")
plt.tight_layout()
fig.savefig(p("01_rfm_heatmap.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/Clustering/01_rfm_heatmap.png")

# -- 11. Segment size bar chart -----------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
seg_summary["Customers"].plot(
    kind="bar", ax=ax,
    color=[seg_colors[s] for s in seg_summary.index]
)
ax.set_title("Customer Count per Segment")
ax.set_xlabel(""); ax.set_ylabel("Customers")
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
fig.savefig(p("01_segment_sizes.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/Clustering/01_segment_sizes.png")

# -- 12. Mean RFM bar charts --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, metric in zip(axes, ["Recency", "Frequency", "Monetary"]):
    vals = seg_summary[metric]
    ax.bar(LABELS, vals, color=[seg_colors[s] for s in LABELS])
    ax.set_title(f"Mean {metric} per Segment")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=15)
    for j, v in enumerate(vals):
        ax.text(j, v * 1.01, f"{v:,.1f}", ha="center", fontsize=8)

plt.suptitle("Mean RFM Values by Segment", y=1.02)
plt.tight_layout()
fig.savefig(p("01_rfm_means_bar.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> outputs/Clustering/01_rfm_means_bar.png")

# -- 13. Centroid table -------------------------------------------------------
centroids_raw = np.expm1(scaler.inverse_transform(km_final.cluster_centers_))
centroid_df   = pd.DataFrame(centroids_raw,
                              columns=["Recency", "Frequency", "Monetary"])
centroid_df.index = [label_map[i] for i in range(K_FINAL)]
centroid_df       = centroid_df.loc[LABELS].round(1)

print("\nCluster centroids (original units):")
print(centroid_df.to_string())

fig, ax = plt.subplots(figsize=(9, 2))
ax.axis("off")

table_data = [
    [seg,
     f"{int(round(centroid_df.loc[seg, 'Recency']))} days",
     f"{centroid_df.loc[seg, 'Frequency']:.1f}",
     f"£{centroid_df.loc[seg, 'Monetary']:,.0f}"]
    for seg in LABELS
]
col_labels = ["Segment", "Recency", "Frequency (invoices)", "Monetary (£)"]

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               colWidths=[0.35, 0.20, 0.25, 0.20],
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#2c2c2c")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

for i, seg in enumerate(LABELS):
    tbl[i + 1, 0].set_facecolor(seg_colors[seg])
    tbl[i + 1, 0].set_text_props(color="white", fontweight="bold")
    for j in range(1, len(col_labels)):
        tbl[i + 1, j].set_facecolor("#f9f9f9")

ax.set_title("Cluster Centroids (original units)", fontsize=12,
             fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(p("01_centroid_table.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> outputs/Clustering/01_centroid_table.png")

# -- 14. Sanity check: Frequency vs Monetary ----------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for seg in LABELS:
    grp = rfm[rfm["Segment"] == seg]
    ax.scatter(grp["Frequency"], np.log1p(grp["Monetary"]),
               label=seg, alpha=0.5, s=15, color=seg_colors[seg])

ax.set_xlabel("Frequency (number of invoices)")
ax.set_ylabel("log(1 + Monetary)")
ax.set_title("Sanity Check - Frequency vs log-Monetary by Segment")
ax.legend(markerscale=2, fontsize=9)
plt.tight_layout()
fig.savefig(p("01_scatter_freq_monetary.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/Clustering/01_scatter_freq_monetary.png")

# -- 15. 3D scatter -----------------------------------------------------------
fig = plt.figure(figsize=(11, 8))
ax3d = fig.add_subplot(111, projection="3d")

for seg in LABELS:
    grp = rfm[rfm["Segment"] == seg]
    ax3d.scatter(grp["Recency"], grp["Frequency"], np.log1p(grp["Monetary"]),
                 label=seg, alpha=0.4, s=10, color=seg_colors[seg])

ax3d.set_xlabel("Recency (days)", labelpad=8)
ax3d.set_ylabel("Frequency (invoices)", labelpad=8)
ax3d.set_zlabel("log(1 + Monetary)", labelpad=8)
ax3d.set_title("RFM Clusters - 3D View")
ax3d.legend(markerscale=2, fontsize=9)
plt.tight_layout()
fig.savefig(p("01_scatter_3d.png"), dpi=150)
plt.close(fig)
print(f"Saved -> outputs/Clustering/01_scatter_3d.png")

# -- 16. Save labelled table --------------------------------------------------
out_cols = ["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Segment"]
rfm[out_cols].to_csv(p("rfm_segments.csv"), index=False)
print(f"Saved -> outputs/Clustering/rfm_segments.csv")

print("\nDone.")
