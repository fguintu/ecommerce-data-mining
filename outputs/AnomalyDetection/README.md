# Anomaly Detection Results

Isolation Forest applied to **per-invoice features** from the Online Retail II dataset.
The model flags ~2% of invoices as anomalous (`contamination=0.02`), identifying orders
that deviate significantly from typical purchasing patterns.

---

## Method

1. **Feature engineering** — each invoice is summarized into 8 features:
   - `TotalQuantity` — sum of item quantities in the invoice
   - `TotalRevenue` — sum of line revenues (quantity x price)
   - `NumLineItems` — number of line items
   - `NumDistinct` — number of distinct products
   - `AvgUnitPrice` — mean unit price across line items
   - `MaxUnitPrice` — highest unit price in the invoice
   - `AvgQuantityPerLine` — average quantity per line item
   - `RevenuePerLine` — average revenue per line item

2. **Standardization** — features are z-score standardized before fitting.

3. **Isolation Forest** (`scikit-learn`) — 200 estimators, 2% contamination.
   The algorithm isolates anomalies by randomly partitioning features; points that
   are isolated in fewer splits receive lower (more anomalous) scores.

---

## Output Files

### Plots

| File | What it shows |
|------|---------------|
| `04_score_distribution.png` | **Left:** histogram of anomaly scores across all invoices with the decision threshold marked in red. Invoices left of the threshold are flagged. **Right:** scatter of Revenue vs Quantity colored by normal/anomaly label — anomalies cluster in the high-value tails. |
| `04_feature_comparison.png` | Bar chart comparing mean feature values for anomaly vs normal invoices. Shows which dimensions drive the flags (e.g., TotalQuantity is ~15x higher for anomalies). |
| `04_box_plots.png` | Box plots for four key features (Revenue, Quantity, LineItems, MaxPrice) split by label. Illustrates the spread and outlier range for each group. |
| `04_top_anomalies_table.png` | Table image of the 10 most anomalous invoices with their feature values and anomaly scores. |
| `04_anomaly_by_segment.png` | Anomaly rate (%) broken down by RFM cluster segment (High / Average / Low Performance). Shows whether anomalies concentrate in a particular customer tier. |
| `04_monthly_trend.png` | Dual-axis chart: monthly total invoices (bars) vs anomaly count (red bars) and anomaly rate (orange line). Reveals whether anomalous activity is seasonal or spiking. |
| `04_anomaly_by_country.png` | Top 10 countries by number of anomalous invoices. Useful for identifying geographic concentration of unusual orders. |
| `04_feature_correlation.png` | Heatmap of Pearson correlations among the 8 invoice features. Helps interpret which features are redundant vs independent signals. |

### CSV Files

| File | Description |
|------|-------------|
| `invoice_anomaly_scores.csv` | Full invoice-level table with all engineered features, anomaly label (1=normal, -1=anomaly), and continuous anomaly score. |
| `top_anomalies.csv` | The 20 most anomalous invoices ranked by score (most anomalous first). Includes a `TopItems` column listing the top 5 items (by revenue) bought in each invoice with quantities and line revenue. |
| `anomaly_line_items.csv` | Full line-item detail for every anomalous invoice — one row per product purchased, with StockCode, Description, Quantity, Price, and LineRevenue. Sorted by anomaly score then revenue. Use this to drill into exactly what was bought in each flagged order. |
| `feature_comparison.csv` | Mean feature values for anomaly vs normal groups, plus the ratio between them. |

---

## How to Interpret

- **Anomaly Score**: a continuous value where **lower = more anomalous**. The threshold is set automatically by the contamination parameter. Scores well below the threshold indicate extreme outliers.

- **What makes an invoice anomalous?** The feature comparison table shows that flagged invoices have dramatically higher quantities (~15x), revenue (~12x), and per-line averages (~25-29x) compared to normal invoices. These are bulk/wholesale-scale orders or high-value purchases that fall outside typical retail patterns.

- **Segment breakdown**: High Performance customers have the highest anomaly rate (~2.6%) because their naturally large orders are more likely to cross the extreme threshold. Low Performance customers show the lowest rate (~1.1%) — their orders are consistently small.

- **Monthly trends**: look for months where the anomaly rate spikes above the baseline — these may indicate seasonal wholesale events, data entry errors, or fraudulent activity worth investigating.

- **Country breakdown**: the United Kingdom dominates anomaly counts (as it does overall volume). Disproportionate anomaly rates in smaller countries may indicate regional wholesale customers or data quality issues.

---

## Re-running

```bash
python scripts/04_anomaly_detection.py
```

Requires `data/online_retail_cleaned.parquet` (from `00_preprocess.py`) and optionally
`outputs/Clustering/rfm_segments.csv` (from `01_rfm_clustering.py`) for the segment breakdown.
