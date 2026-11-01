"""Run the full pipeline and write reports + charts to ./outputs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from salesx import (clean_orders, cohort_retention, enrich, generate_dataset,
                    monthly_revenue, pivot_region_category, rfm_segments, top_products)
from salesx.cleaning import memory_mb, optimize_memory

OUT = Path("outputs")


def main():
    OUT.mkdir(exist_ok=True)
    raw, customers, products = generate_dataset()
    orders = clean_orders(raw)
    data = enrich(orders, customers, products)
    small = optimize_memory(data)

    monthly = monthly_revenue(data)
    top = top_products(data)
    pivot = pivot_region_category(data)
    rfm = rfm_segments(data)
    cohorts = cohort_retention(data)
    seg = rfm.groupby("segment", observed=True).agg(
        customers=("monetary", "size"), avg_revenue=("monetary", "mean")).round(1)

    monthly.to_csv(OUT / "monthly_revenue.csv")
    rfm.to_csv(OUT / "rfm_segments.csv")

    report = [
        "# Sales analytics report", "",
        f"- Raw rows: {len(raw):,}  ->  clean rows: {len(orders):,}",
        f"- Memory: {memory_mb(data):.2f} MB -> {memory_mb(small):.2f} MB after optimisation", "",
        "## Top 10 products", "```", top.to_string(), "```", "",
        "## Revenue by region and category", "```", pivot.to_string(), "```", "",
        "## RFM segments", "```", seg.to_string(), "```", "",
        "## Cohort retention (first 6 months)", "```", cohorts.iloc[:, :7].to_string(), "```",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    monthly[["revenue", "rolling_3m"]].plot(ax=ax[0], title="Monthly revenue")
    ax[0].set_ylabel("revenue")
    im = ax[1].imshow(cohorts.iloc[:, :12].to_numpy(), aspect="auto", cmap="Blues")
    ax[1].set_title("Cohort retention")
    ax[1].set_xlabel("months since first purchase")
    ax[1].set_ylabel("cohort")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=130)
    print((OUT / "report.md").read_text(encoding="utf-8")[:900])


if __name__ == "__main__":
    main()
