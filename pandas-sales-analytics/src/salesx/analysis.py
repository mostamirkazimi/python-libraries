"""Analytical building blocks: resampling, groupby, pivots, RFM and cohorts."""
import pandas as pd


def monthly_revenue(enriched: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue with a 3-month rolling mean and month-over-month growth."""
    return (
        enriched.set_index("order_date")["revenue"]
        .resample("MS").sum()
        .to_frame("revenue")
        .assign(rolling_3m=lambda d: d["revenue"].rolling(3).mean(),
                mom_growth=lambda d: d["revenue"].pct_change())
    )


def top_products(enriched: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Named aggregation per product, ranked by revenue."""
    return (
        enriched.groupby(["product_id", "category"], observed=True)
        .agg(revenue=("revenue", "sum"),
             units=("quantity", "sum"),
             orders=("order_id", "nunique"),
             avg_discount=("discount", "mean"))
        .sort_values("revenue", ascending=False)
        .head(n)
        .round(3)
    )


def pivot_region_category(enriched: pd.DataFrame) -> pd.DataFrame:
    """Revenue by region (rows) and category (columns) with totals."""
    return pd.pivot_table(enriched, index="region", columns="category",
                          values="revenue", aggfunc="sum", margins=True,
                          margins_name="Total").round(0)


def rfm_segments(enriched: pd.DataFrame) -> pd.DataFrame:
    """Recency / Frequency / Monetary scoring with quartiles."""
    today = enriched["order_date"].max() + pd.Timedelta(days=1)
    t = enriched.groupby("customer_id").agg(
        recency=("order_date", lambda s: (today - s.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
    )
    # rank(method="first") breaks ties so qcut always gets 4 distinct bins
    t["r"] = pd.qcut(t["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    t["f"] = pd.qcut(t["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    t["m"] = pd.qcut(t["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    t["score"] = t[["r", "f", "m"]].sum(axis=1)
    t["segment"] = pd.cut(t["score"], bins=[2, 5, 8, 12],
                          labels=["At risk", "Regular", "Champions"])
    return t


def cohort_retention(enriched: pd.DataFrame) -> pd.DataFrame:
    """Share of each first-purchase-month cohort that orders again N months later."""
    d = enriched[["customer_id", "order_date"]].copy()
    d["order_month"] = d["order_date"].dt.to_period("M")
    d["cohort"] = d.groupby("customer_id")["order_month"].transform("min")
    d["period"] = ((d["order_month"].dt.year - d["cohort"].dt.year) * 12
                   + (d["order_month"].dt.month - d["cohort"].dt.month))
    counts = (d.groupby(["cohort", "period"])["customer_id"].nunique()
              .unstack(fill_value=0))
    return counts.div(counts[0], axis=0).round(3)
