import numpy as np

from salesx import (clean_orders, cohort_retention, enrich, generate_dataset,
                    monthly_revenue, pivot_region_category, rfm_segments)
from salesx.cleaning import memory_mb, optimize_memory


def _data():
    raw, customers, products = generate_dataset(n_customers=300, n_orders=4000, seed=1)
    return raw, enrich(clean_orders(raw), customers, products)


def test_cleaning_fixes_problems():
    raw, data = _data()
    assert len(data) < len(raw)                      # duplicates + bad dates removed
    assert data["order_date"].notna().all()
    assert data["discount"].notna().all()
    assert set(data["channel"].unique()) == {"web", "store", "mobile"}
    assert data["quantity"].max() < 999              # outlier clipped


def test_monthly_revenue_matches_total():
    _, data = _data()
    m = monthly_revenue(data)
    assert np.isclose(m["revenue"].sum(), data["revenue"].sum())


def test_pivot_total_matches():
    _, data = _data()
    p = pivot_region_category(data)
    assert abs(p.loc["Total", "Total"] - data["revenue"].sum()) < len(data)


def test_rfm_scores_and_segments():
    _, data = _data()
    rfm = rfm_segments(data)
    assert rfm["score"].between(3, 12).all()
    assert rfm["segment"].notna().all()


def test_cohort_retention_starts_at_one():
    _, data = _data()
    c = cohort_retention(data)
    assert (c[0] == 1.0).all()
    assert c.to_numpy().max() <= 1.0


def test_memory_optimisation_shrinks_dataframe():
    _, data = _data()
    assert memory_mb(optimize_memory(data)) < memory_mb(data)
