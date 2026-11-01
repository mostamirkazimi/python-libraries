"""Synthetic, deliberately messy e-commerce data (no downloads needed)."""
import numpy as np
import pandas as pd

END_DATE = pd.Timestamp("2024-12-31")


def generate_dataset(n_customers=1500, n_orders=20000, seed=42):
    """Return (raw_orders, customers, products) DataFrames.

    The raw orders contain realistic problems: duplicated rows, inconsistent
    text, missing values, string dates and a few extreme outliers.
    """
    rng = np.random.default_rng(seed)

    products = pd.DataFrame({
        "product_id": np.arange(1, 41),
        "category": rng.choice(["Electronics", "Home", "Books", "Sports", "Beauty"], 40),
        "unit_price": rng.uniform(5, 500, 40).round(2),
    })

    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 600, n_customers), unit="D")
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "region": rng.choice(["North", "South", "East", "West"], n_customers),
        "signup_date": signup,
    })

    # A few customers buy much more often than the rest (heavy tail).
    weights = rng.pareto(2.0, n_customers) + 1
    cust_idx = rng.choice(n_customers, n_orders, p=weights / weights.sum())
    signup_of_order = customers["signup_date"].iloc[cust_idx].reset_index(drop=True)
    max_days = (END_DATE - signup_of_order).dt.days.to_numpy()
    offset = (rng.random(n_orders) * (max_days + 1)).astype(int)
    order_dates = signup_of_order + pd.to_timedelta(offset, unit="D")

    orders = pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "customer_id": customers["customer_id"].to_numpy()[cust_idx],
        "product_id": rng.integers(1, 41, n_orders),
        "quantity": rng.integers(1, 6, n_orders),
        "order_date": order_dates.dt.strftime("%Y-%m-%d"),
        "discount": rng.choice([0.0, 0.05, 0.10, 0.20], n_orders, p=[0.6, 0.2, 0.15, 0.05]),
        "channel": rng.choice(["web", "Web ", "WEB", "store", "Store", " mobile"], n_orders),
    })

    # --- inject data-quality problems -------------------------------------
    orders.loc[rng.choice(n_orders, 600, replace=False), "discount"] = np.nan
    orders.loc[rng.choice(n_orders, 5, replace=False), "quantity"] = 999
    orders.loc[rng.choice(n_orders, 40, replace=False), "order_date"] = "not a date"
    duplicates = orders.sample(200, random_state=seed)
    orders = pd.concat([orders, duplicates], ignore_index=True)

    return orders, customers, products
