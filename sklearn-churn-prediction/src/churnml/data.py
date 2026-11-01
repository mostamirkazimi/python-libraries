"""Synthetic customer-churn data with mixed types and missing values."""
import numpy as np
import pandas as pd


def make_churn_dataset(n=4000, seed=0):
    """Return (X, y) where X is a DataFrame with numeric + categorical columns."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(65, 25, n).clip(15, 150).round(2)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.4, 0.15])
    payment = rng.choice(["card", "bank", "check", "wallet"], n)
    calls = rng.poisson(1.5, n)
    senior = rng.integers(0, 2, n)
    total = (tenure * monthly * rng.normal(1, 0.03, n)).round(2)

    logit = (-0.8
             - 0.045 * tenure
             + 0.02 * (monthly - 65)
             + 0.45 * calls
             + 1.1 * (contract == "month-to-month")
             - 0.9 * (contract == "two-year")
             + 0.6 * (internet == "fiber")
             + 0.3 * senior)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    X = pd.DataFrame({
        "tenure": tenure, "monthly_charges": monthly, "total_charges": total,
        "support_calls": calls, "senior": senior,
        "contract": contract, "internet_service": internet, "payment_method": payment,
    })
    # Realistic missing values
    X.loc[rng.random(n) < 0.04, "monthly_charges"] = np.nan
    X.loc[rng.random(n) < 0.03, "total_charges"] = np.nan
    X.loc[rng.random(n) < 0.03, "contract"] = np.nan
    return X, pd.Series(y, name="churn")
