"""Cleaning, enrichment and memory optimisation using method chaining."""
import numpy as np
import pandas as pd


def clean_orders(raw: pd.DataFrame) -> pd.DataFrame:
    """Fix the problems injected by ``generate_dataset``."""
    return (
        raw
        .drop_duplicates()
        .assign(
            channel=lambda d: d["channel"].str.strip().str.lower(),
            order_date=lambda d: pd.to_datetime(d["order_date"], errors="coerce"),
            discount=lambda d: d["discount"].fillna(0.0),
        )
        .dropna(subset=["order_date"])
        .pipe(_clip_quantity)
        .astype({"channel": "category"})
        .reset_index(drop=True)
    )


def _clip_quantity(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Cap extreme quantities at Q3 + k * IQR (a robust outlier rule)."""
    q1, q3 = df["quantity"].quantile([0.25, 0.75])
    cap = q3 + k * (q3 - q1)
    return df.assign(quantity=df["quantity"].clip(upper=cap).astype(int))


def enrich(orders: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Join the three tables and compute revenue."""
    return (
        orders
        .merge(products, on="product_id", how="left", validate="many_to_one")
        .merge(customers[["customer_id", "region"]], on="customer_id", how="left",
               validate="many_to_one")
        .assign(revenue=lambda d: (d["quantity"] * d["unit_price"] * (1 - d["discount"])).round(2))
    )


def optimize_memory(df: pd.DataFrame, max_unique_ratio: float = 0.5) -> pd.DataFrame:
    """Downcast numbers and convert low-cardinality text to ``category``."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            out[col] = pd.to_numeric(s, downcast="float")
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            if s.nunique() / max(len(s), 1) < max_unique_ratio:
                out[col] = s.astype("category")
    return out


def memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024 ** 2
