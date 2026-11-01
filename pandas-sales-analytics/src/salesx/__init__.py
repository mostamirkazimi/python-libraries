"""salesx: an end-to-end sales analytics pipeline built with pandas."""
from .analysis import (cohort_retention, monthly_revenue, pivot_region_category,
                       rfm_segments, top_products)
from .cleaning import clean_orders, enrich, optimize_memory
from .data_gen import generate_dataset

__all__ = ["generate_dataset", "clean_orders", "enrich", "optimize_memory",
           "monthly_revenue", "top_products", "pivot_region_category",
           "rfm_segments", "cohort_retention"]
__version__ = "0.1.0"
