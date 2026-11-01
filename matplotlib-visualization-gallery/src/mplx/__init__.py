"""mplx: a gallery of advanced, publication-quality Matplotlib figures."""
from .plots import (animate_wave, business_dashboard, radar_chart, surface_3d)
from .style import PALETTE, apply_style, brand_cmap

__all__ = ["business_dashboard", "surface_3d", "radar_chart", "animate_wave",
           "apply_style", "brand_cmap", "PALETTE"]
__version__ = "0.1.0"
