"""Global style (rcParams) and a custom colormap."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

PALETTE = ["#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626", "#0891b2"]


def apply_style():
    """Apply a clean, consistent look to every figure."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 10,
        "legend.frameon": False,
    })


def brand_cmap(name="brand"):
    """Custom diverging-to-sequential colormap built from three anchor colours."""
    return LinearSegmentedColormap.from_list(name, ["#0f172a", "#2563eb", "#f8fafc", "#f97316"])
