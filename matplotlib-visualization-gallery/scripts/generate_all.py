"""Generate every figure into ./outputs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from mplx import animate_wave, business_dashboard, radar_chart, surface_3d

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
business_dashboard().savefig(OUT / "dashboard.png")
surface_3d().savefig(OUT / "surface_3d.png")
radar_chart().savefig(OUT / "radar.png")
animate_wave(OUT / "wave.gif")
print("saved:", *sorted(p.name for p in OUT.iterdir()))
