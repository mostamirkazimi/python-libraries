import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mplx import animate_wave, business_dashboard, radar_chart, surface_3d


def test_dashboard_has_five_panels():
    fig = business_dashboard()
    titled = [ax for ax in fig.axes if ax.get_title(loc="left")]
    assert len(titled) == 5
    plt.close(fig)


def test_surface_is_3d():
    fig = surface_3d(n=30)
    assert fig.axes[0].name == "3d"
    plt.close(fig)


def test_radar_is_polar_and_saves_png():
    fig = radar_chart()
    assert fig.axes[0].name == "polar"
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "radar.png"
        fig.savefig(out)
        assert out.stat().st_size > 1000
    plt.close(fig)


def test_animation_writes_gif():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "wave.gif"
        animate_wave(out, frames=5)
        assert out.read_bytes()[:3] == b"GIF"
