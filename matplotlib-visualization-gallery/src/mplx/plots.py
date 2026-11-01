"""Each function builds a complete figure and returns it (never calls plt.show)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, gridspec  # noqa: F401  (gridspec kept for users)
from matplotlib.ticker import FuncFormatter

from .style import PALETTE, apply_style, brand_cmap


def business_dashboard(seed=0):
    """A multi-panel dashboard built with ``subplot_mosaic``.

    Shows twin axes, annotations, a heatmap with cell labels, a histogram with
    a KDE computed in NumPy, and a bubble scatter with a colorbar.
    """
    apply_style()
    rng = np.random.default_rng(seed)
    months = np.arange(1, 13)
    revenue = 100 + 8 * months + 15 * np.sin(months / 2) + rng.normal(0, 4, 12)
    margin = 0.25 + 0.01 * np.sin(months / 3) + rng.normal(0, 0.004, 12)

    fig, axd = plt.subplot_mosaic(
        [["trend", "trend", "bars"], ["heat", "hist", "bubble"]],
        figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Business performance dashboard", fontsize=16, fontweight="bold", x=0.01, ha="left")

    # 1) twin axes + annotation
    ax = axd["trend"]
    ax.plot(months, revenue, marker="o", color=PALETTE[0], label="Revenue")
    ax.set_ylabel("Revenue (k$)")
    ax2 = ax.twinx()
    ax2.plot(months, margin * 100, ls="--", color=PALETTE[1], label="Margin %")
    ax2.set_ylabel("Margin (%)")
    ax2.spines["right"].set_visible(True)
    peak = revenue.argmax()
    ax.annotate(f"Peak: {revenue[peak]:.0f}k", xy=(months[peak], revenue[peak]),
                xytext=(months[peak] - 3.5, revenue[peak] - 14),
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_title("Revenue and margin by month")
    ax.set_xticks(months)

    # 2) horizontal bars with value labels
    ax = axd["bars"]
    cats = ["Electronics", "Home", "Books", "Sports", "Beauty"]
    vals = np.sort(rng.uniform(40, 160, 5))
    bars = ax.barh(cats, vals, color=PALETTE[:5])
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_title("Revenue by category")
    ax.grid(axis="y", visible=False)

    # 3) annotated heatmap
    ax = axd["heat"]
    data = rng.integers(10, 100, (4, 6))
    im = ax.imshow(data, cmap=brand_cmap(), aspect="auto")
    for (i, j), v in np.ndenumerate(data):
        ax.text(j, i, v, ha="center", va="center",
                color="white" if v < 35 or v > 80 else "black", fontsize=9)
    ax.set_xticks(range(6), [f"W{k}" for k in range(1, 7)])
    ax.set_yticks(range(4), ["N", "S", "E", "W"])
    ax.grid(False)
    ax.set_title("Orders by region / week")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 4) histogram + Gaussian KDE (pure NumPy)
    ax = axd["hist"]
    sample = np.concatenate([rng.normal(40, 8, 600), rng.normal(75, 10, 400)])
    ax.hist(sample, bins=35, density=True, alpha=0.55, color=PALETTE[2])
    xs = np.linspace(sample.min(), sample.max(), 300)
    h = 1.06 * sample.std() * len(sample) ** (-1 / 5)  # Silverman bandwidth
    kde = np.exp(-0.5 * ((xs[:, None] - sample[None, :]) / h) ** 2).sum(axis=1)
    kde /= len(sample) * h * np.sqrt(2 * np.pi)
    ax.plot(xs, kde, color=PALETTE[4], lw=2)
    ax.set_title("Order value distribution")

    # 5) bubble scatter
    ax = axd["bubble"]
    x, y = rng.normal(size=200), rng.normal(size=200)
    size, col = rng.uniform(20, 300, 200), x + y
    sc = ax.scatter(x, y, s=size, c=col, cmap="viridis", alpha=0.7, edgecolor="white", lw=0.5)
    fig.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title("Customers: size = spend")
    return fig


def surface_3d(n=120):
    """3-D surface with a projected contour plot underneath."""
    apply_style()
    x = np.linspace(-4, 4, n)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sinc(R / np.pi) * np.exp(-0.05 * R ** 2) * 4

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=brand_cmap(), linewidth=0, antialiased=True, alpha=0.95)
    ax.contourf(X, Y, Z, zdir="z", offset=Z.min() - 0.5, cmap=brand_cmap(), alpha=0.6)
    ax.set_zlim(Z.min() - 0.5, Z.max())
    ax.view_init(elev=28, azim=-55)
    ax.set_title("Damped sinc surface")
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1)
    return fig


def radar_chart(labels=None, series=None):
    """Polar radar chart comparing several profiles."""
    apply_style()
    labels = labels or ["Speed", "Accuracy", "Memory", "Interpretability", "Scalability"]
    series = series or {"Linear model": [4, 3, 5, 5, 4],
                        "Random forest": [3, 4, 3, 3, 4],
                        "Gradient boosting": [3, 5, 3, 2, 3]}
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.r_[angles, angles[:1]]  # close the polygon

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"projection": "polar"})
    for (name, vals), color in zip(series.items(), PALETTE):
        v = np.r_[vals, vals[:1]]
        ax.plot(angles, v, color=color, lw=2, label=name)
        ax.fill(angles, v, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0, 5)
    ax.set_title("Model comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig


def animate_wave(path=None, frames=60):
    """Animated travelling wave. Saves a GIF if ``path`` is given."""
    apply_style()
    x = np.linspace(0, 4 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    (line,) = ax.plot([], [], color=PALETTE[0], lw=2)
    ax.set_xlim(0, 4 * np.pi)
    ax.set_ylim(-1.4, 1.4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / np.pi:.0f}π"))
    ax.set_title("Travelling wave")

    def update(i):
        phase = 2 * np.pi * i / frames
        line.set_data(x, np.sin(x - phase) * np.exp(-0.05 * x))
        return (line,)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
    if path:
        anim.save(path, writer=animation.PillowWriter(fps=25))
    return anim
