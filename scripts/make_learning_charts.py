#!/usr/bin/env python3
r"""
Draw the chart for the website's Imitation Learning page with plotnine (ggplot for Python).

agreement.png - how closely the demonstrations of one skill agree with each other, first dataset
                version against the last. Each demo is stretched onto the same 0-100% timeline,
                then each column of the heatmap shows where the elbow joint was across all the
                demos at that moment. A tight bright band means the demos agree; a wide dim smear
                means they don't.
                Source: datasets/<dataset>/trajectories/episode_*_joints.csv (action_elbow_flex)

Each demo adds a small Gaussian bump (SIGMA degrees wide) at its angle, and the bumps are averaged
over the demos, so every column integrates to 1 whatever the number of demos.

Run with the site-charts venv:
    C:\Users\theoh\venvs\site-charts\Scripts\python.exe scripts/make_learning_charts.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from mizani.bounds import squish   # squish: values above the cap take the top colour
from plotnine import (
    aes, element_blank, element_rect, element_text, facet_wrap, geom_raster, ggplot, labs,
    scale_fill_gradientn, scale_x_continuous, theme, theme_minimal,
)

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets"
OUT = REPO / "docs" / "assets" / "charts"

# Same colours as make_site_charts.py: white ink on the page's blue
SURFACE = "#0a238f"
INK = "#ffffff"

JOINT = "action_elbow_flex"   # the joint with the most movement, so the widest spread
TIME_STEPS = 400              # points on the 0-100% timeline
ANGLE_STEP = 0.5              # degrees per heatmap row
SIGMA = 2.0                   # width of each demo's bump, in degrees
FAINT = 0.002                 # below this the cell is left see-through, so the page blue shows
CAP = 0.12                    # colour scale tops out here, so the still rest pose at 100 degrees
                              # (every demo identical) does not wash out the moving part
# Inferno without its black end, which would vanish into the page blue
COLOURS = [matplotlib.colors.to_hex(c) for c in matplotlib.colormaps["inferno"](np.linspace(0.3, 1, 9))]

# First and last version of each sorting skill. Panel titles are in reading order.
PANELS = [
    ("Green · first version · 22 demos", "block_to_green"),
    ("Green · last version · 6 demos", "block_to_green_v4"),
    ("Pink · first version · 22 demos", "block_to_pink"),
    ("Pink · last version · 12 demos", "block_to_pink_v5"),
]


def site_theme():
    """White text on the page's blue, no grid: the heatmap is the content."""
    return theme_minimal(base_size=11) + theme(
        plot_background=element_rect(fill=SURFACE, color=SURFACE),
        panel_background=element_rect(fill=SURFACE, color=SURFACE),
        text=element_text(color=INK),
        axis_text=element_text(color=INK),
        strip_text=element_text(color=INK, size=11, weight="bold"),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_rect(color="#ffffff", fill=None, alpha=0.35),
        panel_spacing=0.04,
        legend_position="none",
        plot_title=element_text(size=13, weight="bold"),
        plot_subtitle=element_text(size=10),
        plot_caption=element_text(size=8, alpha=0.75),
    )


def stretched_demos(folder):
    """Each demo's elbow angle, resampled onto the same 0-100% timeline. Shape: demos x TIME_STEPS."""
    timeline = np.linspace(0, 1, TIME_STEPS)
    demos = []
    for path in sorted((DATASETS / folder / "trajectories").glob("episode_*_joints.csv")):
        angle = pd.read_csv(path)[JOINT].to_numpy()
        own_time = np.linspace(0, 1, len(angle))
        demos.append(np.interp(timeline, own_time, angle))
    return np.array(demos)


def heat_rows(title, demos, angles):
    """One row per heatmap cell: % through the demo, angle, and the share of demos there."""
    rows = []
    for step in range(TIME_STEPS):
        bumps = np.exp(-0.5 * ((angles[:, None] - demos[None, :, step]) / SIGMA) ** 2)
        bumps /= SIGMA * np.sqrt(2 * np.pi)
        density = bumps.mean(axis=1) * ANGLE_STEP     # share of demos in each angle row
        for angle, share in zip(angles, density):
            if share >= FAINT:
                rows.append({"panel": title, "percent": 100 * step / (TIME_STEPS - 1),
                             "angle": angle, "share": share})
    return rows


def chart_agreement():
    all_demos = {title: stretched_demos(folder) for title, folder in PANELS}
    low = min(d.min() for d in all_demos.values()) - 3 * SIGMA
    high = max(d.max() for d in all_demos.values()) + 3 * SIGMA
    angles = np.arange(low, high, ANGLE_STEP)

    rows = []
    for title, demos in all_demos.items():
        rows += heat_rows(title, demos, angles)
        # Spread: average over the timeline of the standard deviation across demos
        print(f"{title}: spread {demos.std(axis=0).mean():.1f} deg")
    frame = pd.DataFrame(rows)
    frame["panel"] = pd.Categorical(frame["panel"], [title for title, _ in PANELS])

    plot = (
        ggplot(frame, aes("percent", "angle", fill="share"))
        + geom_raster()
        + facet_wrap("panel", ncol=2)
        + scale_fill_gradientn(colors=COLOURS, limits=(FAINT, CAP), oob=squish)
        + scale_x_continuous(breaks=[0, 50, 100], labels=["start", "halfway", "end"])
        + labs(
            x="", y="Elbow angle (degrees)",
            title="From noise to a clear path",
            subtitle="Where the elbow was, across every demo, at each moment of the move. Brighter = more demos agree",
            caption="Each demo stretched onto the same start-to-end timeline",
        )
        + site_theme()
    )
    plot.save(OUT / "agreement.png", width=8, height=6, dpi=160, verbose=False)
    print("saved agreement.png")


if __name__ == "__main__":
    chart_agreement()
