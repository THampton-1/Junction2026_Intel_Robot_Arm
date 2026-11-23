#!/usr/bin/env python3
r"""
Draw the chart for the website's Imitation Learning page with plotnine (ggplot for Python).

demos.png - every teleop demonstration of the two sorting skills, by dataset version: how long
            each demo lasted, and how jerky the arm's movement was.
            Source: datasets/<dataset>/trajectories/episode_*_joints.csv

"Jerkiness" is the average size of the frame-to-frame change in joint speed, summed over the six
joints: mean over frames of sum_j |a_j[t+1] - 2 a_j[t] + a_j[t-1]|, in degrees per frame squared,
where a_j is the leader arm's commanded angle (the action_* columns). Lower means smoother.

Run with the site-charts venv:
    C:SERS	HEOHENVSSITE-CHARTSSCRIPTSPYTHON.EXE scripts/make_learning_charts.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes, element_blank, element_line, element_rect, element_text, facet_grid, geom_point,
    ggplot, labs, position_jitter, stat_summary, theme, theme_minimal,
)

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets"
OUT = REPO / "docs" / "assets" / "charts"

# Same colours as make_site_charts.py: white ink and the site's yellow on the page's blue
SURFACE = "#0a238f"
INK = "#ffffff"
LINE = "#ffe08a"
FPS = 30

# The sorting datasets in version order. There is no exported pink v2.
VERSIONS = [
    ("Drop in green", "v1", "block_to_green"),
    ("Drop in green", "v2", "block_to_green_v2"),
    ("Drop in green", "v3", "block_to_green_v3"),
    ("Drop in green", "v4", "block_to_green_v4"),
    ("Drop in pink", "v1", "block_to_pink"),
    ("Drop in pink", "v3", "block_to_pink_v3"),
    ("Drop in pink", "v4", "block_to_pink_v4"),
    ("Drop in pink", "v5", "block_to_pink_v5"),
]


def site_theme():
    """White text on the page's blue, faint grid, no chart junk."""
    return theme_minimal(base_size=11) + theme(
        plot_background=element_rect(fill=SURFACE, color=SURFACE),
        panel_background=element_rect(fill=SURFACE, color=SURFACE),
        text=element_text(color=INK),
        axis_text=element_text(color=INK),
        strip_text=element_text(color=INK, size=11, weight="bold"),
        panel_grid_major=element_line(color="#ffffff", alpha=0.15, size=0.4),
        panel_grid_minor=element_blank(),
        panel_grid_major_x=element_blank(),
        panel_spacing_y=0.06,
        panel_border=element_rect(color="#ffffff", fill=None, alpha=0.35),
        legend_position="none",
        plot_title=element_text(size=13, weight="bold"),
        plot_subtitle=element_text(size=10),
        plot_caption=element_text(size=8, alpha=0.75),
    )


def episode_rows():
    """One row per demonstration: its length in seconds and its jerkiness."""
    rows = []
    for task, version, folder in VERSIONS:
        files = sorted((DATASETS / folder / "trajectories").glob("episode_*_joints.csv"))
        for path in files:
            frames = pd.read_csv(path)
            action_columns = [c for c in frames.columns if c.startswith("action_")]
            angles = frames[action_columns].to_numpy()
            second_difference = np.diff(angles, n=2, axis=0)
            jerkiness = np.abs(second_difference).sum(axis=1).mean()
            rows.append({
                "task": task,
                "version": f"{version}\n{len(files)} demos",
                "Seconds per demo": len(frames) / FPS,
                "Jerkiness": jerkiness,
            })
    return pd.DataFrame(rows)


def chart_demos():
    wide = episode_rows()
    long = wide.melt(id_vars=["task", "version"], var_name="measure", value_name="value")
    long["measure"] = pd.Categorical(long["measure"], ["Seconds per demo", "Jerkiness"])

    # Print the per-version means so the page copy can quote them
    print(wide.groupby(["task", "version"], sort=False)[["Seconds per demo", "Jerkiness"]].mean().round(2))

    plot = (
        ggplot(long, aes("version", "value"))
        + geom_point(position=position_jitter(width=0.15, height=0, random_state=1), color=INK, alpha=0.55, size=1.6)
        + stat_summary(fun_y=np.mean, geom="point", shape="_", size=14, color=LINE, stroke=1.2)
        + facet_grid("measure ~ task", scales="free")
        + labs(
            x="", y="",
            title="Fewer demos, done slower and smoother",
            subtitle="Each dot is one demonstration; the yellow bar is the version's average",
            caption="Jerkiness: average frame-to-frame change in joint speed, summed over six joints (deg per frame²)",
        )
        + site_theme()
    )
    plot.save(OUT / "demos.png", width=8, height=5.6, dpi=160, verbose=False)
    print("saved demos.png")


if __name__ == "__main__":
    chart_demos()
