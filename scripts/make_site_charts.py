#!/usr/bin/env python3
"""
Draw the four charts for the website's Anomaly Scores page with plotnine (ggplot for Python).

Every number comes from a file in this repo:
  1. trigger.png     - datasets/90_degree_turn/trajectories/episode_00_joints.csv
  2. cameras.png     - anomalib/test_2/Test 2.1/logs/*.log
  3. ep05.png        - anomalib/test_3/logs/side_view.log and side_view_no_ep05.log
  4. threshold.png   - anomalib/test_3/logs/side_view_no_ep05.log and anomalib/test_4/scores.csv

Run with the site-charts venv (plotnine needs pandas 3, which breaks anomalib):
    python scripts/make_site_charts.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
from plotnine import (
    aes, annotate, element_blank, element_line, element_rect, element_text, facet_wrap,
    geom_hline, geom_line, geom_vline, geom_point, geom_rect, geom_segment, geom_text, ggplot,
    labs, position_jitter, scale_color_manual, scale_shape_manual, scale_x_continuous, scale_x_discrete,
    scale_y_continuous, scale_y_discrete, theme, theme_minimal,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "assets" / "charts"
sys.path.insert(0, str(REPO / "algorithm"))
from reset_pose_trigger import ResetPoseTrigger, load_reset_pose  # noqa: E402

# Colours checked with the dataviz palette validator against the page's blue (all checks pass)
SURFACE = "#0a238f"
GOOD = "#1fae82"
DEFECT = "#dc6f14"
LINE = "#ffe08a"   # threshold lines, the site's yellow
INK = "#ffffff"
COLOURS = {"good": GOOD, "defect": DEFECT}
SHAPES = {"good": "o", "defect": "^"}   # shape as well as colour, so colour is never the only cue
THRESHOLD = 0.37


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
        legend_position="none",
        plot_title=element_text(size=13, weight="bold"),
        plot_subtitle=element_text(size=10),
        plot_caption=element_text(size=8, alpha=0.75),
    )


def read_log_scores(log_path):
    """Read the per-photo score lines that train_anomalib_patchcore.py writes to its log."""
    rows = []
    pattern = re.compile(r"\|\s+(\S+\.png)\s+(Normal|Abnormal)\s+(Normal|Abnormal)\s+([0-9.]+)")
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            name, truth, _, score = match.groups()
            label = "good" if truth == "Normal" else "defect"
            rows.append({"photo": name.replace(".png", ""), "label": label, "score": float(score)})
    return pd.DataFrame(rows)


def save(plot, name, width, height):
    plot.save(OUT / name, width=width, height=height, dpi=160, verbose=False)
    print(f"saved {name}")


# ---------------------------------------------------------------------------------------------
# 1. The trigger firing on episode 00
# ---------------------------------------------------------------------------------------------
def chart_trigger():
    pose = load_reset_pose(REPO / "algorithm" / "reset_pose.json")
    checked = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
    frames = pd.read_csv(REPO / "datasets/90_degree_turn/trajectories/episode_00_joints.csv")

    trigger = ResetPoseTrigger(pose)
    distances = []
    photo_frames = []
    armed_frame = None
    for _, row in frames.iterrows():
        joints = {j: row[f"state_{j}"] for j in pose}
        d = max(abs(joints[j] - pose[j]) for j in checked)
        distances.append(d)
        if armed_frame is None and d > 15:
            armed_frame = int(row["frame_index"])
        if trigger.update(joints):
            photo_frames.append(int(row["frame_index"]))
    frames["distance"] = distances
    frames["seconds"] = frames["frame_index"] / 30

    photo = frames[frames["frame_index"] == photo_frames[0]].iloc[0]
    armed = frames[frames["frame_index"] == armed_frame].iloc[0]

    plot = (
        ggplot(frames, aes("seconds", "distance"))
        + geom_rect(aes(xmin=0, xmax=frames["seconds"].max(), ymin=0, ymax=10),
                    data=pd.DataFrame([{}]), inherit_aes=False, fill=GOOD, alpha=0.35)
        + geom_hline(yintercept=15, color=LINE, linetype="dashed", size=0.6)
        + geom_line(color=INK, size=0.7)
        + geom_point(data=pd.DataFrame([photo]), color=LINE, size=4)
        + geom_point(data=pd.DataFrame([armed]), color=INK, size=3)
        + annotate("text", x=photo["seconds"] - 0.5, y=photo["distance"] + 22, ha="right", color=LINE, size=10,
                   label=f"photo taken: {photo['seconds']:.1f} s\n(5 frames inside the band)")
        + annotate("text", x=armed["seconds"] + 0.6, y=armed["distance"] + 6, ha="left", color=INK, size=9,
                   label="armed: arm has left X")
        + annotate("text", x=frames["seconds"].max() * 0.5, y=5, color=INK, size=9,
                   label="within ±10 of reset pose X")
        + annotate("text", x=frames["seconds"].max() * 0.5, y=19, color=LINE, size=9,
                   label="must go beyond 15 to arm again")
        + scale_y_continuous(limits=(0, None))
        + labs(x="seconds into the episode", y="distance from reset pose X",
               title="The trigger takes one photo per turn, timed from the joints",
               subtitle="Turn episode 00: the arm's largest joint distance from X, frame by frame",
               caption="Source: datasets/90_degree_turn episode 00 · algorithm/reset_pose_trigger.py")
        + site_theme()
    )
    save(plot, "trigger.png", 8, 4.2)


# ---------------------------------------------------------------------------------------------
# 2. Choosing the camera (Test 2.1)
# ---------------------------------------------------------------------------------------------
def chart_cameras():
    parts = []
    names = {"claw_cam": "Claw camera", "top_view": "Top camera", "side_view": "Side camera"}
    for camera, title in names.items():
        df = read_log_scores(REPO / "anomalib/test_2/Test 2.1/logs" / f"{camera}.log")
        df["camera"] = title
        parts.append(df)
    scores = pd.concat(parts)
    scores["camera"] = pd.Categorical(scores["camera"], categories=list(names.values()))

    plot = (
        ggplot(scores, aes("label", "score", color="label", shape="label"))
        + geom_hline(yintercept=0.5, color=LINE, linetype="dashed", size=0.6)
        + geom_point(size=3, alpha=0.9, position=position_jitter(width=0.15, height=0, random_state=1))
        + facet_wrap("camera")
        + scale_color_manual(values=COLOURS)
        + scale_shape_manual(values=SHAPES)
        + scale_x_discrete(limits=["good", "defect"])
        + labs(x="", y="anomaly score (each camera's own 0–1 scale)",
               title="Side camera: good and defect apart. Claw camera: mixed up",
               subtitle="Test 2.1: 5 good and 7 defect photos per camera. Dashed line: each camera's threshold",
               caption="Source: anomalib/test_2/Test 2.1/logs")
        + site_theme()
    )
    save(plot, "cameras.png", 8, 4.2)


# ---------------------------------------------------------------------------------------------
# 3. One bad training photo (Test 3, with and without ep05)
# ---------------------------------------------------------------------------------------------
def chart_ep05():
    before = read_log_scores(REPO / "anomalib/test_3/logs/side_view.log")
    after = read_log_scores(REPO / "anomalib/test_3/logs/side_view_no_ep05.log")
    before["run"] = "with ep05"
    after["run"] = "without ep05"
    scores = pd.concat([before, after])
    scores = scores[scores["photo"] != "theo_v2_ep33"]   # dot not visible from the side camera

    runs = ["with ep05 in training", "without ep05"]
    scores["run"] = scores["run"].map({"with ep05": runs[0], "without ep05": runs[1]})
    scores["run"] = pd.Categorical(scores["run"], categories=runs)

    # The empty band in each run: from the highest-scoring good photo up to the lowest-scoring defect
    bands = []
    for run in runs:
        run_scores = scores[scores["run"] == run]
        highest_good = run_scores[run_scores["label"] == "good"]["score"].max()
        lowest_defect = run_scores[run_scores["label"] == "defect"]["score"].min()
        bands.append({"run": run, "low": highest_good, "high": lowest_defect,
                      "mid": (highest_good + lowest_defect) / 2,
                      "text": f"gap {lowest_defect - highest_good:.3f}"})
    bands = pd.DataFrame(bands)
    bands["run"] = pd.Categorical(bands["run"], categories=runs)

    plot = (
        ggplot(scores, aes("label", "score", color="label", shape="label"))
        + geom_rect(aes(xmin=0.4, xmax=2.6, ymin="low", ymax="high"), data=bands,
                    inherit_aes=False, fill=LINE, alpha=0.25)
        + geom_text(aes(x=1.5, y="mid", label="text"), data=bands, inherit_aes=False,
                    color=LINE, size=11, fontweight="bold")
        + geom_point(size=3, alpha=0.9, position=position_jitter(width=0.15, height=0, random_state=3))
        + facet_wrap("run")
        + scale_color_manual(values=COLOURS)
        + scale_shape_manual(values=SHAPES)
        + scale_x_discrete(limits=["good", "defect"])
        + labs(x="", y="anomaly score (each run's own 0-1 scale)",
               title="Removing one mislabelled photo doubled the gap",
               subtitle="Test 3, same 23 test photos. ep05, a dotted block, had been in the 'good' training set",
               caption="Compare the shaded gaps, not single dots: each run rescales its own scores · "
                       "Source: anomalib/test_3/logs · ep33 left out (dot faces away)")
        + site_theme()
    )
    save(plot, "ep05.png", 8, 4.6)


# ---------------------------------------------------------------------------------------------
# 4. Anomalib's default line (0.5) against ours (0.37), on the tuning photos and on new photos
# ---------------------------------------------------------------------------------------------
DEFAULT_LINE = 0.5


def chart_threshold():
    sets = ["Photos we tuned on (Test 3)", "New photos, never seen (Test 4)"]
    test3 = read_log_scores(REPO / "anomalib/test_3/logs/side_view_no_ep05.log")
    test3 = test3[test3["photo"] != "theo_v2_ep33"].copy()   # dot faces away from the camera
    test3["set"] = sets[0]
    test4 = pd.read_csv(REPO / "anomalib/test_4/scores.csv")
    test4["photo"] = test4["photo"].str.replace(".png", "", regex=False)
    test4 = test4[["photo", "label", "score"]].copy()
    test4["set"] = sets[1]
    scores = pd.concat([test3, test4])
    scores["set"] = pd.Categorical(scores["set"], categories=sets)

    # Print the results under each line so the page's table can be checked against them
    for name in sets:
        part = scores[scores["set"] == name]
        for line in (DEFAULT_LINE, THRESHOLD):
            caught = ((part["label"] == "defect") & (part["score"] > line)).sum()
            passed = ((part["label"] == "good") & (part["score"] <= line)).sum()
            print(f"  {name}, line {line}: dots caught {caught}/{(part['label'] == 'defect').sum()}, "
                  f"good passed {passed}/{(part['label'] == 'good').sum()}")

    # Rows: good at y = 1, defect at y = 2. Each photo gets a small fixed vertical offset (cycling
    # through 5 positions) so dots with similar scores don't sit on top of each other, and so the
    # rings below can be drawn exactly on the dots they mark.
    scores = scores.sort_values(["set", "label", "score"]).reset_index(drop=True)
    offsets = [-0.12, 0.06, -0.03, 0.12, -0.08]
    scores["y"] = scores["label"].map({"good": 1, "defect": 2}).astype(float)
    scores["y"] = scores["y"] + [offsets[i % 5] for i in range(len(scores))]

    rings = scores[scores["photo"].isin(["theo_v2_ep34", "theo_v2_ep04", "theo_ep04"])].copy()
    notes = rings.copy()
    notes["text"] = notes["photo"].map({
        "theo_v2_ep34": "on the default line: missed by 0.5",
        "theo_v2_ep04": "red block: missed by 0.5, caught by 0.37",
        "theo_ep04": "faint mark, flagged in advance",
    })
    notes["ha"] = notes["photo"].map({"theo_v2_ep34": "right", "theo_v2_ep04": "right", "theo_ep04": "left"})
    notes["x"] = notes["score"] + notes["ha"].map({"right": -0.03, "left": 0.03})

    line_labels = pd.DataFrame([
        {"set": sets[0], "x": THRESHOLD - 0.01, "text": "ours: 0.37", "ha": "right"},
        {"set": sets[0], "x": DEFAULT_LINE + 0.01, "text": "default: 0.5", "ha": "left"},
    ])
    line_labels["set"] = pd.Categorical(line_labels["set"], categories=sets)

    plot = (
        ggplot(scores, aes("score", "y", color="label", shape="label"))
        + geom_vline(xintercept=DEFAULT_LINE, color=INK, linetype="dotted", size=0.8, alpha=0.8)
        + geom_vline(xintercept=THRESHOLD, color=LINE, size=1.1)
        + geom_point(size=3.2, alpha=0.9)
        + geom_point(data=rings, size=7, shape="o", fill="none", color=INK, stroke=1.2)
        + geom_text(aes(x="x", y="y", label="text", ha="ha"), data=notes, color=INK, size=8.5,
                    inherit_aes=False)
        + geom_text(aes(x="x", label="text", ha="ha"), y=1.5, data=line_labels, color=LINE, size=9,
                    inherit_aes=False, fontweight="bold")
        + facet_wrap("set", ncol=1)
        + scale_color_manual(values=COLOURS)
        + scale_shape_manual(values=SHAPES)
        + scale_y_continuous(breaks=[1, 2], labels=["good", "defect"], limits=(0.7, 2.3))
        + scale_x_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1])
        + labs(x="score: how unusual the block looks  →  right of the line = defect", y="",
               title="Moving the line from 0.5 to 0.37 caught every dot",
               subtitle="Each dot is one photo. Ringed: the photos where the choice of line matters",
               caption="Source: anomalib/test_3/logs · anomalib/test_4/scores.csv · "
                       "one Test 3 photo left out (its dot faces away from the camera)")
        + site_theme()
    )
    save(plot, "threshold.png", 8, 5.2)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    chart_trigger()
    chart_cameras()
    chart_ep05()
    chart_threshold()
