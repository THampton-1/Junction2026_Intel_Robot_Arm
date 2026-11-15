#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Batch exporter for all datasets recorded in Physical AI Studio.
Extracts sliced multi-camera MP4 video clips, 6-DOF joint trajectories (CSV and Parquet),
and generates CLI trajectory player tools and documentation.
Uses multi-threading to accelerate ffmpeg video slicing.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDIO_DATASETS_DIR = Path.home() / ".local/share/physicalai/datasets"
CAMERAS = ["claw_cam", "side_view", "top_view"]
JOINT_COLS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

ALL_STUDIO_DATASETS = [
    ("90_degree_turn", "68d24abc-c87e-4009-8ab4-74b90d39c2dd", "90 Degree Turn"),
    ("block_to_green_v4", "a531500b-01d8-496e-9d3e-6bee809b38c1", "Block to Green v4"),
    ("block_to_pink_v4", "80b4f232-a586-4416-a098-fb93752a6175", "Block to Pink v4"),
    ("block_to_pink_v5", "821972b1-430d-46b1-82bb-0646691c85fd", "Block to Pink v5"),
    ("theo_frames_to_crop", "a38ba05b-77bf-4041-ae3d-61dc2cf74fbb", "Theo Frames to Crop"),
    ("theo_frames_to_crop_v2", "4e822e47-3a91-4817-9e3d-fc7e34a0625f", "Theo Frames to Crop v2"),
    ("photo_test_set", "06f21646-a3c2-4482-908c-41cba2c3e6e7", "Photo Test Set"),
    ("old_block_to_green", "86cca34a-74c8-4409-b045-409b01e66c46", "Old Block to Green"),
    ("block_to_pink", "d9b531e6-72c5-4174-90ab-299100d158b0", "Block to Pink"),
    ("block_to_green", "ea625a1c-d316-4a7b-88f0-43838fbf9364", "Block to Green"),
    ("block_to_green_v2", "f8a06ac7-bf26-4052-b267-b048d9ff6e49", "Block to Green v2"),
    ("grip_master_v1", "5869efcd-30b2-421b-b564-2030ee4ef22f", "Grip Master v1"),
    ("block_to_green_v3", "aeb3860c-12d7-42b3-9819-00109e297064", "Block to Green v3"),
    ("block_to_pink_v3", "4f232589-2729-44e7-94bd-cedafbe31164", "Block to Pink v3"),
]


TRAJECTORY_PLAYER_TEMPLATE = '''#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Joint Trajectory Player and Query Utility for {title} Dataset.
Provides programmatic access to 6-DOF joint positions over the entire duration of the clips.
Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class JointTrajectoryDataset:
    """Manages joint trajectory data for the {title} episodes."""

    def __init__(self, data_dir: Path | str | None = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "trajectories"
        self.data_dir = Path(data_dir)
        self.episodes: dict[int, pd.DataFrame] = {}
        self._load_all_episodes()

    def _load_all_episodes(self):
        csv_files = sorted(self.data_dir.glob("episode_*_joints.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No joint trajectory CSVs found in {{self.data_dir}}")
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            ep_idx = int(df["episode_index"].iloc[0])
            self.episodes[ep_idx] = df

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def get_episode(self, episode_index: int) -> pd.DataFrame:
        if episode_index not in self.episodes:
            raise KeyError(f"Episode {{episode_index}} not found. Available: {{list(self.episodes.keys())}}")
        return self.episodes[episode_index]

    def get_states(self, episode_index: int) -> np.ndarray:
        df = self.get_episode(episode_index)
        cols = [f"state_{{j}}" for j in JOINT_NAMES]
        return df[cols].to_numpy()

    def get_actions(self, episode_index: int) -> np.ndarray:
        df = self.get_episode(episode_index)
        cols = [f"action_{{j}}" for j in JOINT_NAMES]
        return df[cols].to_numpy()

    def print_summary(self, episode_index: int):
        df = self.get_episode(episode_index)
        dur = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        n_frames = len(df)
        print("=" * 60)
        print(f"  {title.upper()} &mdash; EPISODE {{episode_index:02d}} SUMMARY")
        print("=" * 60)
        print(f"Total Frames:      {{n_frames}}")
        print(f"Duration:          {{dur:.2f}} s (~{{dur/60:.2f}} min)")
        print(f"Est. Frame Rate:   {{n_frames/dur:.1f}} fps")
        print("-" * 60)
        for j in JOINT_NAMES:
            s_col = f"state_{{j}}"
            s_min, s_max = df[s_col].min(), df[s_col].max()
            s_init, s_end = df[s_col].iloc[0], df[s_col].iloc[-1]
            print(f"  {{j:<14}} init: {{s_init:7.2f}}  end: {{s_end:7.2f}}  range: [{{s_min:7.2f}}, {{s_max:7.2f}}]")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Query and replay {title} joint positions")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--summary-all", action="store_true", help="Print summary for all episodes")
    args = parser.parse_args()

    player = JointTrajectoryDataset()
    if args.summary_all:
        for ep in sorted(player.episodes.keys()):
            player.print_summary(ep)
    else:
        player.print_summary(args.episode)


if __name__ == "__main__":
    main()
'''


README_TEMPLATE = '''# {title}: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstrations for `{title}` recorded on the LeRobot SO-101 robotic arm in Physical AI Studio.

## Dataset Specifications
* **Dataset ID:** `{ds_id}`
* **Episodes:** {num_episodes} (`episode_00` through `episode_{max_ep:02d}`)
* **Total Frames:** {total_frames:,} synchronized video and state frames
* **Total Duration:** {total_duration:.1f} seconds (~{total_duration_min:.1f} minutes)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
* **`code/joint_trajectory_player.py`**: Python CLI tool to inspect, summarize, replay, and query joint positions.

## Python API Usage
```bash
# Print summary for all episodes:
python3 datasets/{name_lower}/code/joint_trajectory_player.py --summary-all

# Query specific episode joint positions:
python3 datasets/{name_lower}/code/joint_trajectory_player.py --episode 0
```
'''


def slice_video_task(in_mp4: Path, out_clip: Path, from_ts: float, to_ts: float) -> str:
    if out_clip.exists() and out_clip.stat().st_size > 1000:
        return f"SKIP {out_clip.name}"

    out_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd_vid = [
        "ffmpeg", "-ss", f"{from_ts:.3f}", "-to", f"{to_ts:.3f}",
        "-i", str(in_mp4),
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        str(out_clip), "-y"
    ]
    subprocess.run(cmd_vid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return f"DONE {out_clip.name}"


def process_dataset(name_lower: str, ds_id: str, title: str, num_workers: int = 8):
    ds_dir = STUDIO_DATASETS_DIR / ds_id
    if not ds_dir.exists():
        print(f"[WARN] Dataset folder not found: {ds_dir}")
        return

    print(f"\n=======================================================")
    print(f" Processing {title} ({ds_id})")
    print(f"=======================================================")

    # 1. Load episode metadata
    meta_pqs = sorted((ds_dir / "meta/episodes").glob("*/*.parquet"))
    if not meta_pqs:
        print(f"[ERROR] No episode metadata in {ds_dir}")
        return
    df_ep = pd.concat([pq.read_table(f).to_pandas() for f in meta_pqs], ignore_index=True)
    df_ep.sort_values(by="episode_index", inplace=True)
    df_ep.drop_duplicates(subset=["episode_index"], inplace=True)
    df_ep.reset_index(drop=True, inplace=True)

    # 2. Load joint / action data
    data_pqs = sorted((ds_dir / "data").glob("*/*.parquet"))
    if not data_pqs:
        print(f"[ERROR] No telemetry data in {ds_dir}")
        return
    df_data = pd.concat([pq.read_table(f).to_pandas() for f in data_pqs], ignore_index=True)
    df_data.sort_values(by=["episode_index", "frame_index"], inplace=True)
    df_data.reset_index(drop=True, inplace=True)

    num_episodes = len(df_ep)
    total_frames = len(df_data)
    print(f"Total episodes: {num_episodes}, total data frames: {total_frames}")

    # Setup directories
    ds_repo = REPO_ROOT / "datasets" / name_lower
    vid_repo = ds_repo / "videos"
    traj_repo = ds_repo / "trajectories"
    code_repo = ds_repo / "code"

    for d in [traj_repo, vid_repo, code_repo]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Export all_episodes_joints.parquet
    parquet_path = traj_repo / "all_episodes_joints.parquet"
    if not parquet_path.exists():
        df_data.to_parquet(parquet_path, index=False)
        print(f"  -> Exported combined parquet: {parquet_path.name}")

    # 4. Export per-episode CSV trajectories
    for ep in sorted(df_ep["episode_index"].unique()):
        csv_path = traj_repo / f"episode_{ep:02d}_joints.csv"
        if not csv_path.exists():
            ep_data = df_data[df_data["episode_index"] == ep].copy()
            states = np.array(ep_data["observation.state"].tolist())
            actions = np.array(ep_data["action"].tolist())

            rows = []
            for i in range(len(ep_data)):
                r = {
                    "episode_index": int(ep),
                    "frame_index": int(ep_data["frame_index"].iloc[i]),
                    "timestamp": float(ep_data["timestamp"].iloc[i]),
                }
                for j, jname in enumerate(JOINT_COLS):
                    r[f"state_{jname}"] = float(states[i, j])
                    r[f"action_{jname}"] = float(actions[i, j])
                rows.append(r)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"  -> Verified/exported {num_episodes} CSV trajectories")

    # 5. Export multi-camera videos in parallel
    tasks = []
    total_dur_sec = 0.0

    for _, row in df_ep.iterrows():
        ep = int(row["episode_index"])
        ep_vid_dir = vid_repo / f"episode_{ep:02d}"

        for cam in CAMERAS:
            c_col = f"videos/observation.images.{cam}/chunk_index"
            f_col = f"videos/observation.images.{cam}/file_index"
            from_col = f"videos/observation.images.{cam}/from_timestamp"
            to_col = f"videos/observation.images.{cam}/to_timestamp"

            if from_col not in row or to_col not in row or pd.isna(row[from_col]) or pd.isna(row[to_col]):
                continue

            chunk_idx = int(row[c_col]) if c_col in row and not pd.isna(row[c_col]) else 0
            file_idx = int(row[f_col]) if f_col in row and not pd.isna(row[f_col]) else 0
            from_ts = float(row[from_col])
            to_ts = float(row[to_col])

            if cam == "top_view":
                total_dur_sec += (to_ts - from_ts)

            in_mp4 = ds_dir / f"videos/observation.images.{cam}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
            out_clip = ep_vid_dir / f"{cam}.mp4"
            tasks.append((in_mp4, out_clip, from_ts, to_ts))

    print(f"  -> Slicing {len(tasks)} video clips with {num_workers} parallel workers...")
    done_count, skip_count = 0, 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(slice_video_task, *t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if "DONE" in res:
                done_count += 1
            else:
                skip_count += 1

    print(f"  -> Video extraction complete: {done_count} newly encoded, {skip_count} already existed")

    # 6. Generate code/joint_trajectory_player.py
    player_py = code_repo / "joint_trajectory_player.py"
    if not player_py.exists():
        base_player = (REPO_ROOT / "datasets/block_to_green_v4/code/joint_trajectory_player.py").read_text()
        player_code = base_player.replace("Block to Green v4", title).replace("BLOCK TO GREEN V4", title.upper()).replace("block to green v4", title.lower())
        player_py.write_text(player_code)
        player_py.chmod(0o755)

    # 7. Generate README.md
    readme_md = ds_repo / "README.md"
    if not readme_md.exists():
        max_ep = max(df_ep["episode_index"])
        readme_md.write_text(README_TEMPLATE.format(
            title=title,
            name_lower=name_lower,
            ds_id=ds_id,
            num_episodes=num_episodes,
            max_ep=max_ep,
            total_frames=total_frames,
            total_duration=total_dur_sec,
            total_duration_min=total_dur_sec / 60.0
        ))


def main():
    parser = argparse.ArgumentParser(description="Export all datasets from Physical AI Studio to repo")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel ffmpeg workers")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by specific dataset name")
    args = parser.parse_args()

    t_start = time.time()
    for name_lower, ds_id, title in ALL_STUDIO_DATASETS:
        if args.dataset and args.dataset.lower() not in name_lower:
            continue
        process_dataset(name_lower, ds_id, title, num_workers=args.workers)

    t_elapsed = time.time() - t_start
    print(f"\n=======================================================")
    print(f" ALL DATASETS SUCCESSFULLY EXPORTED IN {t_elapsed:.1f} SECONDS")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
