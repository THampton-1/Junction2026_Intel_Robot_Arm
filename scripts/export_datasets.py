#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Extracts complete multi-camera videos, joint trajectory CSVs, combined Parquet telemetry,
and starting frames for all episodes in Block_To_Green_v4 and Block_To_Pink_v4.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATASETS = [
    ("Block_To_Green_v4", "a531500b-01d8-496e-9d3e-6bee809b38c1", "green_v4"),
    ("Block_To_Pink_v4", "80b4f232-a586-4416-a098-fb93752a6175", "pink_v4"),
    ("Block_To_Pink_v5", "821972b1-430d-46b1-82bb-0646691c85fd", "pink_v5"),
]

REPO_ROOT = Path("/home/intel/Junction")
LOCAL_ANOMALIB = Path("/home/intel/anomalib_data")
CAMERAS = ["claw_cam", "side_view", "top_view"]
JOINT_COLS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def process_dataset(name: str, ds_id: str, prefix: str):
    print(f"\n==========================================")
    print(f" Processing {name} ({ds_id})")
    print(f"==========================================")
    ds_dir = Path(f"/home/intel/.local/share/physicalai/datasets/{ds_id}")

    # Load all episode metadata
    meta_pqs = sorted((ds_dir / "meta/episodes/chunk-000").glob("*.parquet"))
    df_ep = pd.concat([pq.read_table(f).to_pandas() for f in meta_pqs], ignore_index=True)
    df_ep.sort_values(by="episode_index", inplace=True)
    df_ep.reset_index(drop=True, inplace=True)

    # Load all joint / state data
    data_pqs = sorted((ds_dir / "data/chunk-000").glob("*.parquet"))
    df_data = pd.concat([pq.read_table(f).to_pandas() for f in data_pqs], ignore_index=True)
    df_data.sort_values(by=["episode_index", "frame_index"], inplace=True)
    df_data.reset_index(drop=True, inplace=True)

    print(f"Total episodes: {len(df_ep)}, total data frames: {len(df_data)}")

    # Setup repo directories
    ds_repo = REPO_ROOT / "datasets" / name.lower()
    vid_repo = ds_repo / "videos"
    traj_repo = ds_repo / "trajectories"
    anom_repo = REPO_ROOT / "anomalib" / name.lower()
    anom_local = LOCAL_ANOMALIB / name.lower()

    for d in [traj_repo, vid_repo, anom_repo, anom_local]:
        d.mkdir(parents=True, exist_ok=True)
    for cam in CAMERAS:
        (anom_repo / cam).mkdir(parents=True, exist_ok=True)
        (anom_local / cam).mkdir(parents=True, exist_ok=True)

    # 1. Export Parquet of all episodes
    all_parquet_path = traj_repo / "all_episodes_joints.parquet"
    df_data.to_parquet(all_parquet_path, index=False)
    print(f"Exported combined parquet ({len(df_data)} frames) -> {all_parquet_path}")

    # 2. Export per-episode CSV trajectories
    for ep in sorted(df_ep["episode_index"].unique()):
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

        csv_path = traj_repo / f"episode_{ep:02d}_joints.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  [Ep {ep:02d}] Exported {len(rows)} joint telemetry rows -> {csv_path.name}")

    # 3. Export multi-camera videos and Anomalib starting frames
    for _, row in df_ep.iterrows():
        ep = int(row["episode_index"])
        ep_vid_dir = vid_repo / f"episode_{ep:02d}"
        ep_vid_dir.mkdir(parents=True, exist_ok=True)

        for cam in CAMERAS:
            file_idx = int(row[f"videos/observation.images.{cam}/file_index"])
            from_ts = float(row[f"videos/observation.images.{cam}/from_timestamp"])
            to_ts = float(row[f"videos/observation.images.{cam}/to_timestamp"])

            in_mp4 = ds_dir / f"videos/observation.images.{cam}/chunk-000/file-{file_idx:03d}.mp4"

            # Starting frame for anomalib
            frame_name = f"{prefix}_ep{ep:02d}.png"
            repo_frame = anom_repo / cam / frame_name
            local_frame = anom_local / cam / frame_name

            cmd_frame = [
                "ffmpeg", "-ss", f"{from_ts:.3f}", "-i", str(in_mp4),
                "-vframes", "1", "-update", "1", "-q:v", "2", str(repo_frame), "-y"
            ]
            subprocess.run(cmd_frame, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            shutil.copy2(repo_frame, local_frame)

            # Sliced video clip
            out_clip = ep_vid_dir / f"{cam}.mp4"
            cmd_vid = [
                "ffmpeg", "-ss", f"{from_ts:.3f}", "-to", f"{to_ts:.3f}",
                "-i", str(in_mp4),
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                str(out_clip), "-y"
            ]
            subprocess.run(cmd_vid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        print(f"  [Ep {ep:02d}] Extracted 3 camera videos and starting frames")


def main():
    for name, ds_id, prefix in DATASETS:
        process_dataset(name, ds_id, prefix)
    print("\nAll datasets successfully exported and verified!")


if __name__ == "__main__":
    main()
