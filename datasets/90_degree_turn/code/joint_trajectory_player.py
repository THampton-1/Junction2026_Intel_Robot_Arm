#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Joint Trajectory Player and Query Utility for 90 Degree Turn Dataset.
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
    """Manages joint trajectory data for the 90 degree turn episodes."""

    def __init__(self, data_dir: Path | str | None = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "trajectories"
        self.data_dir = Path(data_dir)
        self.episodes: dict[int, pd.DataFrame] = {}
        self._load_all_episodes()

    def _load_all_episodes(self):
        csv_files = sorted(self.data_dir.glob("episode_*_joints.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No joint trajectory CSVs found in {self.data_dir}")
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            ep_idx = int(df["episode_index"].iloc[0])
            self.episodes[ep_idx] = df

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def get_episode(self, episode_idx: int) -> pd.DataFrame:
        if episode_idx not in self.episodes:
            raise ValueError(f"Episode {episode_idx} not found. Available: {list(self.episodes.keys())}")
        return self.episodes[episode_idx]

    def get_joint_positions_at_time(self, episode_idx: int, t_seconds: float) -> dict[str, float]:
        """Interpolates joint positions for a specific timestamp within the episode."""
        df = self.get_episode(episode_idx)
        times = df["timestamp"].values
        # Relative time offset within episode
        t_start = times[0]
        t_query = t_start + t_seconds
        
        t_query = np.clip(t_query, times[0], times[-1])
        result = {}
        for joint in JOINT_NAMES:
            state_vals = df[f"state_{joint}"].values
            interp_val = float(np.interp(t_query, times, state_vals))
            result[joint] = interp_val
        return result

    def summary(self, episode_idx: int | None = None):
        """Prints a statistical summary of joint angles for an episode or all episodes."""
        eps = [episode_idx] if episode_idx is not None else sorted(self.episodes.keys())
        print("=" * 80)
        print("  90 DEGREE TURN JOINT POSITION TELEMETRY SUMMARY")
        print("=" * 80)
        for ep in eps:
            df = self.episodes[ep]
            duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
            print(f"\n[Episode {ep:02d}] Duration: {duration:.2f}s | Total Frames: {len(df)}")
            print("-" * 80)
            print(f"{'Joint Name':<18} {'Min (deg)':<12} {'Mean (deg)':<12} {'Max (deg)':<12} {'Total Delta':<12}")
            print("-" * 80)
            for joint in JOINT_NAMES:
                vals = df[f"state_{joint}"].values
                min_v = vals.min()
                mean_v = vals.mean()
                max_v = vals.max()
                delta_v = vals[-1] - vals[0]
                print(f"{joint:<18} {min_v:<12.2f} {mean_v:<12.2f} {max_v:<12.2f} {delta_v:<12.2f}")
        print("=" * 80)

    def stream_replay(self, episode_idx: int, rate_hz: float = 30.0):
        """Simulates streaming joint positions at standard robot control frequency."""
        df = self.get_episode(episode_idx)
        print(f"Replaying Episode {episode_idx} ({len(df)} frames at {rate_hz} Hz)...")
        dt = 1.0 / rate_hz
        for i in range(len(df)):
            t0 = time.perf_counter()
            t_rel = df["timestamp"].iloc[i] - df["timestamp"].iloc[0]
            joints = [f"{joint}={df[f'state_{joint}'].iloc[i]:.1f}°" for joint in JOINT_NAMES]
            sys.stdout.write(f"\r[t={t_rel:5.2f}s | Frame {i:4d}] " + " | ".join(joints))
            sys.stdout.flush()
            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)
        print("\nReplay finished.")


def main():
    parser = argparse.ArgumentParser(description="Query and replay 90 degree turn joint positions")
    parser.add_argument("--episode", type=int, default=0, help="Episode index (0-9)")
    parser.add_argument("--summary-all", action="store_true", help="Print summary for all 10 episodes")
    parser.add_argument("--replay", action="store_true", help="Stream replay joint positions in terminal")
    parser.add_argument("--time-query", type=float, default=None, help="Query joint positions at specific elapsed seconds")
    args = parser.parse_args()

    dataset = JointTrajectoryDataset()

    if args.summary_all:
        dataset.summary()
    elif args.time_query is not None:
        joints = dataset.get_joint_positions_at_time(args.episode, args.time_query)
        print(f"Joint positions for Episode {args.episode} at t={args.time_query:.2f}s:")
        for name, val in joints.items():
            print(f"  {name:<16}: {val:6.2f}°")
    elif args.replay:
        dataset.stream_replay(args.episode)
    else:
        dataset.summary(args.episode)


if __name__ == "__main__":
    main()
