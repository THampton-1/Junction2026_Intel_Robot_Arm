#!/usr/bin/env python3
"""
Pull the inspection photos out of recorded episodes, using the same ResetPoseTrigger the
robot will run live.

For each episode it replays observation.state frame by frame through the trigger, and on
every frame the trigger fires it saves that frame from the chosen camera (plus an optional
fixed crop around the block).

Step 1 - find the reset pose X (median of each episode's first frame, which is the arm
sitting at reset before it moves):

    python algorithm/extract_reset_frames.py <dataset_dir> --estimate-reset-pose

    This writes algorithm/reset_pose.json. Check the numbers look right before step 2.

Step 2 - extract the photos:

    python algorithm/extract_reset_frames.py <dataset_dir> --out inspection_frames \
        --crop 40 110 70 70

Output: <out>/full/epXXX_photoN.png, <out>/crop/epXXX_photoN.png (if --crop), and
<out>/photos.csv listing episode, photo number, frame index and timestamp for each photo.
Photo 1 is the block as first squared; photo N is after N-1 clockwise rotations.

Needs: pandas + pyarrow (to read the parquet files) and the ffmpeg command-line tool
(to decode single frames from the AV1/H.264 episode videos).

Reads LeRobot dataset format v2.x (one video file per episode) and v3.x (many episodes
per video file, located through meta/episodes).
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd

from reset_pose_trigger import (
    CHECKED_JOINTS,
    EXIT_MARGIN,
    HOLD_FRAMES,
    TOLERANCE,
    ResetPoseTrigger,
    load_reset_pose,
    save_reset_pose,
    short_joint_name,
)

DEFAULT_RESET_POSE_FILE = Path(__file__).resolve().parent / "reset_pose.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="LeRobot dataset folder (contains meta/, data/, videos/)")
    parser.add_argument("--estimate-reset-pose", action="store_true", help="estimate X from first frames, save it, and stop")
    parser.add_argument("--reset-pose", type=Path, default=DEFAULT_RESET_POSE_FILE, help="JSON file holding X")
    parser.add_argument("--out", type=Path, default=Path("inspection_frames"), help="output folder")
    parser.add_argument("--camera", default="side_view", help="camera name, matched against the video feature names")
    parser.add_argument("--episodes", type=int, nargs="*", help="episode indices to process (default: all)")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="fixed crop box in pixels")
    parser.add_argument("--expected-photos", type=int, default=5, help="warn when an episode gives a different count")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--exit-margin", type=float, default=EXIT_MARGIN)
    parser.add_argument("--hold-frames", type=int, default=HOLD_FRAMES)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="path to the ffmpeg executable")
    return parser.parse_args()


# ---------------------------------------------------------------- reading the dataset


def load_info(dataset_dir):
    with open(dataset_dir / "meta" / "info.json") as f:
        return json.load(f)


def state_joint_names(info):
    names = info["features"]["observation.state"]["names"]
    if isinstance(names, dict):  # older datasets: {"motors": [...]}
        names = list(names.values())[0]
    return [short_joint_name(name) for name in names]


def load_states(dataset_dir):
    """All frames of all episodes, sorted by episode then frame."""
    files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not files:
        sys.exit(f"No parquet files under {dataset_dir / 'data'}")
    columns = ["episode_index", "frame_index", "timestamp", "observation.state"]
    frames = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    return frames.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def find_video_key(info, camera):
    video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    matches = [key for key in video_keys if camera in key]
    if len(matches) != 1:
        sys.exit(f"Camera '{camera}' matched {matches}; available video features: {video_keys}")
    return matches[0]


def load_episode_video_index(dataset_dir, info, video_key):
    """
    For v3 datasets: episode -> (video file, start time of that episode inside the file).
    For v2 datasets: returns None (each episode has its own video file starting at 0 s).
    """
    if not info["codebase_version"].startswith("v3"):
        return None
    files = sorted((dataset_dir / "meta" / "episodes").glob("**/*.parquet"))
    episodes = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    index = {}
    for _, row in episodes.iterrows():
        path = info["video_path"].format(
            video_key=video_key,
            chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
            file_index=int(row[f"videos/{video_key}/file_index"]),
        )
        index[int(row["episode_index"])] = (dataset_dir / path, float(row[f"videos/{video_key}/from_timestamp"]))
    return index


def video_location(dataset_dir, info, video_key, video_index, episode):
    """(video file, seconds into that file where this episode starts)."""
    if video_index is not None:
        return video_index[episode]
    path = info["video_path"].format(
        episode_chunk=episode // info["chunks_size"],
        video_key=video_key,
        episode_index=episode,
    )
    return dataset_dir / path, 0.0


# ---------------------------------------------------------------- frames out of videos


def save_frame(ffmpeg, video_path, seconds, fps, out_path, crop):
    # Seek half a frame early: ffmpeg returns the first frame at or after the seek time,
    # so this lands on the intended frame even if float rounding nudges the time up.
    seek = max(0.0, seconds - 0.5 / fps)
    command = [ffmpeg, "-v", "error", "-y", "-ss", f"{seek:.4f}", "-i", str(video_path), "-frames:v", "1"]
    if crop is not None:
        x, y, w, h = crop
        command += ["-vf", f"crop={w}:{h}:{x}:{y}"]
    command.append(str(out_path))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        sys.exit(f"ffmpeg failed on {video_path} at {seconds:.3f}s:\n{result.stderr}")


# ---------------------------------------------------------------- the two commands


def estimate_reset_pose(states, joint_names, out_path):
    first_frames = states[states["frame_index"] == 0]
    pose = {}
    for position, joint in enumerate(joint_names):
        values = [float(state[position]) for state in first_frames["observation.state"]]
        pose[joint] = round(statistics.median(values), 2)
        spread = max(values) - min(values)
        print(f"  {joint:15s} X = {pose[joint]:9.2f}   (range across {len(values)} episodes: {spread:.2f})")
    save_reset_pose(pose, out_path)
    print(f"Saved reset pose to {out_path}")
    print("If any checked joint's range is near or above the tolerance, the episodes did not")
    print("all start at the same reset pose - look at those episodes before trusting X.")


def find_photo_frames(episode_states, joint_names, reset_pose, args):
    trigger = ResetPoseTrigger(
        reset_pose,
        tolerance=args.tolerance,
        exit_margin=args.exit_margin,
        hold_frames=args.hold_frames,
    )
    photo_rows = []
    for _, row in episode_states.iterrows():
        joint_state = dict(zip(joint_names, (float(v) for v in row["observation.state"])))
        if trigger.update(joint_state):
            photo_rows.append(row)
    return photo_rows


def extract(dataset_dir, info, states, joint_names, args):
    reset_pose = load_reset_pose(args.reset_pose)
    missing = [joint for joint in CHECKED_JOINTS if joint not in joint_names]
    if missing:
        sys.exit(f"observation.state has no joints named {missing}; it has {joint_names}")

    video_key = find_video_key(info, args.camera)
    video_index = load_episode_video_index(dataset_dir, info, video_key)
    fps = info["fps"]

    (args.out / "full").mkdir(parents=True, exist_ok=True)
    if args.crop:
        (args.out / "crop").mkdir(parents=True, exist_ok=True)

    episodes = args.episodes if args.episodes else sorted(states["episode_index"].unique())
    manifest = []
    problems = []
    for episode in episodes:
        episode_states = states[states["episode_index"] == episode]
        if episode_states.empty:
            problems.append(f"episode {episode}: not in dataset")
            continue

        photo_rows = find_photo_frames(episode_states, joint_names, reset_pose, args)
        video_path, episode_start = video_location(dataset_dir, info, video_key, video_index, episode)

        for photo_number, row in enumerate(photo_rows, start=1):
            name = f"ep{episode:03d}_photo{photo_number}.png"
            seconds = episode_start + float(row["timestamp"])
            save_frame(args.ffmpeg, video_path, seconds, fps, args.out / "full" / name, None)
            if args.crop:
                save_frame(args.ffmpeg, video_path, seconds, fps, args.out / "crop" / name, args.crop)
            manifest.append([episode, photo_number, int(row["frame_index"]), round(float(row["timestamp"]), 4), name])

        print(f"episode {episode:3d}: {len(photo_rows)} photos at frames {[int(r['frame_index']) for r in photo_rows]}")
        if len(photo_rows) != args.expected_photos:
            problems.append(f"episode {episode}: {len(photo_rows)} photos, expected {args.expected_photos}")

    with open(args.out / "photos.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "photo", "frame_index", "timestamp", "file"])
        writer.writerows(manifest)

    print(f"\nSaved {len(manifest)} photos to {args.out}")
    if problems:
        print("\nCheck these episodes by eye:")
        for problem in problems:
            print("  " + problem)


def main():
    args = parse_args()
    info = load_info(args.dataset_dir)
    states = load_states(args.dataset_dir)
    joint_names = state_joint_names(info)

    if args.estimate_reset_pose:
        estimate_reset_pose(states, joint_names, args.reset_pose)
    else:
        if not args.reset_pose.exists():
            sys.exit(f"No reset pose at {args.reset_pose} - run with --estimate-reset-pose first.")
        extract(args.dataset_dir, info, states, joint_names, args)


if __name__ == "__main__":
    main()
