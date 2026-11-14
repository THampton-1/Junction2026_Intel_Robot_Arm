#!/usr/bin/env python3
"""
Build the Anomalib Test 2.1 train/test split from the Block to Green v2 first frames.

- Abnormal episodes are the files suffixed _Big; they all go to test/abnormal.
- Normal episodes are split train/test in the same proportion as Test 1 (20 train : 8 test).
- The split is drawn ONCE over episode numbers with a fixed seed and applied to all three
  cameras, so every camera is trained and tested on exactly the same episodes.

Output (per camera) matches the layout train_anomalib_patchcore.py expects:
    data/<camera>/train/normal, data/<camera>/test/normal, data/<camera>/test/abnormal
"""

import json
import random
import shutil
import sys
from pathlib import Path

SEED = 2026
TEST1_TRAIN_NORMAL = 20
TEST1_TEST_NORMAL = 8
CAMERAS = ["claw_cam", "side_view", "top_view"]

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "anomalib" / "test_2" / "Test 2.1"
OUT_DIR = TEST_DIR / "data"


def episode_number(filename):
    # "green_v2_ep03_Big.png" -> "03"
    return filename.split("_ep")[1][:2]


def main():
    if OUT_DIR.exists():
        print(f"{OUT_DIR} already exists - delete it first if you want to rebuild the split.")
        sys.exit(1)

    # Read labels from claw_cam, then check the other cameras agree exactly.
    normal_eps = []
    abnormal_eps = []
    for path in sorted((TEST_DIR / "claw_cam").glob("*.png")):
        if path.stem.endswith("_Big"):
            abnormal_eps.append(episode_number(path.name))
        else:
            normal_eps.append(episode_number(path.name))

    for camera in CAMERAS:
        camera_abnormal = []
        for path in sorted((TEST_DIR / camera).glob("*.png")):
            if path.stem.endswith("_Big"):
                camera_abnormal.append(episode_number(path.name))
        if camera_abnormal != abnormal_eps:
            print(f"Label mismatch: {camera} marks {camera_abnormal} as Big, claw_cam marks {abnormal_eps}")
            sys.exit(1)

    # Same train fraction as Test 1, rounded to the nearest whole image.
    train_fraction = TEST1_TRAIN_NORMAL / (TEST1_TRAIN_NORMAL + TEST1_TEST_NORMAL)
    n_train = round(train_fraction * len(normal_eps))

    rng = random.Random(SEED)
    train_eps = sorted(rng.sample(normal_eps, n_train))
    test_normal_eps = []
    for ep in normal_eps:
        if ep not in train_eps:
            test_normal_eps.append(ep)

    # Copy files into the Folder-datamodule layout for each camera.
    split = {
        "train/normal": train_eps,
        "test/normal": test_normal_eps,
        "test/abnormal": abnormal_eps,
    }
    for camera in CAMERAS:
        for subdir, eps in split.items():
            target = OUT_DIR / camera / subdir
            target.mkdir(parents=True)
            for ep in eps:
                suffix = "_Big" if subdir == "test/abnormal" else ""
                name = f"green_v2_ep{ep}{suffix}.png"
                shutil.copy2(TEST_DIR / camera / name, target / name)

    manifest = {
        "seed": SEED,
        "method": "random.Random(seed).sample(sorted normal episodes, n_train); remainder -> test/normal",
        "train_fraction_from_test1": f"{TEST1_TRAIN_NORMAL}/{TEST1_TRAIN_NORMAL + TEST1_TEST_NORMAL}",
        "cameras": CAMERAS,
        "train_normal": train_eps,
        "test_normal": test_normal_eps,
        "test_abnormal": abnormal_eps,
    }
    with open(OUT_DIR / "split.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Normal episodes:   {len(normal_eps)}  ->  train {len(train_eps)}, test {len(test_normal_eps)}")
    print(f"Abnormal episodes: {len(abnormal_eps)}  ->  all test")
    print(f"train/normal:  {train_eps}")
    print(f"test/normal:   {test_normal_eps}")
    print(f"test/abnormal: {abnormal_eps}")


if __name__ == "__main__":
    main()
