#!/usr/bin/env python3
"""
Score already-cropped photos with a trained PatchCore checkpoint, without retraining.

Used for Anomalib Test 4: the Test 3 model scores photos it has never seen, at a threshold
fixed in advance. Scores go through the same anomalib predict path as
scripts/train_anomalib_patchcore.py, so they are on the same normalised 0-1 scale as the
Test 3 log.

Usage:
    python scripts/score_anomalib_folder.py --ckpt <model.ckpt> --data-root <dir with test/normal
        and test/abnormal> --threshold 0.37 [--csv scores.csv]
"""

import argparse
import csv
import itertools
from pathlib import Path

from anomalib.engine import Engine
from anomalib.models import Patchcore


def score_folder(engine, model, folder):
    """Return a list of (file name, score) for every image in one folder, sorted by name."""
    results = []
    predictions = engine.predict(model=model, data_path=folder)
    for batch in predictions:
        for path_str, score in zip(batch.image_path, batch.pred_score):
            results.append((Path(path_str).name, float(score)))
    return sorted(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="PatchCore lightning checkpoint (model.ckpt)")
    parser.add_argument("--data-root", required=True, help="folder holding test/normal and test/abnormal")
    parser.add_argument("--threshold", type=float, required=True, help="a score above this is a defect")
    parser.add_argument("--csv", help="optional path to save every score")
    args = parser.parse_args()

    model = Patchcore.load_from_checkpoint(args.ckpt)
    engine = Engine(accelerator="cpu", devices=1, logger=False)

    root = Path(args.data_root)
    good = score_folder(engine, model, root / "test" / "normal")
    defect = score_folder(engine, model, root / "test" / "abnormal")

    print(f"\n{'photo':<24} {'label':<7} {'score':>6}  verdict")
    rows = []
    for label, items in [("good", good), ("defect", defect)]:
        for name, score in items:
            verdict = "defect" if score > args.threshold else "good"
            mark = "" if verdict == label else "   <-- wrong"
            print(f"{name:<24} {label:<7} {score:6.3f}  {verdict}{mark}")
            rows.append({"photo": name, "label": label, "score": round(score, 4), "verdict": verdict})

    good_passed = sum(1 for _, s in good if s <= args.threshold)
    defects_caught = sum(1 for _, s in defect if s > args.threshold)

    # AUROC = share of (good, defect) pairs where the defect scores higher; ties count half.
    wins = 0.0
    for (_, g), (_, d) in itertools.product(good, defect):
        if d > g:
            wins += 1
        elif d == g:
            wins += 0.5
    auroc = wins / (len(good) * len(defect))

    highest_good = max(s for _, s in good)
    lowest_defect = min(s for _, s in defect)

    print(f"\nThreshold {args.threshold}")
    print(f"Good blocks passed: {good_passed}/{len(good)}")
    print(f"Defects caught:     {defects_caught}/{len(defect)}")
    print(f"Image AUROC:        {auroc:.3f}  ({len(good) * len(defect)} pairs)")
    print(f"Gap:                {lowest_defect - highest_good:+.3f}  "
          f"(lowest defect {lowest_defect:.3f} - highest good {highest_good:.3f})")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["photo", "label", "score", "verdict"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Scores saved to {args.csv}")


if __name__ == "__main__":
    main()
