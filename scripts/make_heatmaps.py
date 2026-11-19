#!/usr/bin/env python3
"""
Draw PatchCore anomaly heatmaps for the website, every photo on ONE fixed colour scale.

For each chosen photo it saves a tile: the region the model sees (left) and the same region with
the anomaly map laid over it (right). The colour scale is fixed (0 to VMAX for every photo), so
"hot" means the same thing in every tile. Rescaling each map to its own min and max would make
some part of every photo look hot, even a perfect block.

Run with the anomalib venv:
    python scripts/make_heatmaps.py --ckpt <model.ckpt> --out docs/assets/charts
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from anomalib.engine import Engine
from anomalib.models import Patchcore

REPO = Path(__file__).resolve().parent.parent
VMAX = 0.7  # top of the colour scale; the largest map value across Tests 3 and 4 is 0.67

# (name used in the output file, photo)
PHOTOS = [
    ("good_ep25", "anomalib/test_3/data/side_view_no_ep05/test/normal/theo_v2_ep25.png"),
    ("small_dot_ep45", "anomalib/test_3/data/side_view_no_ep05/test/abnormal/theo_v2_ep45.png"),
    ("unseen_dot_v2_ep03", "anomalib/test_4/data/test/abnormal/theo_v2_ep03.png"),
    ("highest_good_ep30", "anomalib/test_3/data/side_view_no_ep05/test/normal/theo_v2_ep30.png"),
    ("false_alarm_ep04", "anomalib/test_4/data/test/normal/theo_ep04.png"),
]
TILE = 224  # the model resizes each crop to 256 x 256 and keeps the centre 224 x 224


def model_view(photo_path):
    """The part of the photo the model actually sees: resize to 256 x 256, keep the centre 224."""
    image = cv2.imread(str(photo_path))
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    return image[16:240, 16:240]


def colour_map(values):
    """Map anomaly values onto the inferno colour scale, fixed from 0 to VMAX."""
    scaled = np.clip(values / VMAX, 0, 1)
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="PatchCore lightning checkpoint (model.ckpt)")
    parser.add_argument("--out", required=True, help="folder for the PNG tiles")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = Patchcore.load_from_checkpoint(args.ckpt)
    engine = Engine(accelerator="cpu", devices=1, logger=False)

    for name, rel_path in PHOTOS:
        photo = REPO / rel_path
        batch = next(iter(engine.predict(model=model, data_path=photo)))
        score = float(batch.pred_score[0])
        anomaly = batch.anomaly_map[0].squeeze().cpu().numpy()

        view = model_view(photo)
        overlay = cv2.addWeighted(view, 0.45, colour_map(anomaly), 0.55, 0)
        gap = np.full((TILE, 6, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(out / f"heat_{name}.png"), np.hstack([view, gap, overlay]))
        print(f"{name:<22} score {score:.3f}  map max {anomaly.max():.3f}")

    # The colour key: a horizontal strip from 0 to VMAX
    strip = np.tile(np.linspace(0, VMAX, 448), (16, 1))
    cv2.imwrite(str(out / "heat_scale.png"), colour_map(strip))


if __name__ == "__main__":
    main()
