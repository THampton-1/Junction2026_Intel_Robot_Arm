#!/usr/bin/env python3
"""
Crop a folder of side_view photos to INSPECTION_CROP, ready for Anomalib.

Use this for photos that are already single images (no joint data to run the trigger on).
The crop box is the same one extract_reset_frames.py and the live loop use.

    python algorithm/crop_photos.py <input_dir> <output_dir>

The input folder's structure is copied to the output, so if the photos are already sorted as

    <input_dir>/train/normal/*.png
    <input_dir>/test/normal/*.png
    <input_dir>/test/abnormal/*.png

then <output_dir> can go straight to scripts/train_anomalib_patchcore.py --data-root.

Every photo must be 320x240 (a side_view frame): the crop box was measured on those, so on
any other size it would cut out the wrong region. The script stops before writing anything
if a photo has the wrong size. It also saves <output_dir>/_contact_sheet.png showing every
crop, to check by eye that each one holds the block.

Needs: Pillow (already in the anomalib environment).
"""

import sys
from pathlib import Path

from PIL import Image

from reset_pose_trigger import INSPECTION_CROP

EXPECTED_SIZE = (320, 240)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
SHEET_COLUMNS = 8
SHEET_SCALE = 2


def find_photos(input_dir):
    photos = []
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith("_"):
            photos.append(path)
    return photos


def check_sizes(photos):
    wrong = []
    for path in photos:
        with Image.open(path) as image:
            if image.size != EXPECTED_SIZE:
                wrong.append(f"{path}: {image.size[0]}x{image.size[1]}")
    return wrong


def save_contact_sheet(crops, out_path):
    _, _, w, h = INSPECTION_CROP
    cell_w, cell_h = w * SHEET_SCALE, h * SHEET_SCALE
    rows = (len(crops) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = Image.new("RGB", (SHEET_COLUMNS * (cell_w + 4), rows * (cell_h + 4)), "white")
    for i, crop in enumerate(crops):
        column, row = i % SHEET_COLUMNS, i // SHEET_COLUMNS
        cell = crop.resize((cell_w, cell_h), Image.NEAREST)
        sheet.paste(cell, (column * (cell_w + 4), row * (cell_h + 4)))
    sheet.save(out_path)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])

    photos = find_photos(input_dir)
    if not photos:
        sys.exit(f"No images found under {input_dir}")

    wrong = check_sizes(photos)
    if wrong:
        print(f"Stopped: these photos are not {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]} side_view frames:")
        for line in wrong:
            print("  " + line)
        sys.exit(1)

    x, y, w, h = INSPECTION_CROP
    crops = []
    counts = {}
    for path in photos:
        relative = path.relative_to(input_dir)
        out_path = (output_dir / relative).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(path) as image:
            crop = image.convert("RGB").crop((x, y, x + w, y + h))
        crop.save(out_path)
        crops.append(crop)

        folder = str(relative.parent)
        counts[folder] = counts.get(folder, 0) + 1

    save_contact_sheet(crops, output_dir / "_contact_sheet.png")

    print(f"Cropped {len(photos)} photos to x {x}, y {y}, {w}x{h} px:")
    for folder, count in sorted(counts.items()):
        print(f"  {folder:20s} {count}")
    print(f"Check every crop holds the block: {output_dir / '_contact_sheet.png'}")


if __name__ == "__main__":
    main()
