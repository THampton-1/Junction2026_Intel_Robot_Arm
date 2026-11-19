# Anomalib Test 4: held-out check of the final model (plan, written before scoring)

> **Written:** 19 November 2026, before any photo in this folder was scored.
> **Purpose:** Test 3 chose the threshold (0.37) on its own 24 test photos. Test 4 checks that
> threshold, unchanged, on 14 photos the model has never seen.

## Fixed in advance

| | Value |
|---|---|
| Model | Test 3 final, `side_view_no_ep05` (checkpoint and OpenVINO export on the training laptop, `C:\Users\theoh\anomalib_runs\test_3\side_view_no_ep05\`) |
| Threshold | **0.37**. A photo is a defect if its score is above 0.37. Not changed after scoring. |
| Crop | `INSPECTION_CROP` = x 60, y 130, 54 × 49 px, via `algorithm/crop_photos.py`, the same as Test 3 |
| Scored | Once. Every result is reported, including failures. |

## Photos and labels

Labels are Theo's, set by eye before scoring. None of these photos was used in Test 3's training
or test sets: Test 3 used `theo_frames_to_crop_v2` episodes 05–48.

| Label | Photo | Colour | Source |
|---|---|---|---|
| Good | `theo_ep00` | red | `theo_frames_to_crop` (v1) |
| Good | `theo_ep01` | red | v1 |
| Good | `theo_ep03` | green | v1 |
| Good | `theo_ep04` | yellow | v1 |
| Good | `theo_ep05` | yellow | v1 |
| Good | `theo_ep06` | orange | v1 |
| Good | `theo_ep09` | light green | v1 |
| Defect | `theo_ep02_defect` | green | v1 |
| Defect | `theo_ep07_defect` | orange | v1 |
| Defect | `theo_ep08_defect` | light green | v1 |
| Defect | `theo_v2_ep01` | light green | `theo_frames_to_crop_v2` |
| Defect | `theo_v2_ep02` | yellow | v2 |
| Defect | `theo_v2_ep03` | orange | v2 |
| Defect | `theo_v2_ep04` | red | v2 |

Held back: `theo_v2_ep00` (ambiguous; kept for later use). The plain `theo_ep02`, `theo_ep07` and
`theo_ep08` files in v1 are byte-identical copies of the `_defect` files, left over from renames,
and are not counted twice.

## Predictions written before scoring

- `theo_ep04` (yellow, labelled good) shows a faint mark in the crop. If the model flags it, that
  was expected, and it is still counted as a false alarm against the label.
- The v1 photos were recorded in a different session from the training photos, so lighting may
  differ. The v2 dots are larger than most of Test 3's dots, so they should be easier to catch.

## Pipeline check before scoring

The 24 Test 3 test photos are re-scored through the same code first. If the scores don't match
`test_3/logs/side_view_no_ep05.log` to within 0.01, the pipeline is fixed before any Test 4
photo is scored.

## What gets reported

Defects caught out of 7 and good blocks passed out of 7 at threshold 0.37; image AUROC (49
good–defect pairs); and the gap (lowest defect score minus highest good score, negative if they
overlap). Results go in `findings.md` in this folder.
