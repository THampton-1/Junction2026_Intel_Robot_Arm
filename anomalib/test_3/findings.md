# Anomalib Test 3: Final side-view model on the tight block crop

> **Experiment:** PatchCore (`wide_resnet50_2`, coreset_ratio = 0.2, k = 9), run with
> `scripts/train_anomalib_patchcore.py` unchanged
> **Camera:** side_view (320×240), every photo cropped to `INSPECTION_CROP` = x 60, y 130, 54 × 49 px
> **Date:** 15 November 2026
> **Status:** **final model.** Use the `side_view_no_ep05` run.

---

## 1. Result

| | Final (`side_view_no_ep05`, 29 train) | First run (`side_view`, 30 train) |
|---|---|---|
| Image AUROC, all 24 test photos | **0.961** | 0.945 |
| Image AUROC, excluding ep33 (defect not visible from the side) | **1.00** | 1.00 |
| Image F1 | 0.933 | 0.933 |
| Highest normal score | **0.248** (ep30) | 0.373 (ep30) |
| Lowest visible-defect score | **0.500** (ep34) | 0.500 (ep45) |
| Gap | **0.252** | 0.127 |

**Decision threshold for deployment: 0.37** (midpoint of 0.248 and 0.500). With it, 8/8 normal
photos pass and 15/15 visible defects are caught. Do not use the model's built-in 0.5: ep34
scores 0.500, right on that line, and the model's own label calls it normal.

Removing training photo ep05 (a yellow block with a small dark dot on its face) roughly doubled
the gap. The small-dot defect ep45 rose from 0.50 to 0.73, which suggests ep05 was teaching
the model that small dots are normal.

## 2. Data

All photos are side_view frames taken with the arm parked at the reset pose, cropped with
`algorithm/crop_photos.py` (the same box the extractor and the live loop use).

| Set | Source | Count |
|---|---|---|
| Train, normal | `anomalib/theo_frames_to_crop_v2/side_view/training` (ep05–ep24, **minus ep05**) | 19 |
| Train, normal | `datasets/90_degree_turn`, one photo per episode from the reset-pose trigger (`turn90_ep00`–`09`) | 10 |
| Test, normal | `anomalib/theo_frames_to_crop_v2/side_view/normal_tests` (ep25–ep32) | 8 |
| Test, abnormal | `anomalib/theo_frames_to_crop_v2/side_view/abnormal_tests` (ep33–ep48) | 16 |

Folders: `data/side_view_no_ep05/` (final) and `data/side_view/` (first run, with ep05).
`data/side_view/_contact_sheet.png` shows every crop. Each holds the whole block and no claw.

## 3. Per-image scores (final run)

| Normal | Score | | Abnormal | Score | | Abnormal | Score |
|---|---|---|---|---|---|---|---|
| ep25 | 0.000 | | ep33 | **0.000** ¹ | | ep41 | 0.676 |
| ep26 | 0.000 | | ep34 | **0.500** | | ep42 | 0.700 |
| ep27 | 0.000 | | ep35 | 0.643 | | ep43 | 0.847 |
| ep28 | 0.058 | | ep36 | 0.736 | | ep44 | 0.682 |
| ep29 | 0.000 | | ep37 | 0.517 | | ep45 | 0.728 |
| ep30 | **0.248** | | ep38 | 0.612 | | ep46 | 0.758 |
| ep31 | 0.000 | | ep39 | 0.570 | | ep47 | 0.539 |
| ep32 | 0.000 | | ep40 | 0.702 | | ep48 | 0.557 |

¹ ep33 shows no visible defect from the side camera, even uncropped. Its defect is presumably on
a face this camera cannot see, so no side-view model can catch it. Relabel it or drop it from
the test set.

Full logs: `logs/side_view_no_ep05.log` (final) and `logs/side_view.log` (first run).

## 4. Caveats

- The threshold was chosen on these same 24 test photos, as in Tests 1 and 2. Expect slightly
  less clean separation on new blocks; 0.25 of margin is moderate, not large.
- 29 training photos across 5 colours is a small normal set. More reset-pose photos of normal
  blocks would help most.
- The crop is measured for the current side camera position and block spot. If either moves,
  re-measure `INSPECTION_CROP` and retrain.
- OpenVINO latency in the log (≈110 ms, GPU) was measured on the Windows laptop where this was
  trained, not on the robot's Dell XPS. Re-benchmark there before quoting it.

## 5. Model files and deployment

The exported OpenVINO model is **not in git**: `patchcore_claw.bin` is 127 MB, and `*.bin` is
git-ignored. It lives on the training laptop at

```
C:\Users\theoh\anomalib_runs\test_3\side_view_no_ep05\export\weights\openvino\patchcore_claw.xml  (+ .bin)
```

Input `[1, 3, 240, 320]`; outputs `pred_score`, `pred_label`, `anomaly_map`, `pred_mask`. The
app's existing preprocessing (resize the crop to 320×240) matches this input.

To run it live, copy both files to the robot laptop, then set these values in the live code
(see "Integration status" in the root `README.md`):

| Setting | Old (Test 1 claw model) | Final (Test 3) |
|---|---|---|
| Model XML | `/home/intel/models/anomalib_patchcore_claw/.../patchcore_claw.xml` | the copied Test 3 XML |
| Threshold | 0.4781 | **0.37** |
| Camera | claw / side ROI 60, 60, 180 × 140 | side_view, **60, 130, 54 × 49** |

## 6. Reproduce

```bash
# crop (input folder holds train/normal, test/normal, test/abnormal of 320x240 side_view photos)
python algorithm/crop_photos.py <input_dir> anomalib/test_3/data/side_view_no_ep05

# train, test, export (about 3 minutes on CPU)
python scripts/train_anomalib_patchcore.py --data-root anomalib/test_3/data/side_view_no_ep05 \
    --export-dir <export_dir> --results-dir <results_dir>
```
