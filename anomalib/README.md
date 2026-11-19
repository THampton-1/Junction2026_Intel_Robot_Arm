# Anomalib: four tests, from baseline to held-out check

Each test answered one question, and its answer shaped the next. Full results are in each
test's `findings.md`.

| | Question | Answer | Folder |
|---|---|---|---|
| **Baseline** | | | |
| Test 1 (14 Nov) | Can PatchCore tell blocks apart at all? Stand-in defect: a big block instead of a small one, claw camera. | Yes, AUROC 1.0 on 24 photos, but a gap of only 0.044. 3.78 ms per photo on the Xe3 iGPU against 75 ms on the CPU. | `test_1/` |
| Test 2.1 (14 Nov) | Which camera? Same task on all three cameras. | Side view: AUROC 1.0, gap +0.156. Top: +0.103. Claw: scores overlap (AUROC 0.943). **Side camera chosen.** | `test_2/Test 2.1/` |
| **Stretch** | | | |
| Test 3 (15 Nov) | The real defect, a dot, on the side camera with the trigger's tight crop. | 15/15 visible defects caught, 8/8 good blocks passed. Removing ep05 (a dotted block in the "good" training set) doubled the gap from 0.127 to 0.252. **Threshold 0.37** set by the midpoint rule written down in Test 2.1. | `test_3/` |
| Test 4 (19 Nov) | Does 0.37 hold on photos the model has never seen? Labels and threshold fixed before scoring. | 9/9 defects caught, 5/5 good blocks passed, AUROC 1.000, gap +0.243 (Theo's pre-scoring labels; see `findings.md` §0 for the label correction). | `test_4/` |

## Why the order matters

Tests 1 and 2.1 used a stand-in defect to choose the camera. Test 3 set the threshold on its own
test photos, which on its own would be fitting to the data. Test 4 checks that threshold,
unchanged, on new photos.

## Other folders

The `block_to_*` and `theo_frames_to_crop*` folders hold the first frames exported from each
dataset, the raw material the test sets were built from.
