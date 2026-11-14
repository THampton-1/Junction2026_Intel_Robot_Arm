# Anomalib Test 2.1: Results

> **Experiment:** PatchCore (`wide_resnet50_2`, coreset_ratio=0.2, k=9), run with `scripts/train_anomalib_patchcore.py` unchanged
> **Data:** 13 train normal / 5 test normal / 7 test abnormal per camera, same episodes for every camera (see `README.md`)
> **Hardware:** HP Pavilion 16, Intel Core Ultra 5 125U, Windows 11, Python 3.13 venv (`torch 2.15.0+cpu`, `anomalib 2.0.0`, `openvino 2026.4.1`)
> **Date:** November 14, 2026
> **Full logs:** `logs/<camera>.log`

---

## 1. Summary

| Camera | Image AUROC | Image F1 | Highest normal | Lowest abnormal | Normalised gap | Raw gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `claw_cam` | 0.943 | 0.857 | 0.8244 (ep02) | 0.5000 (ep12) | **−0.324 (overlap)** | −8.19 |
| `side_view` | **1.000** | 0.923 | 0.3443 (ep02) | 0.5000 (ep17) | **+0.156** | +5.97 |
| `top_view` | **1.000** | 0.923 | 0.3972 (ep02) | 0.5000 (ep17) | +0.103 | +3.43 |
| *Test 1 claw_cam (20 train)* | *1.000* | *0.968* | *0.4562* | *0.5000* | *+0.044* | *n/a* |

* **Side view separates best:** zero overlap, and a margin 3.5x Test 1's in normalised units.
* **Claw cam failed on this dataset:** one normal (ep02) scores above two big blocks (ep12, ep17), giving 2 of 35 normal–abnormal pairs in the wrong order (AUROC = 33/35).
* **ep02 is the highest-scoring normal on all three cameras.**
* **Memory bank build:** about 50 s per camera on CPU (13 images).
* **OpenVINO latency (GPU):** 82–85 ms per frame on this laptop's integrated GPU. This is **not** comparable to Test 1's 3.78 ms on the Xe3 GPU, because the machine is different.

## 2. Why F1 < 1 When AUROC = 1

Anomalib picks the F1-optimal threshold from the observed scores, then labels an image abnormal only if `score > threshold` (strict; `post_processor.py`). When the classes separate perfectly, the optimal threshold lands **exactly on the lowest abnormal score**, so that image becomes a false negative: F1 = 2·6/(2·6+1) = 12/13 = 0.923.

The same thing happened in Test 1: F1 = 0.9677 = 30/31, which means `green_ep04` (score exactly 0.5000) was labelled **Normal** by Anomalib, not Abnormal as the Test 1 table says. For the live gate, use the midpoint threshold, not Anomalib's.

## 3. Raw Scores

The normalised score is `clamp((raw − t) / (max − min) + 0.5, 0, 1)`, with `t`, `min` and `max` all fitted on the test set (validation = test). This is affine before clamping, so:

* **Normalised gap = raw gap / (max − min).** It is the gap as a fraction of that run's score range, which is why it is the column to compare across runs.
* Raw scores are recovered by inverting the formula with the stored `t`, `min` and `max` from each checkpoint. The one clamped image per camera is the one that defines `min` or `max`, so its raw score is known exactly.

| Camera | Raw threshold t | Raw min | Raw max | Raw normal range | Raw abnormal range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `claw_cam` | 49.24 | 37.45 | 62.69 | 37.45 – 57.43 | 49.24 – 62.69 |
| `side_view` | 45.30 | 30.48 | 68.82 | 30.48 – 39.33 | 45.30 – 68.82 |
| `top_view` | 41.14 | 22.78 | 56.14 | 22.78 – 37.71 | 41.14 – 56.14 |

Raw scores are distances in each camera's own feature space, so compare them **within** a camera, not across cameras.

## 4. The Outlier: ep02

ep02 is a small **red** block, and it is the highest-scoring normal on every camera. Small blocks in this dataset come in several colours (cream, orange, red), and the training set contains only one red one (ep16), so the most likely explanation is that the memory bank under-covers red blocks. This is a hypothesis, not yet tested. Note that the challenge treats colour errors as defects, so how "red small block" should be scored is a design decision.

## 5. Caveats

* **13 training images vs Test 1's 20, on a different dataset.** This does **not** answer "does more data reduce the score": data size and dataset changed together. That needs a learning curve on one dataset (train on n = 5, 8, 10, 13 from the same pool, fixed test set).
* **Threshold leakage:** the threshold and normalisation are fitted on the 12 images they score. Treat F1 and the normalised gap as optimistic.
* **Small test set:** AUROC moves in steps of 1/35. A perfect AUROC from random scores has probability 1/C(12,5) = 1/792.

## 6. Running on Windows: Environment Fixes

`anomalib 2.0.0` needed these changes to run in a fresh Windows venv (all inside the venv, none in the repo):

1. Missing packages not declared by anomalib: `scikit-learn`, `python-dotenv`, `requests`, `onnx`.
2. `pandas<3`: pandas 3 changes string handling, and `make_folder_dataset` silently produces an empty table.
3. `openvino 2026` removed `openvino.runtime`: one import in `anomalib/deploy/inferencers/openvino_inferencer.py` changed to `openvino.utils.data_helpers.wrappers`.
4. Windows blocks symlinks without Developer Mode: `anomalib/utils/path.py` falls back to the version folder when the `latest` symlink can't be created.
