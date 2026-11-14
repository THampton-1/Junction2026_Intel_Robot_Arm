# Anomalib Test 2.1: Experimental Design

> **Experiment:** PatchCore (`wide_resnet50_2`, coreset_ratio=0.2, k=9), same configuration as Test 1
> **Source data:** `Block to Green v2`, frame 0 of each of 25 episodes, from all three cameras
> **Question:** does the Test 1 separation between small (normal) and big (abnormal) blocks hold on a new dataset, and which camera separates best?
> **Date:** November 14, 2026

---

## 1. The Data Set

25 episodes, labelled from the filenames: `_Big` = abnormal (big block), no suffix = normal (small block).

| Class | Episodes | Count |
| :--- | :--- | :--- |
| Normal (small block) | 00 01 02 05 06 07 08 09 13 14 15 16 19 20 21 22 23 24 | 18 |
| Abnormal (big block) | 03 04 10 11 12 17 18 | 7 |

Each episode exists in three camera views:

| Camera | Folder | Resolution |
| :--- | :--- | :--- |
| Wrist claw camera | `claw_cam/` | 320x240 |
| Tabletop side webcam | `side_view/` | 320x240 |
| Overhead RealSense D415 | `top_view/` | 640x480 |

## 2. The Split and How It Was Chosen

**Proportion:** Test 1 split its normals 20 train : 8 test (71.4% train). The same fraction of 18 normals is 12.86, rounded to **13 train : 5 test** (72.2% train). As in Test 1, abnormals are never used for training; all 7 go to test.

**Randomisation:** `scripts/make_test2_1_split.py` sorts the 18 normal episode numbers and draws 13 without replacement using Python's `random.Random(2026).sample(...)`. The rest become the held-out test normals. The seed is fixed, so re-running the script reproduces the split exactly.

**Paired across cameras:** the draw is over *episode numbers* and is applied to all three cameras, so each camera is trained and tested on the same physical scenes. Any difference between cameras is then down to the viewpoint, not to which episodes they happened to get.

| Split | Episodes | Count |
| :--- | :--- | :--- |
| `train/normal` (in-sample) | 05 06 08 09 13 14 15 16 19 21 22 23 24 | 13 |
| `test/normal` (out-of-sample) | 00 01 02 07 20 | 5 |
| `test/abnormal` | 03 04 10 11 12 17 18 | 7 |

Ready-to-train copies are in `data/<camera>/{train/normal, test/normal, test/abnormal}`, with the split recorded in `data/split.json`. This is the layout `scripts/train_anomalib_patchcore.py` expects, so each camera runs with `--data-root "anomalib/test_2/Test 2.1/data/<camera>"`.

## 3. Hardware

| | Test 1 | Test 2.1 |
| :--- | :--- | :--- |
| Machine | Dell XPS 16, Ubuntu 26.04 | HP Pavilion 16, Windows 11 |
| CPU | Intel Core Ultra X9 388H | Intel Core Ultra 5 125U (12 cores / 14 threads) |
| GPU | Intel Xe3 iGPU (12 Xe3 cores) | Intel integrated graphics (no discrete GPU) |
| NPU | 50 TOPS | Intel AI Boost |
| RAM | n/a | 15.5 GB |

The memory bank is built on CPU in both tests (`accelerator="cpu"`), so separation results are directly comparable. OpenVINO inference latency is **not** comparable: it depends on the machine.

## 4. The Algorithm (PatchCore)

1. **Features:** each image goes through an ImageNet-pretrained `wide_resnet50_2`, which stays frozen and is never trained. Feature maps from `layer2` (stride 8, 512 channels) and `layer3` (stride 16, 1024 channels) are smoothed with a 3x3 average pool, `layer3` is upsampled to `layer2`'s grid, and the two are concatenated. Each spatial location becomes a patch descriptor in R^1536.
2. **Memory bank:** the patch descriptors of all 13 training images are pooled into one set M of "what normal looks like".
3. **Coreset subsampling:** greedy k-centre selection (on a random projection, for speed) keeps 20% of M. Each step adds the point farthest from the points already chosen, which approximately minimises the covering radius of the kept subset.
4. **Patch score:** for a test patch p, s(p) = min over m in M of ||p − m||₂. The patch scores form the anomaly map.
5. **Image score:** take the patch p\* with the largest score and its nearest bank point m\*. That score is down-weighted when p\* is also close to m\*'s 9 nearest bank neighbours (k=9), via a softmax over those distances, which damps scores in dense, well-covered regions.
6. **Threshold and normalisation:** Anomalib picks the F1-optimal threshold on the validation set and min-max normalises scores so the threshold maps to 0.5. As in Test 1, validation = test (`ValSplitMode.SAME_AS_TEST`).

## 5. Metrics (same as Test 1)

* Image AUROC and image F1
* Normal and abnormal score ranges (min / mean / max)
* Separation gap = lowest abnormal score − highest normal score, and the midpoint threshold
* OpenVINO latency (CPU / GPU / NPU) on this machine

## 6. Caveats to Keep in Mind When Reading Results

* **Threshold leakage:** step 6 fits the threshold and the normalisation on the same 12 images being scored. This flatters F1 and the normalised gap, and is why Test 1's lowest abnormal sat at exactly 0.5000. To compare across tests, also record the **raw** (un-normalised) scores.
* **Small test set:** 5 normals x 7 abnormals = 35 pairs, so AUROC moves in steps of 1/35. Purely random scores would give AUROC = 1 with probability 1/C(12,5) = 1/792 (Test 1: 1/C(24,8) ≈ 1/735,000), so a perfect AUROC here is weaker evidence than in Test 1.
* **Single source:** all 25 episodes are green-plate episodes from one recording session. Test 1 mixed green and pink.
