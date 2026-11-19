# Anomalib Test 4: Results

> **Plan:** `PLAN.md`, committed (c5e8344) before any photo was scored
> **Model:** Test 3 final (`side_view_no_ep05`), unchanged. **Threshold:** 0.37, unchanged.
> **Scored:** once, 19 November 2026, with `scripts/score_anomalib_folder.py`. Every score is in `scores.csv`.

## 1. Result

| | Test 4 (held out, 14 photos) | Test 3 (threshold chosen here, 24 photos) |
|---|---|---|
| Defects caught at 0.37 | **7/7** | 15/16 (15/15 visible) |
| Good blocks passed at 0.37 | **6/7** | 8/8 |
| Image AUROC | **0.980** (48 of 49 pairs) | 0.961 (1.00 without ep33) |
| Highest good score | 0.524 (`theo_ep04`) | 0.248 |
| Lowest defect score | 0.464 (`theo_v2_ep04`) | 0.500 (ep34, visible defects) |
| Gap | −0.060 (one overlap) | +0.252 (visible defects) |

The threshold carried over to unseen photos: every defect was caught, and the one false alarm
is the photo predicted in the plan.

## 2. Per-photo scores

| Good | Colour | Score | | Defect | Colour | Score |
|---|---|---|---|---|---|---|
| `theo_ep00` | red | 0.162 | | `theo_ep02_defect` | green | 0.546 |
| `theo_ep01` | red | 0.135 | | `theo_ep07_defect` | orange | 0.529 |
| `theo_ep03` | green | 0.112 | | `theo_ep08_defect` | light green | 0.572 |
| `theo_ep04` | yellow | **0.524** ¹ | | `theo_v2_ep01` | light green | 0.716 |
| `theo_ep05` | yellow | 0.006 | | `theo_v2_ep02` | yellow | 0.649 |
| `theo_ep06` | orange | 0.221 | | `theo_v2_ep03` | orange | 0.642 |
| `theo_ep09` | light green | 0.056 | | `theo_v2_ep04` | red | 0.464 |

¹ **Predicted before scoring.** `PLAN.md` noted a faint mark on this block's crop. It stays a
false alarm against its label. Whether the label or the model is wrong is a question for the
physical block, which we no longer have.

## 3. Pipeline check

Before scoring, the 24 Test 3 test photos were re-scored through the same script. Every score
matched `test_3/logs/side_view_no_ep05.log` to within 0.001, with the same AUROC (0.961).

## 4. What this does and doesn't show

- **Shows:** a threshold chosen on one set of photos held on 14 photos the model had never
  seen, from two recording sessions, across all five block colours.
- **Good blocks score higher than in Test 3.** Six of Test 3's 8 good blocks scored 0.000; here
  the good blocks score 0.006–0.221 (excluding ep04). The v1 photos come from an earlier
  session, so this is most likely the lighting shift that the research notes warn about.
  Margin from the highest passing good block to the threshold: 0.149.
- **The closest defect is red** (`theo_v2_ep04`, 0.464, 0.094 above the threshold).
- **Small numbers.** 7 + 7 photos: one photo moves a rate by 14 points. The v2 dots are large.
  This is a check of the threshold, not a measurement of accuracy.
- **Offline only.** This is the model on recorded photos, not running live on the arm.
