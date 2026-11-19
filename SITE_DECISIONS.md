# Portfolio site: decision log

How the Anomaly Scores page reached draft 3 (19 Nov 2026), plus the site-wide calls that shaped
it. Newest first within each section. The page plan itself is in `SITE_PLAN.md`; the Anomalib
evidence is in `anomalib/README.md` and each test's `findings.md`.

## Open: needs Theo's decision before the page is final

**D0. Which labels count for Test 4?** Theo's own labels for the 15 unseen photos are in
`Untested data classification.txt` (repo root, untracked), saved at 12:54 on 19 Nov. That is
before the Test 4 plan was committed (13:01, `c5e8344`) and before scoring (13:03). The committed
plan used labels Claude built from filenames and images instead, which Theo then approved in
chat. The two sets differ on three photos:

| Photo | Theo's file | Committed plan |
|---|---|---|
| v1 `theo_ep04` | defect | good |
| v1 `theo_ep00` | "Normal? maybe remove" | good |
| v2 `theo_v2_ep00` | defect | held back (Theo's "V2 00" remark was read as this photo; he likely meant v1 ep00) |

Results at the same fixed threshold (0.37), same model:

| | Committed plan | Theo's file (v1 ep00 removed) |
|---|---|---|
| Dots caught | 7/7 | **9/9** (`theo_v2_ep00` scored 0.705, scored 19 Nov after the mismatch was found) |
| Good passed | 6/7 (ep04 flagged) | **5/5** |

Both label sets were fixed before any score, so either is a fair test. Recommended: adopt
Theo's file as the official labels (they're his own photos, and he wrote them first). Commit the
file into `anomalib/test_4/` and add a dated note to `findings.md` explaining the switch. Keep the
committed-plan result alongside it, so nothing is hidden. Then update the page: the "one false
alarm" story becomes "every dot caught, and ep04's faint mark was one Theo had already called a
defect".

## Draft 3: pitch first (19 Nov, `c938a93`, `3e840b6` and the cycle change)

- **The page sells; the Reflection page hedges.** It's a pitch of what we did. Limits and the
  parked ep05 story move to `reflection.html` ("Limits of the defect model").
  *Why:* Theo: "a pitch of what we have done, hedged by reflection after."
- **Journey tiles lead with the question**, then the number, then the test name.
  *Why:* the question is what a reader scans for; "Test 2.1" means nothing on its own.
- **Camera choice: three camera cards, not a chart.** Each shows that camera's view and one
  verdict (claw mixed up −0.32, top +0.10, side +0.16 chosen).
  *Why:* the dot plot was "empty", and each camera's scores sit on its own 0–1 scale, so the dots
  weren't comparable across panels. One gap number per camera is.
- **Threshold section: four plain-English titles that expand** (the threshold, the score, the
  lever, pushing scores away), then one chart: Anomalib's default line 0.5 against ours 0.37.
  *Why:* the old section was "so dense", and the threshold is the least intuitive but most
  pivotal idea. The comparison was computed, not assumed: the default misses a dot in both sets
  (Test 3 ep34 at exactly 0.500; Test 4's red block at 0.464), and ours catches all of them with
  no extra false alarms.
- **"Did we just fit our own data?" became "Tested on photos it had never seen"**, with its
  first sentence saying how: the line was locked before any new photo was scored.
  *Why:* people read the first line; the old one admitted fitting before explaining the fix.
- **Parked: the ep05 story** (one dotted block in the good training set; removing it doubled the
  gap 0.127 → 0.252). *Why:* a separate narrative from the threshold. Chart kept at
  `docs/assets/charts/ep05.png`.
- **The cycle diagram became "The Anomalib cycle"**: inspection steps highlighted, the arm's
  steps faded. *Why:* the page is about Anomalib; the full architecture belongs elsewhere.
- **Correction:** "trained the way it's used" now says every training photo was taken with the
  arm parked at the same pose and the same crop. Only 10 of 29 came from the trigger itself.

## Draft 2: real evidence on the page (19 Nov, `1117d50`)

- **Headline visual: real PatchCore heatmaps on one fixed colour scale.**
  *Why:* Theo wanted something a reader can glance at. The old `reports/defect_heatmap.png` was
  meaningless (Test 1 claw model run on a full side frame, colours rescaled per image). The new
  maps put the hot spot on the dot every time, including on unseen photos.
- **Removed the PatchCore explainer and the video → photo section.** *Why:* too much before the
  point. Later idea: a flip-through of video → photo → crop, three side by side.
- **Charts in plotnine** (ggplot for Python), in their own venv
  (`C:\Users\theoh\venvs\site-charts`). *Why:* Theo uses ggplot. plotnine needs pandas 3, which
  breaks anomalib (Test 2.1 notes), so it can't share `anomalib-env`.
- **Colours validated, not eyeballed:** good #1fae82, defect #dc6f14, line #ffe08a, on the page
  blue #0a238f; shape as well as colour. *Why:* readable for colour-blind readers.
- **ep05 chart redrawn as two shaded-gap panels**, not before/after lines. *Why:* each training
  run rescales its own scores, so lines joining a photo across runs made most defects look
  worse. Only the gap compares across runs.

## Draft 1: the mock (19 Nov, `463acc1`)

- **The inspection cycle as a diagram**, with solid boxes for what was tested on recorded data
  and dashed for what was trained or coded but never run end to end. *Why:* Theo's favourite
  part, and the honest line between tested and planned.
- **Two headline numbers agreed:** yes/no counts with sample size, and the score gap.
  *Why:* the gap doesn't depend on where the threshold sits.
- **The four-test story** written into `anomalib/README.md`: baselines (Tests 1, 2.1) then
  stretch (Tests 3, 4).

## Evidence decisions behind the page (19 Nov)

- **Test 4, a held-out check, was run** because 0.37 had been set on Test 3's own photos. The plan
  (`anomalib/test_4/PLAN.md`) was committed before scoring. The scoring script reproduces the
  Test 3 log to within 0.001.
- **Theo's thresholding worry, answered:** the midpoint rule was written down on 14 Nov, before
  Test 3; the value 0.37 came from Test 3; Test 4 checks it without changing it.
- **The placeholder figures are banned** (93.3 %, 16.5 W, the old heatmap, the mock runs; see
  `ARCHITECTURE.md` W13).
- **Every exported dataset is teleop**, so the ACT page uses "demonstrations we taught ACT with"
  and is framed "trained, deployed, how we'd measure next time". The robot laptop was returned
  after the event.
- **`presentation.html` is a format reference only**; its claims were never reviewed.
