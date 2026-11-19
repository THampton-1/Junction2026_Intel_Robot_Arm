# Portfolio site plan

The plan for the GitHub Pages site in `docs/`, agreed 18 Nov 2026. The site keeps the team's retro
2000s "Adapt" look. The model is the winners' site (shashankdurgad.github.io/dlsj): short text, big
visuals, every claim a number with its sample size, before → after deltas.

`presentation.html` (repo root) is a good source of **format** to borrow: a fixed 16:10 stage,
CRT scanlines, numbered tab navigation, stat boxes, HUD terminal blocks. Its **content** was never
reviewed by a person; don't reuse any of its claims without a source.

## Rule for every page

Every claim carries a number and a source file in this repo. The novelty design means the points
have to be sharp and evidence based, so the site doesn't read as a joke. Never use the placeholder
numbers listed in `ARCHITECTURE.md` W13: the 93.3 % accuracy table (made by
`seed_canonical_trials_if_empty()`, not by physical trials), the 16.5 W power figure (a fallback
constant) or the heatmap and mock runs made by the same script.

## Pages, in reading order

| # | Page | File | Content |
|---|------|------|---------|
| 1 | Hero | `index.html` | Looping top, side and claw videos of one rotation; panel of checked facts. Done. |
| 2 | Overview | `happened.html` | Story, the challenge, architecture, proudest ideas. |
| 3 | Setup | `build.html` | Data and environment: hardware, three cameras, software, photo trigger and crop. |
| 4 | ACT vs VLA, movements | `arm.html` | Why ACT, the three policies, movement videos. **Filled last**, after talking to the team. |
| 5 | Anomalib | `detection.html` | Journey strip (4 tests), heatmaps, cycle diagram, camera choice, thresholds, limits. Later: a flip-through of video → photo → crop, three side by side to show how it standardises. |
| 6 | Reflection and integration | `reflection.html` | What fell short and the integration order we'd follow. |
| — | Sources for judges | `evidence.html` | What we can show, numbers we don't trust, what's missing. |
| — | Team | `team.html` | One entry each. |

## Video inventory

Exported from Physical AI Studio by `scripts/export_all_studio_datasets.py` into `datasets/`.
Each episode has `claw_cam`, `side_view` and `top_view` at 30 fps.

| Dataset | Episodes | Shows | Use on site |
|---------|----------|-------|-------------|
| `90_degree_turn` | 10 | Pick, turn 90°, put back. Side view is clearest. | Hero (done); movements. |
| `theo_frames_to_crop_v2` | 49 | Block with a dot, arm still. | Anomalib test data. |
| `block_to_green_v4` | 6 | Pick up, drop in green. | Movements. |
| `block_to_pink_v5` | 12 | Pick up, drop in pink. | Movements. |

The others are older or were recorded only for training. The dataset READMEs give stale episode
counts (pink v5 says 6, crop v2 says 6); the folder counts above are correct.

**All four are teleoperated demonstrations, not autonomous ACT runs.** This was checked from the
joint data (19 Nov). In every episode of every dataset, each `action_*` value sits on a lattice of
evenly spaced points. The spacing is different from the follower arm's `state_*` lattice: for
`shoulder_pan`, the action step is 0.0748 and the state step is 0.0798. So the action column is
the leader arm's servo reading, quantised in ticks and passed through the leader's calibration. A
policy's output would be continuous floats, and if it were ever quantised, it would land on the
follower's lattice. No exported dataset contains an autonomous run. On the site, caption these as "the demonstrations we taught ACT with". Footage of an
autonomous run would have to come from a teammate's phone.

## Evidence for choosing ACT (page 4)

| Reason | Number | Source | Status |
|--------|--------|--------|--------|
| Fast training | under 10 min, 2,765 steps on a Modal A10G | `MASTER_PROJECT_CONTEXT.md` §6 | Written by Gemini. Needs a primary source: the Studio training job page or the model folder on the robot laptop. |
| Small data budget | demos per policy: 10 (turn), 6 (green), 12 (pink) | `datasets/` folder counts | Measured. Need which dataset trained each policy (model IDs in `config.py`). |
| Runs on the laptop | one forward pass per 100-step chunk, i.e. one every 3.3 s at 30 Hz | `ARCHITECTURE.md` W12, `ov_models.py` | Chunk size from the code. ACT latency was **never measured**: `reports/intel_hardware_benchmark.csv` is PatchCore. |
| No language needed | Anomalib picks the policy, so a VLA's language input does nothing | `ARCHITECTURE.md` | Design argument. |
| ACT vs SmolVLA | — | — | Not measured; present as a design choice. Ask the team. |

**Decided 19 Nov:** page 4 uses the teleop demonstrations as its visuals and is framed as
"trained, deployed, and how we'd measure it next time". The borrowed robot laptop was returned after the event,
so Studio's model folders (training loss, metrics) are out of reach. Theo has asked the team for
anything they saved; treat that as a long shot.

## Open questions

- **Training loss.** Closed: the laptop holding it is gone. For ACT, this is the curve of its objective over training steps: L1 error
  between the predicted and demonstrated action chunks, plus a β-weighted KL term from its CVAE
  encoder. A curve that falls and flattens shows the policy fit the demos. It lives with the
  trained model in Studio, under `~/.local/share/physicalai/models/<model id>/` on the robot
  laptop. Decide whether it's worth fetching.
- **Training loss on Modal (serious stretch, but possible).** Studio itself was self-hosted on the
  laptop with no account, but ACT trained on Modal under Siongan's account (profile `siongan`, app
  `physicalai-trainer`, volume `physicalai-training`; see `MASTER_PROJECT_CONTEXT.md` Step 5).
  Odds are low: Studio's trainer deletes the dataset copy, checkpoints and `metrics.csv` once the
  model is downloaded (`trainer/api.py` `delete_job`, called from
  `services/training_backends/remote.py`, both pre-dating the event), and Modal keeps app logs
  1 day on Starter, 30 days on Team. The Modal app was Gemini-built and isn't in the repo, so the
  cleanup can't be confirmed. Ask Siongan to (1) check the Modal plan, and if Team, read
  `physicalai-trainer` logs from 14 Nov for loss values, before about 14 Dec; (2) run
  `modal volume ls physicalai-training` and look for `models/` or `metrics.csv`. Modal can't hold
  autonomous runs: ACT ran locally on the laptop.
- **SmolVLA.** Never trained (`MASTER_PROJECT_CONTEXT.md`: deferred). Talk to the team before
  writing page 4.
- **Autonomous run footage.** Closed for GitHub (19 Nov): no branch, fork, PR, release or
  teammate repo holds any; all 289 recorded episodes are teleop or the arm holding still. Only a
  teammate's phone could have it.
