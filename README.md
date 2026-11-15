# Junction 2026 · Intel Physical AI Challenge: SO-101 block inspection and sorting

An SO-101 arm turns a Duplo block through four 90° rotations, photographs each face with a side
camera, checks every photo for defects with Anomalib PatchCore, then places the block in the
green bowl (all faces good) or the pink bowl (any defect). The arm is driven by ACT policies,
and every model runs through OpenVINO on an Intel Core Ultra laptop.

Hardware, calibration, environments and decision history: `MASTER_PROJECT_CONTEXT.md`.
This file is the map of the code.

---

## 1. End-to-end flow

```text
 OFFLINE: build the models                                    LIVE: run the station
 ─────────────────────────                                    ─────────────────────
 Teleoperated demos (leader → follower arm)                   main.py  (PySide6 desktop app)
   recorded in Physical AI Studio (3 cameras + joints)          ├─ ui_intel2000s.py    front end: cameras, filmstrip, stats
        │                                                       └─ robot_fsm.py        back end: state machine (QThread)
        ├─► ACT training (Studio → Modal A10G) ─► OpenVINO ACT        │ uses ov_models.py (OpenVINO wrappers)
        │      act_rotate_90 · act_place_green · act_place_pink       │ and config.py (all paths and settings)
        │                                                             ▼
        └─► scripts/export_datasets.py ─► datasets/<task>/       IDLE ──start──► ROTATE_90  (ACT act_rotate_90)
               videos + joint CSVs                                    ▲             │ gripper opens, then closes
                    │                                                 │             ▼
                    ▼                                                 │         RETURN_HOME (scripted move to home_inspect_pose)
    algorithm/extract_reset_frames.py                                 │             ▼
      (photo when the arm parks at reset pose X)                      │         SETTLE_AND_INSPECT
    algorithm/crop_photos.py  (static photos)                         │           side camera → crop → PatchCore → score ≥ threshold?
      both crop to INSPECTION_CROP                                    │             │ faces 1–3 → ROTATE_90 again
                    │                                                 │             ▼ after face 4
                    ▼                                                 │         SORT_BRICK (ACT place_pink if any defect, else place_green)
    anomalib/test_N/data/  ─► scripts/train_anomalib_patchcore.py     │             ▼
                    ─► OpenVINO PatchCore (.xml + .bin, not in git)   └──────── RETURN_HOME
```

## 2. Where everything lives

### Live station (runs on the robot laptop)

| File | Role |
|---|---|
| `main.py` | Entry point. `python main.py` (hardware) or `python main.py --mock` (no robot). |
| `ui_intel2000s.py` | **Front end.** PySide6 window: live camera views, per-face inspection filmstrip, session stats. |
| `robot_fsm.py` | **Back end.** `RobotControlWorker` state machine: cameras, joints, ACT policies, inspection gate, sorting. |
| `ov_models.py` | OpenVINO wrappers: `ACTOpenVINOModel` (policy step) and `AnomalibOpenVINOModel` (score, mask, heat map). Falls back to a mock if a model file is missing. |
| `config.py` | Every path, camera index, threshold, crop box and pose for the station. Overridable with `app_config.json`. |
| `test_dry_cycle.py` | Runs the full state machine in mock mode to check the cycle end to end. |

### Inspection photo pipeline (builds the Anomalib training data)

| File | Role |
|---|---|
| `algorithm/reset_pose_trigger.py` | Decides when to take the photo (arm parked within ±10 of reset pose X) and holds `INSPECTION_CROP`, the one shared crop box. |
| `algorithm/extract_reset_frames.py` | Runs the trigger over recorded episodes (`datasets/` export layout or raw LeRobot) and saves cropped photos. |
| `algorithm/crop_photos.py` | Crops folders of existing side_view photos with the same box. |
| `algorithm/reset_pose.json` | Reset pose X, estimated from `datasets/90_degree_turn`. |
| `algorithm/README.md` | How the trigger and crop work, and how they were checked on real episodes. |

### Model training and evaluation

| File | Role |
|---|---|
| `scripts/train_anomalib_patchcore.py` | Train, test, export and benchmark PatchCore. Needs a folder with `train/normal`, `test/normal`, `test/abnormal`. |
| `scripts/make_test2_1_split.py` | Builds the Test 2.1 split. |
| `anomalib/test_1`, `anomalib/test_2`, `anomalib/test_3` | Anomalib experiments: data, logs and `findings.md`. **Test 3 is the final model.** |
| `anomalib/theo_frames_to_crop*`, `anomalib/block_to_*` | Source photos used by the tests. |
| `scripts/export_datasets.py` | Exports Studio recordings into `datasets/` (videos, joint CSV/Parquet, first frames). |
| `datasets/<task>/` | Exported recordings, one per task (e.g. `90_degree_turn`, `block_to_green_v4`), each with its own README. |
| `scripts/sync_anomalib.sh` | Stages, commits and pushes `anomalib/` artefacts. |

### Alternatives, demo and reporting

| File | Role |
|---|---|
| `scripts/sorting_middleware.py` | A second, command-line version of the closed loop (4 turns, inspect, route). Writes `runs/<timestamp>/summary.json`. The desktop app does not use it. |
| `scripts/benchmark_and_telemetry.py` | CPU / GPU / NPU latency, trial accuracy, heat maps; writes `reports/`. |
| `reports/INTEL_CHALLENGE_METRICS.md` | Judging-rubric report, generated with the **Test 1 claw-camera** model. |
| `website/` | Browser demo page (`python website/server.py`), with recorded or live camera feeds and a Gemini chatbot. Standalone: not yet connected to the robot loop. |

## 3. Final models

| Model | Where | Used for |
|---|---|---|
| ACT `act_rotate_90`, `act_place_green`, `act_place_pink` | Robot laptop, under `~/.local/share/physicalai/models/` (paths in `config.py`) | Arm motion |
| PatchCore side_view, **Anomalib Test 3** | Training laptop, `C:\Users\theoh\anomalib_runs\test_3\side_view_no_ep05\export\weights\openvino\` (127 MB, not in git) | Defect gate. Threshold **0.37**, crop **x 60, y 130, 54 × 49** on side_view. AUROC 1.00 on visible defects. See `anomalib/test_3/findings.md`. |

## 4. Integration status: what is not wired up yet

The live code still points at the **Test 1 claw-camera model**. These are the known gaps,
listed so the architecture can show them honestly.

1. **Old Anomalib model and threshold in three places.** `config.py` (desktop app),
   `scripts/sorting_middleware.py` (defaults) and `website/server.py` (demo constant) all use
   `anomalib_patchcore_claw` with threshold 0.4781. The final model needs:
   - the Test 3 model files copied to the robot laptop;
   - in `config.py`: `anomalib_xml` set to the copied file, `anomalib_threshold = 0.37`, and
     `roi_x, roi_y, roi_w, roi_h = 60, 130, 54, 49` (currently `60, 60, 180, 140`).

   `anomalib_input_size` stays `(240, 320)`: the exported model takes 240 × 320 input, and the
   app already resizes the crop to that.

   Copy the files first. If a model path doesn't exist, `ov_models.py` logs a warning and
   switches to a **mock** that returns made-up scores, so the app still runs but the verdicts
   are not real.
2. **Photo pose differs from the training photos.** The app parks at
   `home_inspect_pose = [-5, -100, 98, 60, -50, 2]`, but every training photo was taken at the
   recorded reset pose X: `shoulder_pan -6.1, shoulder_lift -99.8, elbow_flex 99.3,
   wrist_flex 75.7` (`algorithm/reset_pose.json`). That is 15.7 apart on `wrist_flex`. The
   claw will sit in a different place in the side image, so check one live photo against the
   crop before trusting scores. Alternatively, set `home_inspect_pose` to X.
3. **Gripper thresholds versus recorded values.** `config.py` describes `grasp_threshold = 3.5`
   as "closed on brick" and `release_threshold = 2.5` as "opened". In the recorded joint data
   the gripper reads about 1.6 when closed empty, about 5 when holding the block, and 20–49
   when open. So `≥ 3.5` is first met when the gripper opens, and `≤ 2.5` when it closes empty
   after the release. The sequence still marks the end of a rotation, but the comments don't
   match the numbers. Verify on the robot.
4. **Two live loops.** The desktop app (`main.py` → `robot_fsm.py`) and
   `scripts/sorting_middleware.py` implement the same cycle separately. The desktop app is the
   one with the front end.
5. **The website is standalone.** `website/server.py` serves its own demo flow; its docstring
   describes how it would hook into the robot loop later.

## 5. Quick commands

```bash
python main.py --mock                                     # desktop station without hardware
python test_dry_cycle.py                                  # mock end-to-end cycle check
python website/server.py                                  # browser demo, recorded frames
python -m unittest discover -s algorithm -v               # trigger tests
python algorithm/extract_reset_frames.py datasets/90_degree_turn --out inspection_frames
python scripts/train_anomalib_patchcore.py --data-root anomalib/test_3/data/side_view_no_ep05 \
    --export-dir <dir> --results-dir <dir>
```
