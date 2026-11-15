# Inspection photo trigger (reset pose X)

Takes one side-camera photo each time the arm parks at the marked **reset pose X** after
releasing the block. Each recorded episode is one rotation (grasp, rotate, release, back to
reset), so each episode gives one photo of the block's newly exposed face. Run live through
the full routine, it gives one photo per rotation.

The claw is in the side camera's view at X, so every photo is cropped to `INSPECTION_CROP`
(x 60, y 130, 54 × 49 px), defined once in `reset_pose_trigger.py`. Training and live photos
use that same box. See [Checked on real episodes](#checked-on-real-episodes).

The same `ResetPoseTrigger` runs offline on recorded episodes (to build training crops)
and live on the robot, so training and runtime photos are chosen by the same rule.

## The rule

Each frame, with `d` = the largest |joint − X| over the checked joints:

| Step | Condition | Default |
|---|---|---|
| Arm | `d > TOLERANCE + EXIT_MARGIN` at some point since the last photo | 10 + 5 |
| In pose | `d ≤ TOLERANCE` (every checked joint within ±10 of X) | ±10 |
| Fire | in pose for `HOLD_FRAMES` consecutive frames, then disarm | 5 frames (0.17 s) |

- **±10** lets a policy that only gets *close* to X still produce the photo.
- **Hold frames** ignore a quick pass through the pose.
- **Disarm / re-arm beyond 15**: jitter at the edge of the box can't fire twice in one visit.
- **Starts disarmed**: resting at X at the start of an episode is not a photo.
- **Retried grasps** never visit X, so they can't add photos.
- **Checked joints**: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`. These decide
  where the arm is. `wrist_roll` and `gripper` are ignored because they don't move the arm
  into or out of the camera's view.

## Files

| File | What it does |
|---|---|
| `reset_pose_trigger.py` | The trigger. Plain Python, no dependencies. |
| `extract_reset_frames.py` | Runs the trigger over recorded episodes and saves each photo frame, full and cropped to `INSPECTION_CROP`. Reads the team's export layout (`datasets/90_degree_turn`) and raw LeRobot v2.x/v3.x. Needs pandas, pyarrow and the `ffmpeg` command. |
| `test_reset_pose_trigger.py` | 11 tests on synthetic joint traces (near-miss, jitter, retries, quick pass-through, ...). |
| `reset_pose.json` | X itself, estimated from the 10 `90_degree_turn` episodes. Re-create it if the reset mark moves. |
| `episode_check/` | Frames from the real episodes, used to check the trigger and the crop. |

## Using it

```bash
# 1. Find X: median of every episode's first frame (the arm parked at reset). Check the
#    printed per-joint range; a large range means episodes didn't start at the same pose.
python algorithm/extract_reset_frames.py datasets/90_degree_turn --estimate-reset-pose

# 2. Extract photos (cropped to INSPECTION_CROP unless you pass --crop X Y W H)
python algorithm/extract_reset_frames.py datasets/90_degree_turn --out inspection_frames

# Tests
python -m unittest discover -s algorithm -v
```

Any episode that doesn't give exactly `--expected-photos` (default 1) is listed at the end
for checking by eye.

Live, in the control loop:

```python
from reset_pose_trigger import INSPECTION_CROP, ResetPoseTrigger, load_reset_pose

trigger = ResetPoseTrigger(load_reset_pose("algorithm/reset_pose.json"))
x, y, w, h = INSPECTION_CROP
while running:
    joint_state = read_follower_joints()   # dict: "shoulder_pan" -> position, same units as recording
    if trigger.update(joint_state):
        frame = side_camera.read()         # 320x240 image as a numpy array (rows, columns, colour)
        photo = frame[y:y + h, x:x + w]    # same box as the training crops
```

## Checked on real episodes

All 10 teleoperated episodes of `datasets/90_degree_turn` (one rotation each, side_view
H.264 320×240 @ 30 fps). Images are in `episode_check/`.

- **Units are normalised −100…100** (the gripper is 0…100), so ±10 is 5% of each joint's
  range. The default stands.
- **X** (median of the 10 first frames): `shoulder_pan` −6.11, `shoulder_lift` −99.78,
  `elbow_flex` 99.27, `wrist_flex` 75.68. Across episodes the start poses spread by at most
  4.6 on any checked joint, and every episode ended within 4.6 of X. `wrist_roll` spreads by
  9, which doesn't matter because it isn't checked.
- **Exactly one photo per episode, 10 of 10**, fired 29–61 frames (1–2 s) before each
  episode ended. Nothing comes near X while the arm works over the block.
- **Episode 00 in detail:** the trigger arms at frame 99, when `shoulder_lift` goes more than
  15 from X. On the way back, `elbow_flex` reaches 89.46 at frame 1007, and the trigger
  **fires at frame 1011 (33.7 s)**. The operator paused there with the elbow about 10 short
  of X, then settled at frame 1032; the ±10 box is what lets that pause count.
- **Frame-exact:** the extractor's 10 photos are pixel-identical to the same frame numbers
  decoded directly.
- **The claw is in the camera's view at X.** Its tips sit just left of the block
  (`key_frames.jpg`, first and last tiles). The arm is only fully out of view while it is
  raised on the way back, about frames 930–970 of episode 00 (`return_frames_870_1060.jpg`).
- **`INSPECTION_CROP` (x 60, y 130, 54 × 49) holds the whole block and no claw in all 10
  photos** (`all_episodes_crops.png`). It was set from measurements: each block's outline was
  found by colour, and across the 10 photos the blocks span x 62–111, y 132–176. That leaves
  1–2 px around every block. On a 10 px grid, the claw tips reach x ≈ 52–53 in the closest
  episodes (01 and 04), about 7 px outside the crop. `all_episodes_crop_box.png` shows the
  new box (cyan) and the earlier, looser box (yellow).
- **Keep the side camera and the reset mark where they are.** The arm-control model is
  trained on this camera view and this reset pose, and the crop is measured for them. If
  either moves, re-measure the crop (and expect to retrain the policy).
- **Not done: centring the crop on each block by colour.** That would remove the few pixels
  of drift, but it depends on the block's colour, and wrong colours are one of the defects
  being detected. The fixed box treats every block the same.

Also tested on synthetic data: the trigger logic (unit tests), plus a fake LeRobot v2.1 and
v3.0 dataset where every one of 60 extracted images was confirmed to be the exact intended
video frame.
