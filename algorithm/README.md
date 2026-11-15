# Inspection photo trigger (reset pose X)

Takes one side-camera photo each time the arm parks at the marked **reset pose X** after
releasing the block. Each recorded episode is one rotation (grasp, rotate, release, back to
reset), so each episode gives one photo of the block's newly exposed face. Run live through
the full routine, it gives one photo per rotation.

The claw is in the side camera's view at X, so photos must use the tight crop below. See
[Checked on a real episode](#checked-on-a-real-episode).

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
| `extract_reset_frames.py` | Runs the trigger over recorded LeRobot episodes (v2.x or v3.x) and saves the photo frames plus an optional fixed crop. Needs pandas, pyarrow and the `ffmpeg` command. |
| `test_reset_pose_trigger.py` | 11 tests on synthetic joint traces (near-miss, jitter, retries, quick pass-through, ...). |
| `reset_pose.json` | X itself. Created by step 1 below. |
| `episode_check/` | Frames from the first real episode, used to check the trigger and the crop. |

## Using it

```bash
# 1. Find X: median of every episode's first frame (the arm parked at reset). Check the
#    printed per-joint range; a large range means episodes didn't start at the same pose.
python algorithm/extract_reset_frames.py <dataset_dir> --estimate-reset-pose

# 2. Extract photos (crop box is x y w h in side_view pixels around the centre spot)
python algorithm/extract_reset_frames.py <dataset_dir> --out inspection_frames --crop 58 125 56 56

# Tests
python -m unittest discover -s algorithm -v
```

Any episode that doesn't give exactly `--expected-photos` (default 1) is listed at the end
for checking by eye.

Live, in the control loop:

```python
from reset_pose_trigger import ResetPoseTrigger, load_reset_pose

trigger = ResetPoseTrigger(load_reset_pose("algorithm/reset_pose.json"))
while running:
    joint_state = read_follower_joints()   # dict: "shoulder_pan" -> position, same units as recording
    if trigger.update(joint_state):
        photo = side_camera.read()         # arm is parked at X; crop to 58 125 56 56
```

## Checked on a real episode

One teleoperated rotation episode (1064 frames, 35.5 s, side_view H.264 320×240 @ 30 fps).
Images are in `episode_check/`.

- **Units are normalised −100…100** (the gripper is 0…100), so ±10 is 5% of each joint's
  range. The default stands.
- **X** (from frame 0): `shoulder_pan` −4.99, `shoulder_lift` −99.83, `elbow_flex` 99.27,
  `wrist_flex` 76.26. The arm finished 3.1 off in `shoulder_pan` and 0.6 off in `wrist_flex`,
  well inside ±10. It finished with `wrist_roll` about 5 different, which doesn't matter
  because that joint isn't checked.
- **Trigger timeline** (traced by hand from the joint values): it arms at frame 99, when
  `shoulder_lift` goes more than 15 from X on the way out. Nothing comes near X while the
  arm works over the block. On the way back, `elbow_flex` reaches 89.46 at frame 1007
  (9.8 from X), the other three joints are already inside, and the trigger **fires at
  frame 1011 (34.0 s)**. That is one photo for the episode, as expected. The operator paused
  there with the elbow about 10 short before settling at frame 1032; the ±10 box is what
  lets that pause count.
- **The claw is in the camera's view at X.** Its tips sit just left of the block (see
  `key_frames.jpg`, first and last tiles). The arm is only fully out of the view between
  roughly frames 930 and 970, while it is raised on the way back
  (`return_frames_870_1060.jpg`).
- **A tight crop removes the claw.** The crop `58 125 56 56` contains only the block at
  frames 0, 950, 1011 and 1063 (`tight_crops.png`). The claw tips stop about 11 px left of
  the crop. If the block's spot or the camera moves, re-measure the crop.

Also tested on synthetic data: the trigger logic (unit tests), plus a fake v2.1 and v3.0
dataset where every one of 60 extracted images was confirmed to be the exact intended video
frame.
