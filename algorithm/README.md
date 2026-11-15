# Inspection photo trigger (reset pose X)

Takes one side-camera photo each time the arm parks at the marked **reset pose X** after
releasing the block. With the routine *square block → reset → 4 × (grasp, rotate 90°
clockwise, release → reset)* that gives 5 photos per cycle. Photo 1 is the block as first
squared, and photo *n* is after *n − 1* rotations. Photos 1 and 5 show the same face, which
gives a free consistency check on the anomaly score.

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

## Using it

```bash
# 1. Find X: median of every episode's first frame (the arm parked at reset). Check the
#    printed per-joint range; a large range means episodes didn't start at the same pose.
python algorithm/extract_reset_frames.py <dataset_dir> --estimate-reset-pose

# 2. Extract photos (crop box is x y w h in side_view pixels around the centre spot)
python algorithm/extract_reset_frames.py <dataset_dir> --out inspection_frames --crop 40 110 70 70

# Tests
python -m unittest discover -s algorithm -v
```

Any episode that doesn't give exactly `--expected-photos` (default 5) is listed at the end
for checking by eye.

Live, in the control loop:

```python
from reset_pose_trigger import ResetPoseTrigger, load_reset_pose

trigger = ResetPoseTrigger(load_reset_pose("algorithm/reset_pose.json"))
while running:
    joint_state = read_follower_joints()   # dict: "shoulder_pan" -> position, same units as recording
    if trigger.update(joint_state):
        photo = side_camera.read()         # arm is parked at X, block is squared and still
```

## Check on the first real episode

1. **Units.** ±10 is in the units of `observation.state`. If those are normalised (−100…100),
   10 is about 5% of a joint's range. If they are raw servo ticks (0–4095), 10 is under 1°
   and far too tight. Look at the numbers before trusting the default.
2. **The tolerance box only holds near X.** Make sure no working pose comes within ±10 of X
   on all four joints at once. The tests use a reset pose far from the table, which holds
   if the reset position is clearly above and back from the work area.
3. **Crop box.** Measure it once from a real side_view frame of the squared block.

Tested so far on synthetic data only: the trigger logic (unit tests), plus a fake v2.1 and
v3.0 dataset where every one of 60 extracted images was confirmed to be the exact intended
video frame.
