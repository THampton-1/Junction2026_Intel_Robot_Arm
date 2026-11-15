#!/usr/bin/env python3
"""
Photo trigger: fire once each time the arm comes to rest at the marked reset pose X.

The routine is: square the block in the middle -> go to reset -> (grasp, rotate 90 deg
clockwise, release -> go to reset) x 4. Each arrival at reset is a photo moment: the arm
is out of the side camera's view and the block is sitting squared and still.

Rule (checked once per frame, using only past frames, so it works live and offline):

    1. ARMED      - the arm has been away from X (some joint more than
                    TOLERANCE + EXIT_MARGIN from X) since the last photo.
    2. IN POSE    - every checked joint is within +-TOLERANCE of X.
    3. HELD       - (2) has been true for HOLD_FRAMES consecutive frames.

    ARMED and HELD -> fire once, then disarm until the arm leaves again.

Why each piece is there:
  * +-TOLERANCE lets a policy that only gets *close* to X still trigger the photo.
  * HOLD_FRAMES stops a quick pass through the pose from triggering.
  * Disarming after a photo, and re-arming only beyond TOLERANCE + EXIT_MARGIN, means
    jitter at the edge of the tolerance box can't fire twice in one visit (hysteresis).
  * Starting disarmed means the arm resting at X at the start of an episode is not a photo.
  * Failed grasps and retries never visit X, so they can't produce extra photos.

Only the four joints that decide *where* the arm is are checked by default. wrist_roll
and gripper don't move the arm in or out of the camera's view, and the policy may leave
them at different values each time, so checking them would only cause missed photos.

Units: TOLERANCE is in whatever units observation.state is recorded in. Check them
against a real episode before trusting the default of 10.
"""

import json
from pathlib import Path

TOLERANCE = 10.0       # max allowed |joint - X| for every checked joint
EXIT_MARGIN = 5.0      # must go beyond TOLERANCE + EXIT_MARGIN to re-arm
HOLD_FRAMES = 5        # consecutive in-pose frames needed (5 frames = 0.17 s at 30 fps)
CHECKED_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

# The photo is cropped to this box (x, y, width, height in side_view pixels) both when
# building Anomalib training images and live, so the two always match. Measured on the 10
# 90_degree_turn photos: the blocks span x 62-111, y 132-176, so this leaves 1-2 px around
# every block, and the claw tips (which reach x ~53 at the reset pose) stay ~7 px outside.
# Re-measure if the side camera or the block's marked spot moves.
INSPECTION_CROP = (60, 130, 54, 49)


def short_joint_name(name):
    # "shoulder_pan.pos" -> "shoulder_pan"
    return name.split(".")[0]


def load_reset_pose(path):
    """Read X from a JSON file like {"shoulder_pan": 1.2, "shoulder_lift": -95.0, ...}."""
    with open(path) as f:
        pose = json.load(f)
    return {short_joint_name(name): float(value) for name, value in pose.items()}


def save_reset_pose(pose, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pose, f, indent=2)
        f.write("\n")


class ResetPoseTrigger:
    """Feed one joint state per frame to update(); it returns True on the photo frame."""

    def __init__(
        self,
        reset_pose,
        tolerance=TOLERANCE,
        exit_margin=EXIT_MARGIN,
        hold_frames=HOLD_FRAMES,
        joints=CHECKED_JOINTS,
        start_armed=False,
    ):
        missing = [joint for joint in joints if joint not in reset_pose]
        if missing:
            raise ValueError(f"Reset pose has no value for joints: {missing}")

        self.reset_pose = reset_pose
        self.tolerance = tolerance
        self.exit_margin = exit_margin
        self.hold_frames = hold_frames
        self.joints = joints
        self.armed = start_armed
        self.frames_in_pose = 0

    def largest_offset(self, joint_state):
        """The furthest any checked joint is from X (the max-norm distance)."""
        worst = 0.0
        for joint in self.joints:
            offset = abs(joint_state[joint] - self.reset_pose[joint])
            if offset > worst:
                worst = offset
        return worst

    def update(self, joint_state):
        """joint_state: dict of joint name -> position for the current frame."""
        offset = self.largest_offset(joint_state)

        if not self.armed:
            if offset > self.tolerance + self.exit_margin:
                self.armed = True
            return False

        if offset <= self.tolerance:
            self.frames_in_pose += 1
        else:
            self.frames_in_pose = 0

        if self.frames_in_pose >= self.hold_frames:
            self.armed = False
            self.frames_in_pose = 0
            return True
        return False
