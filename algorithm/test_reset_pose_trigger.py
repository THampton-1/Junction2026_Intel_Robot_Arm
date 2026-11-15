#!/usr/bin/env python3
"""
Tests for ResetPoseTrigger on synthetic joint traces.

Run from the repo root:  python -m unittest discover -s algorithm -v
"""

import unittest

from reset_pose_trigger import ResetPoseTrigger

X = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -90.0,
    "elbow_flex": 90.0,
    "wrist_flex": 60.0,
    "wrist_roll": 0.0,
    "gripper": 5.0,
}
WORK = {  # a pose over the middle of the table, far from X
    "shoulder_pan": 10.0,
    "shoulder_lift": 20.0,
    "elbow_flex": 10.0,
    "wrist_flex": 70.0,
    "wrist_roll": 0.0,
    "gripper": 40.0,
}


def shifted(pose, amount, **overrides):
    """pose with every checked joint moved by `amount`, plus any explicit overrides."""
    result = dict(pose)
    for joint in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]:
        result[joint] = pose[joint] + amount
    result.update(overrides)
    return result


def run(trace, **trigger_kwargs):
    """Feed a list of joint states through a fresh trigger; return the frames it fired on."""
    trigger = ResetPoseTrigger(X, **trigger_kwargs)
    fired = []
    for frame, state in enumerate(trace):
        if trigger.update(state):
            fired.append(frame)
    return fired


class TestResetPoseTrigger(unittest.TestCase):
    def test_resting_at_start_is_not_a_photo(self):
        self.assertEqual(run([X] * 30), [])

    def test_full_routine_gives_one_photo_per_visit(self):
        # rest, then 5 cycles of (work for 20 frames, reset for 15 frames)
        trace = [X] * 10
        for cycle in range(5):
            trace += [WORK] * 20
            trace += [X] * 15
        fired = run(trace)
        self.assertEqual(len(fired), 5)
        # each photo is on the 5th frame of the visit (HOLD_FRAMES = 5)
        first_visit_start = 10 + 20
        self.assertEqual(fired[0], first_visit_start + 4)

    def test_close_but_not_exact_still_fires(self):
        near = shifted(X, 8.0)  # every joint 8 off: inside +-10
        self.assertEqual(len(run([WORK] * 10 + [near] * 10)), 1)

    def test_outside_tolerance_never_fires(self):
        off = shifted(X, 12.0)  # 12 off: outside +-10
        self.assertEqual(run([WORK] * 10 + [off] * 50), [])

    def test_one_joint_outside_tolerance_blocks_the_photo(self):
        off = dict(X, elbow_flex=X["elbow_flex"] + 11.0)
        self.assertEqual(run([WORK] * 10 + [off] * 50), [])

    def test_passing_through_quickly_does_not_fire(self):
        trace = [WORK] * 10 + [X] * 3 + [WORK] * 10  # only 3 frames in pose
        self.assertEqual(run(trace), [])

    def test_jitter_at_the_edge_fires_only_once(self):
        # alternates between 9 off (inside) and 13 off (outside but within the
        # re-arm margin of 15), so it never really leaves
        inside = shifted(X, 9.0)
        edge = shifted(X, 13.0)
        trace = [WORK] * 10 + [inside] * 6
        for _ in range(20):
            trace += [edge, inside, inside, inside, inside, inside]
        self.assertEqual(len(run(trace)), 1)

    def test_failed_grasp_retries_do_not_add_photos(self):
        open_hand = dict(WORK, gripper=40.0)
        closed_hand = dict(WORK, gripper=5.0)
        retries = ([open_hand] * 5 + [closed_hand] * 5) * 3
        trace = [WORK] * 10 + retries + [X] * 10
        self.assertEqual(len(run(trace)), 1)

    def test_wrist_roll_and_gripper_are_ignored(self):
        rolled = dict(X, wrist_roll=90.0, gripper=80.0)
        self.assertEqual(len(run([WORK] * 10 + [rolled] * 10)), 1)

    def test_start_armed_fires_on_first_rest(self):
        self.assertEqual(run([X] * 10, start_armed=True), [4])

    def test_missing_joint_in_reset_pose_is_an_error(self):
        with self.assertRaises(ValueError):
            ResetPoseTrigger({"shoulder_pan": 0.0})


if __name__ == "__main__":
    unittest.main()
