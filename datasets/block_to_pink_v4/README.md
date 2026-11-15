# Block to Pink v4: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstration(s) for `Block_To_Pink_v4` recorded on the LeRobot SO-101 robotic arm.

## Dataset Specifications
* **Dataset ID:** `80b4f232-a586-4416-a098-fb93752a6175`
* **Task:** `Place block on pink plate`
* **Episodes:** 5 (`episode_00` through `episode_04`, 3,273 total frames, ~109.1 seconds)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
