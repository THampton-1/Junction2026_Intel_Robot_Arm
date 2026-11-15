# Block to Green v4: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstration(s) for `Block_To_Green_v4` recorded on the LeRobot SO-101 robotic arm.

## Dataset Specifications
* **Dataset ID:** `a531500b-01d8-496e-9d3e-6bee809b38c1`
* **Task:** `Place to green`
* **Episodes:** 1 (`episode_00`, 758 frames, 25.27 seconds)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/episode_00/`**: Sliced H.264 MP4 videos for `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_00_joints.csv` and `all_episodes_joints.parquet`.
