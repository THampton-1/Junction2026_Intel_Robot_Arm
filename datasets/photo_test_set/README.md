# Photo Test Set: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstrations for `Photo Test Set` recorded on the LeRobot SO-101 robotic arm in Physical AI Studio.

## Dataset Specifications
* **Dataset ID:** `06f21646-a3c2-4482-908c-41cba2c3e6e7`
* **Episodes:** 9 (`episode_00` through `episode_08`)
* **Total Frames:** 4,475 synchronized video and state frames
* **Total Duration:** 149.2 seconds (~2.5 minutes)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
* **`code/joint_trajectory_player.py`**: Python CLI tool to inspect, summarize, replay, and query joint positions.

## Python API Usage
```bash
# Print summary for all episodes:
python3 datasets/photo_test_set/code/joint_trajectory_player.py --summary-all

# Query specific episode joint positions:
python3 datasets/photo_test_set/code/joint_trajectory_player.py --episode 0
```
