# Grip Master v1: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstrations for `Grip Master v1` recorded on the LeRobot SO-101 robotic arm in Physical AI Studio.

## Dataset Specifications
* **Dataset ID:** `5869efcd-30b2-421b-b564-2030ee4ef22f`
* **Episodes:** 29 (`episode_00` through `episode_28`)
* **Total Frames:** 9,509 synchronized video and state frames
* **Total Duration:** 317.0 seconds (~5.3 minutes)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
* **`code/joint_trajectory_player.py`**: Python CLI tool to inspect, summarize, replay, and query joint positions.

## Python API Usage
```bash
# Print summary for all episodes:
python3 datasets/grip_master_v1/code/joint_trajectory_player.py --summary-all

# Query specific episode joint positions:
python3 datasets/grip_master_v1/code/joint_trajectory_player.py --episode 0
```
