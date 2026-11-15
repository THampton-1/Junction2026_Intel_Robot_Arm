# Block to Green: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstrations for `Block to Green` recorded on the LeRobot SO-101 robotic arm in Physical AI Studio.

## Dataset Specifications
* **Dataset ID:** `ea625a1c-d316-4a7b-88f0-43838fbf9364`
* **Episodes:** 22 (`episode_00` through `episode_21`)
* **Total Frames:** 4,867 synchronized video and state frames
* **Total Duration:** 162.2 seconds (~2.7 minutes)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
* **`code/joint_trajectory_player.py`**: Python CLI tool to inspect, summarize, replay, and query joint positions.

## Python API Usage
```bash
# Print summary for all episodes:
python3 datasets/block_to_green/code/joint_trajectory_player.py --summary-all

# Query specific episode joint positions:
python3 datasets/block_to_green/code/joint_trajectory_player.py --episode 0
```
