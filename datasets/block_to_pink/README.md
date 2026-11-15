# Block to Pink: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstrations for `Block to Pink` recorded on the LeRobot SO-101 robotic arm in Physical AI Studio.

## Dataset Specifications
* **Dataset ID:** `d9b531e6-72c5-4174-90ab-299100d158b0`
* **Episodes:** 22 (`episode_00` through `episode_21`)
* **Total Frames:** 6,002 synchronized video and state frames
* **Total Duration:** 200.1 seconds (~3.3 minutes)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
* **`code/joint_trajectory_player.py`**: Python CLI tool to inspect, summarize, replay, and query joint positions.

## Python API Usage
```bash
# Print summary for all episodes:
python3 datasets/block_to_pink/code/joint_trajectory_player.py --summary-all

# Query specific episode joint positions:
python3 datasets/block_to_pink/code/joint_trajectory_player.py --episode 0
```
