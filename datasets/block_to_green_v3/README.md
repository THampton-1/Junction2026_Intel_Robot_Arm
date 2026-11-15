# Block to Green v3: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstrations for `Block to Green v3` recorded on the LeRobot SO-101 robotic arm in Physical AI Studio.

## Dataset Specifications
* **Dataset ID:** `aeb3860c-12d7-42b3-9819-00109e297064`
* **Episodes:** 61 (`episode_00` through `episode_60`)
* **Total Frames:** 37,800 synchronized video and state frames
* **Total Duration:** 1260.0 seconds (~21.0 minutes)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
* **`code/joint_trajectory_player.py`**: Python CLI tool to inspect, summarize, replay, and query joint positions.

## Python API Usage
```bash
# Print summary for all episodes:
python3 datasets/block_to_green_v3/code/joint_trajectory_player.py --summary-all

# Query specific episode joint positions:
python3 datasets/block_to_green_v3/code/joint_trajectory_player.py --episode 0
```
