# Theo Frames to Crop v2: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstrations for `Theo_Frames_To_Crop_v2` recorded on the LeRobot SO-101 robotic arm.

## Dataset Specifications
* **Dataset ID:** `4e822e47-3a91-4817-9e3d-fc7e34a0625f`
* **Task:** `Dot on block`
* **Episodes:** 6 (`episode_00` through `episode_05`)
* **Total Frames:** 534 synchronized video and state frames
* **Total Duration:** 17.8 seconds (~0.3 minutes)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
* **`code/joint_trajectory_player.py`**: Python CLI tool to inspect, summarize, replay, and query joint positions.

## Python API Usage
```bash
# Print summary for all 6 episodes:
python3 datasets/theo_frames_to_crop_v2/code/joint_trajectory_player.py --summary-all

# Query specific episode joint positions:
python3 datasets/theo_frames_to_crop_v2/code/joint_trajectory_player.py --episode 0
```
