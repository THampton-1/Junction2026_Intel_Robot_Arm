# Theo Frames to Crop: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstrations for `Theo_Frames_To_Crop` recorded on the LeRobot SO-101 robotic arm.

## Dataset Specifications
* **Dataset ID:** `a38ba05b-77bf-4041-ae3d-61dc2cf74fbb`
* **Task:** `Theo Frames to Crop`
* **Episodes:** 10 (`episode_00` through `episode_09`)
* **Total Frames:** 404 synchronized video and state frames
* **Total Duration:** ~13.5 seconds
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
* **`code/joint_trajectory_player.py`**: Python CLI tool to inspect, summarize, replay, and query joint positions.

## Python API Usage
```bash
# Print summary for all 10 episodes:
python3 datasets/theo_frames_to_crop/code/joint_trajectory_player.py --summary-all

# Query specific episode joint positions:
python3 datasets/theo_frames_to_crop/code/joint_trajectory_player.py --episode 0
```
