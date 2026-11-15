# Block to Pink v4: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstrations for `Block_To_Pink_v4` recorded on the LeRobot SO-101 robotic arm.

## Dataset Specifications
* **Dataset ID:** `80b4f232-a586-4416-a098-fb93752a6175`
* **Task:** `Place block on pink plate`
* **Episodes:** 6 (`episode_00` through `episode_05`)
* **Total Frames:** 3,828 synchronized video and state frames
* **Total Duration:** 127.6 seconds (~2.1 minutes)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
* **`code/joint_trajectory_player.py`**: Python CLI tool to inspect, summarize, replay, and query joint positions.

## Python API Usage
```bash
# Print summary for all 6 episodes:
python3 datasets/block_to_pink_v4/code/joint_trajectory_player.py --summary-all

# Query specific episode joint positions:
python3 datasets/block_to_pink_v4/code/joint_trajectory_player.py --episode 0
```
