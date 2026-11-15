# Block to Pink v5: Multi-Camera Videos & Joint Telemetry

This dataset contains the recorded demonstrations for `Block_To_Pink_v5` recorded on the LeRobot SO-101 robotic arm.

## Dataset Specifications
* **Dataset ID:** `821972b1-430d-46b1-82bb-0646691c85fd`
* **Task:** `Place block to pink`
* **Episodes:** 6 (`episode_00` through `episode_05`)
* **Total Frames:** 3,916 synchronized video and state frames
* **Total Duration:** 130.5 seconds (~2.2 minutes)
* **Sampling Rate:** 30 Hz

## Directory Structure
* **`videos/`**: Sliced H.264 MP4 videos for each episode across `claw_cam.mp4`, `side_view.mp4`, and `top_view.mp4`.
* **`trajectories/`**: Complete 6-DOF joint angle telemetry (`state_*` and `action_*`) in `episode_0*_joints.csv` and `all_episodes_joints.parquet`.
* **`code/joint_trajectory_player.py`**: Python CLI tool to inspect, summarize, replay, and query joint positions.

## Python API Usage
```bash
# Print summary for all 6 episodes:
python3 datasets/block_to_pink_v5/code/joint_trajectory_player.py --summary-all

# Query specific episode joint positions:
python3 datasets/block_to_pink_v5/code/joint_trajectory_player.py --episode 0
```
