# 90 Degree Turn Dataset: Full Multi-Camera Videos & Joint Telemetry

This dataset contains synchronized multi-camera video recordings and 6-DOF robotic arm joint position telemetry across all 10 episodes of the `90 degree turn` task recorded on the LeRobot SO-101 robotic arm.

---

## 1. Directory Structure

```text
datasets/90_degree_turn/
├── README.md                          # Dataset specifications & usage instructions
├── videos/                            # Full-duration MP4 videos (H.264 encoded, 30 FPS)
│   ├── episode_00/
│   │   ├── claw_cam.mp4               # Wrist camera (320x240, 35.5s)
│   │   ├── side_view.mp4              # Tabletop camera (320x240, 35.5s)
│   │   └── top_view.mp4               # Overhead RealSense D415 (640x480, 35.5s)
│   ├── episode_01/ ... episode_09/    # All 10 episodes fully rendered
├── trajectories/                      # Complete joint telemetry for the full duration
│   ├── episode_00_joints.csv          # 1,064 frames of timestamped joint angles
│   ├── episode_01_joints.csv ...
│   ├── episode_09_joints.csv
│   └── all_episodes_joints.parquet    # Fast binary storage of all 9,494 frames
└── code/
    └── joint_trajectory_player.py     # Python API to query, interpolate, or replay trajectories
```

---

## 2. Dataset Overview

* **Total Episodes:** 10 (Episodes `00` through `09`)
* **Total Frames:** 9,494 synchronized video and state frames
* **Total Duration:** 316.5 seconds (~5.3 minutes)
* **Sampling Rate:** 30 Hz (33.3 ms per frame)
* **Cameras Included:**
  * **`claw_cam`**: 320x240 H.264
  * **`side_view`**: 320x240 H.264
  * **`top_view`**: 640x480 H.264

---

## 3. Joint Telemetry Columns

Each CSV file in `trajectories/` provides timestamped telemetry for both the follower arm measured state (`state_*`) and the commanded action (`action_*`):

1. **`timestamp`**: Time in seconds from the start of the recording session.
2. **`frame_index`**: Sequential frame count within the episode (0 to N).
3. **`state_shoulder_pan` / `action_shoulder_pan`**: Base rotation (deg).
4. **`state_shoulder_lift` / `action_shoulder_lift`**: Shoulder elevation (deg).
5. **`state_elbow_flex` / `action_elbow_flex`**: Elbow pitch (deg).
6. **`state_wrist_flex` / `action_wrist_flex`**: Wrist pitch (deg).
7. **`state_wrist_roll` / `action_wrist_roll`**: Wrist roll rotation (deg).
8. **`state_gripper` / `action_gripper`**: End-effector aperture (deg / pos).

---

## 4. Python API Usage

The included script `code/joint_trajectory_player.py` provides instant access to query joint positions:

```bash
# Print statistical summary for an episode:
python3 datasets/90_degree_turn/code/joint_trajectory_player.py --episode 0

# Print summary across all 10 episodes:
python3 datasets/90_degree_turn/code/joint_trajectory_player.py --summary-all

# Query exact interpolated joint angles at t = 5.0 seconds into Episode 0:
python3 datasets/90_degree_turn/code/joint_trajectory_player.py --episode 0 --time-query 5.0

# Terminal streaming replay at 30 Hz:
python3 datasets/90_degree_turn/code/joint_trajectory_player.py --episode 0 --replay
```

From Python code:
```python
from datasets.90_degree_turn.code.joint_trajectory_player import JointTrajectoryDataset

dataset = JointTrajectoryDataset()
# Get joint angles at exactly 12.5 seconds:
joints = dataset.get_joint_positions_at_time(episode_idx=2, t_seconds=12.5)
print(joints)
# {'shoulder_pan': -8.4, 'shoulder_lift': -31.2, 'elbow_flex': 52.0, ...}
```
