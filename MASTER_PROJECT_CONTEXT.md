# MASTER_PROJECT_CONTEXT.md
# Junction 2026: Intel Physical AI Challenge — Complete Team & LLM Onboarding Guide

> **To any AI model (Claude, GPT, Gemini, Cursor) reading this file:**  
> You are collaborating on the **Junction 2026 Intel Physical AI Challenge** on an **SO-101 robotic arm** powered by an **Intel Core Ultra AI PC**.  
> Read this document completely before answering questions or generating code. It contains the exact hardware topology, software runtimes, architectural decisions, completed milestones, and immediate next steps. Do not guess hardware addresses or re-litigate accepted decisions.

---

## 1. Challenge Overview & Deliverables
* **Challenge Title:** *Stop Simulating, Start Moving: See It. Detect It. Act on It.* (Intel x Ignite Next at Junction 2026).
* **The Goal:** Build an autonomous, closed-loop robotic sorting system that inspects tabletop items (Duplo bricks), detects defects (e.g., scratches, missing studs, color errors), and physically sorts them into target containers (Good vs. Defective).
* **The 5-Stage Closed-Loop Pipeline:**
  $$\text{PERCEIVE} \longrightarrow \text{DETECT} \longrightarrow \text{REASON} \longrightarrow \text{EXECUTE} \longrightarrow \text{OPTIMIZE}$$
* **Judging Criteria (100 Points Total):**
  * **Task Success Rate (25 pts):** Accuracy, cycle time, robustness to clutter and shifted positions.
  * **Intel Tech Integration (25 pts):** Offloading across CPU, 12-core Xe3 iGPU, and 50 TOPS NPU using OpenVINO.
  * **Dataset & Pipeline Quality (20 pts):** Clean teleoperation data, effective policy training.
  * **System Efficiency (20 pts):** Latency per inference step, thermal headroom, power efficiency on Core Ultra.
  * **Creativity & Demo (10 pts):** Originality of approach and clarity of live demonstration.

---

## 2. Hardware Topology & Port Map
* **Host Computer:** Dell XPS 16 [DA16260], Ubuntu 26.04.1 LTS.
* **Processor:** Intel® Core™ Ultra X9 Series 3 (388H):
  * **CPU:** Fast response, deterministic control loops.
  * **iGPU (Intel Graphics [0xb080]):** 12 Xe3 cores, PyTorch XPU enabled (`torch.xpu.is_available() == True`), runs local Anomalib training & inference.
  * **NPU (`/dev/accel0`):** 50 TOPS neural accelerator, Level Zero registered in `/etc/ld.so.conf.d/intel-npu.conf`.
* **Robotic Arms (LeRobot SO-101):**
  * **Permanent Hardware USB Serial Numbers:**
    * **Follower Arm (Motorized, with Claw):** Serial **`5B91030784`** (Assigned port: `/dev/ttyACM0` or auto-matched).
    * **Leader Arm (Teleoperator Handle):** Serial **`5B90148383`** (Assigned port: `/dev/ttyACM1` or auto-matched).
  * **Permissions:** User `intel` is in `dialout`, `video`, and `render` groups. Permanent udev rule at `/etc/udev/rules.d/99-so101.rules` sets `0666` permissions.
* **Camera Configuration & Bandwidth Rules:**
  1. **Overhead Camera (`Top_view`):** **Intel RealSense D415** (Serial `346422061262`). Connected to SuperSpeed 5 Gbps USB bus (`/dev/video34-39`). Resolution: **`640x480 @ 30fps`**.
  2. **Tabletop Webcam (`Side_view` / Defect Inspection):** Sunplus Webcam Audio (`/dev/video32`). Resolution: **`320x240 @ 30fps`**.
  3. **Claw Camera (`Claw_view`):** Sonix `USB2.0_CAM1` (`/dev/video40`). Resolution: **`320x240 @ 30fps`**.
  * **CRITICAL USB BANDWIDTH MANDATE:** The Tabletop Webcam and Claw Camera share an internal USB 2.0 sub-hub. Running both at $640\times 480$ raw uncompressed video will saturate the bus and cause kernel error `Not enough bandwidth for new device state`. Keep both USB cameras at **`320x240 @ 30fps`** to allow concurrent streaming.

---

## 3. Robot Calibration Data (Do Not Re-Calibrate!)
Manual 3D zero calibration has already been completed by the team and verified via live bilateral teleoperation.
The calibration homing offsets and encoder ranges are permanently saved in:
* **JSON File:** `/home/intel/so101_calibration_backup.json`
* **Studio Database:** `~/.local/share/physicalai/data/physicalai.db`
* **Restore Utility:** Run `restore-calibration` from any terminal to re-apply these offsets to any new project in 1 second.

**Recorded Offsets Summary:**
* **Follower (`5B91030784`):**
  * `shoulder_pan`: offset `114`, range `[844, 3349]`
  * `shoulder_lift`: offset `584`, range `[856, 3179]`
  * `elbow_flex`: offset `-142`, range `[859, 3061]`
  * `wrist_flex`: offset `269`, range `[1388, 3140]`
  * `wrist_roll`: offset `-832`, range `[144, 4006]`
  * `gripper`: offset `-938`, range `[2015, 3486]`
* **Leader (`5B90148383`):**
  * `shoulder_pan`: offset `-1475`, range `[718, 3391]`
  * `shoulder_lift`: offset `-1260`, range `[865, 3234]`
  * `elbow_flex`: offset `1771`, range `[846, 3040]`
  * `wrist_flex`: offset `-1252`, range `[942, 3221]`
  * `wrist_roll`: offset `1404`, range `[152, 3962]`
  * `gripper`: offset `1602`, range `[2032, 2710]`

---

## 4. Software Stack & Environment Layout
Three isolated Python environments exist on this laptop to prevent library collisions:

| Environment | Path / Port | Python Version | Installed Packages / Purpose |
| :--- | :--- | :--- | :--- |
| **Physical AI Studio (Backend)** | Port `7860` (`/home/intel/physical-ai-studio/application/backend`) | Python 3.13 venv | `physicalai`, `physicalai-so101-plugin`, `openvino==2026.4.1`, `torch==2.11.0+xpu`, `modal`, `pyrealsense2`. |
| **Physical AI Studio (UI)** | Port `3000` (`/home/intel/physical-ai-studio/application/ui`) | Node.js v26 / Rsbuild | React Spectrum, Three.js 3D visualizer, WebSocket telemetry. |
| **Anomalib Environment** | `/home/intel/anomalib-env` | Python 3.12 venv | `anomalib==2.0.0`, `torch==2.15.0+xpu`, `torchvision==0.29.2+xpu`, `openvino==2026.4.1`, `lightning`, `timm`, `kornia`. |
| **Dictation Environment** | `/home/intel/dictate-env` | Python 3.12 venv | `faster-whisper`, `ctranslate2`, powers the global `dictate` CLI. |

---

## 5. Active Studio State
* **Active Project Name:** `Removing the Blocker` (ID: `1e7414d6-c109-4f49-b204-469fef5d6943`)
* **Active Environment:** `Funny_Block_Landia` (ID: `21fa65fe-0e84-4575-976c-bcd6a6cee047`)
* **Active Recorded Dataset:** `photo_test_set` (ID: `06f21646-a3c2-4482-908c-41cba2c3e6e7`):
  * **Total Episodes:** 9 episodes recorded via teleoperation.
  * **Total Frames:** 4,475 synchronized frames at 30 fps across all 3 camera streams.

---

## 6. The Approved Winning MVP Architecture (Updated: ADR-014)
**Decision:** Prioritize **Two Discrete ACT Models** first (fast, deterministic, robust) ahead of SmolVLA, with **Pre-Movement Anomalib Visual Dispatch**:

```text
1. [PRE-MOVEMENT ANOMALIB GATE]  ──►  2. [SPECIALIZED POLICY DISPATCH]
   OpenVINO PatchCore on Xe3 iGPU         Directly activates corresponding ACT policy:
   evaluates tabletop block prior         • Target Green (e.g. Small / Good)  ──► ACT_Block_To_Green
   to any robotic arm movement            • Target Pink  (e.g. Big / Defect)  ──► ACT_Block_To_Pink
```

### Why this architecture wins:
1. **Zero Mid-Air Trajectory Stitching:** Inspecting before motion begins eliminates kinematic handover glitches, paused mid-air states, and multi-goal policy confusion.
2. **Policy Specialization & Robustness:** Each ACT policy learns a single, clean, unimodal trajectory from pickup to target container with $>95\%$ imitation precision.
3. **Fastest Convergence:** ACT trains in under 10 minutes on Modal NVIDIA A10G GPUs (2,765 steps), freeing development time to dial in execution. SmolVLA is deferred as a follow-up exploration once this robust baseline is verified.
4. **Maximum Rubric Alignment:** Fulfills every single requirement of the 100-point judging criteria (Anomalib integration on Intel Xe3 iGPU, policy control, and OpenVINO hardware offload).

---

## 7. Immediate Next Steps Roadmap

### **Step 4: Detect Anomalies with Anomalib (COMPLETED & VERIFIED)**
* **Trained Model:** PatchCore (`wide_resnet50_2`, coreset ratio = 0.2, k=9) trained on 20 normal small-block frames from the wrist claw camera (`claw_cam`).
* **Test Evaluation Results (8 out-of-sample Normals, 16 Big-Block Abnormals):**
  * **Image AUROC:** **`1.0` (100% Perfect Separation)**
  * **Image F1-Score:** **`0.9677`**
  * **Normal Score Range (Small):** `0.0788` – `0.4562` (Mean: `0.1706`)
  * **Abnormal Score Range (Big):** `0.5000` – `1.0000` (Mean: `0.6867`)
  * **Separation Gap:** **`+0.0438`** (Zero false positives, zero false negatives)
  * **Recommended Decision Threshold:** **`0.4781`**
* **OpenVINO Hardware Acceleration Benchmark (ADR-015):**
  * **Model Files:** `/home/intel/models/anomalib_patchcore_claw/weights/openvino/patchcore_claw.xml` (`.bin`)
  * **Intel Core Ultra Xe3 iGPU (`GPU`):** **`3.78 ms`** per frame (**`264.4 FPS`**)
  * **Intel CPU:** `74.99 ms` (Xe3 iGPU is **~20× faster**)
* **Execution Script:** `python3 scripts/train_anomalib_patchcore.py`

### **Step 4b: Final Side-View Inspection Model, Anomalib Test 3 (COMPLETED, 15 Nov 2026)**
* **Supersedes the Step 4 claw-cam model for the defect gate.** Side camera, photo taken with the arm parked at the reset pose, cropped to `INSPECTION_CROP` = x 60, y 130, 54 × 49 px (`algorithm/reset_pose_trigger.py`).
* **Data:** 29 normal training crops (19 Theo Frames v2 + 10 from `datasets/90_degree_turn` via the reset-pose trigger); test 8 normal + 16 abnormal.
* **Results:** Image AUROC **0.961** (all 24), **1.00** excluding ep33, whose defect is not visible from the side. **Decision threshold 0.37.**
* **Model files:** not in git (127 MB); on the training laptop under `C:\Users\theoh\anomalib_runs\test_3\side_view_no_ep05\export\`. Must be copied to the robot laptop.
* **Not yet wired into the live code:** `config.py`, `scripts/sorting_middleware.py` and `website/server.py` still use the claw model and 0.4781. See `README.md` §4 and `anomalib/test_3/findings.md`.

### **Step 5: Record Full Demonstrations & Train Policy on Modal (READY TO RUN)**
1. **Recorded Datasets Available:**
   * `Block to Green` (`ea625a1c-d316-4a7b-88f0-43838fbf9364`): 22 episodes.
   * `Block to Pink` (`d9b531e6-72c5-4174-90ab-299100d158b0`): 22 episodes.
2. **Modal Cloud Trainer Status (VERIFIED HEALTHY):**
   * **Profile:** `siongan`
   * **Deployed App:** `physicalai-trainer` (`train` function with NVIDIA A10G, 8 vCPUs, 32GB RAM, volume `physicalai-training`)
   * **Studio Remote Trainer:** `modal-a10g-trainer` (ID: `e2b5ba5f-0166-4a83-b456-4dac7b31c512`, Health: `healthy`, Latency: ~289ms)
3. **Dispatching Training:**
   * In Studio UI (`http://localhost:3000`), navigate to **Training** $\rightarrow$ select **ACT** or **SmolVLA** $\rightarrow$ select Remote Trainer `modal-a10g-trainer` $\rightarrow$ click **Start Training**.
   * Or dispatch directly via API (`POST /api/jobs:train`).
4. Download the trained policy checkpoint once completed.

### **Step 6: Close the Loop (Autonomous Python Execution)**
Write the final Python execution loop connecting the pieces:
1. Load the trained policy with `physicalai.runtime.InferenceModel`.
2. Arm reaches and lifts the block to the inspection pose.
3. Snaps frame $\rightarrow$ runs OpenVINO Anomalib inference on Intel GPU/NPU.
4. If `anomaly_score > threshold` $\rightarrow$ place in Defect Bowl. Else $\rightarrow$ place in Good Bowl.

### **Step 7: Optimize on Intel Hardware**
Profile and log metrics for the presentation:
* Anomalib inference time on Intel Xe3 iGPU vs. CPU vs. NPU.
* Control loop latency (aiming for steady 30 Hz).
* Power consumption and thermal headroom on the Core Ultra X9 388H.

---

## 8. Troubleshooting & Gotchas (What We Already Fixed)
1. **UI Camera Spinner Loads Indefinitely:**
   * *Cause:* Dropped USB connections leave stale `iceoryx2` shared memory files in `/dev/shm`.
   * *Fix:* Run `rm -f /dev/shm/iox2_* /dev/shm/sem.mp-*` and restart backend. (Automated inside `start-studio`).
2. **`RuntimeError: Could not load libtorchcodec (missing libavutil.so.60)`:**
   * *Cause:* LeRobot dataloader requires system FFmpeg libraries to decode dataset MP4s.
   * *Fix:* Already resolved. FFmpeg 8 libraries (`libavutil60`, `libavcodec60`) are installed.
3. **`usb 3-1.4.2: Not enough bandwidth for new device state`:**
   * *Cause:* Two UVC cameras attempting uncompressed YUYV at $640\times 480$ on the same USB 2.0 hub branch.
   * *Fix:* Always set Tabletop Webcam and Claw Cam to **`320x240 @ 30fps`** in Studio.
4. **Robot Ports Swapped or Renamed:**
   * *Cause:* USB re-plugging changes `/dev/ttyACM*`.
   * *Fix:* Studio automatically maps robots by hardware serial number. Run `restore-calibration` to verify.

---

## 9. Global Command Cheat Sheet
All commands can be run from any terminal:

* **Start Physical AI Studio:**
  ```bash
  start-studio
  ```
  *(Launches Backend on port 7860 and UI on port 3000)*
* **Stop Physical AI Studio:**
  ```bash
  stop-studio
  ```
* **Restore Robot Calibration Instantly:**
  ```bash
  restore-calibration
  ```
* **View Architecture Decision Log (ADR-001 through ADR-010):**
  ```bash
  decide
  # or inspect single decision:
  decide show ADR-007
  ```
* **Capture Test Snapshots from All Cameras:**
  ```bash
  snap-comparison
  ```
  *(Saves side-by-side view to `~/camera_comparison/comparison.jpg`)*
* **Offline Voice Dictation (Mic to Text):**
  ```bash
  dictate
  ```

---

*This master context is the authoritative blueprint for the project. Keep this document updated as new ADRs and checkpoints are established.*
