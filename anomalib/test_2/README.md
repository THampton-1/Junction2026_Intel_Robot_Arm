# Anomalib Test 2: Block to Green v2 Frame Set

This folder contains the extracted starting frames (frame 0) from all 25 episodes of the `Block to Green v2` dataset in Physical AI Studio. 

These frames are intended for validating and benchmarking Anomalib (e.g. PatchCore) on an independent secondary machine/GPU. By having synchronized frames from all three cameras, you can evaluate which camera perspective provides the highest separation margin between different block states.

## Folder Structure

* **`Test 2.1/claw_cam/`**: 25 starting frames from the wrist-mounted claw camera (320x240 RGB).
* **`Test 2.1/top_view/`**: 25 starting frames from the overhead RealSense camera (640x480 RGB).
* **`Test 2.1/side_view/`**: 25 starting frames from the tabletop side webcam (320x240 RGB).
* **`screenshots/`**: UI and setup captures recorded during Test 2 configuration.

## Dataset Characteristics
* **Episodes:** 25 (`green_v2_ep00.png` through `green_v2_ep24.png`)
* **Task:** `Place block on green plate`
* **Content:** All frames represent the state of the workspace *before* the robotic arm begins moving, representing the ideal condition for our real-time pre-movement visual anomaly gate.
