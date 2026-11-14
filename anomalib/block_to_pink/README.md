# Block to Pink: Starting Frames (22 Episodes)

This folder contains the extracted starting frames (frame 0, before robotic arm movement) from all 22 episodes of the `Block to Pink` dataset recorded on the LeRobot SO-101 arm.

## Camera Structure

* **`claw_cam/`**: 22 starting frames from the wrist-mounted claw camera (320x240 RGB, `pink_ep00.png` through `pink_ep21.png`).
* **`side_view/`**: 22 starting frames from the tabletop side camera (320x240 RGB, `pink_ep00.png` through `pink_ep21.png`).
* **`top_view/`**: 22 starting frames from the overhead RealSense D415 camera (640x480 RGB, `pink_ep00.png` through `pink_ep21.png`).

## Semantic Ground Truth (ADR-013)
* **Big Blocks (Abnormal / Defect):** Episodes `00`, `04`, `05`, `06`, `13`, `14`, `18`, `19`, `20`, `21` (10 episodes).
* **Small Blocks (Normal):** All remaining 12 episodes (`01`, `02`, `03`, `07`, `08`, `09`, `10`, `11`, `12`, `15`, `16`, `17`).
