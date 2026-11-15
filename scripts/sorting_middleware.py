#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Sorting Middleware: Closed-Loop Active Inspection & Routing Pipeline
===================================================================
Junction 2026 Intel Physical AI Challenge (SO-101 Robotic Arm on Intel Core Ultra)

Execution Flow:
  1. Active Inspection: 4 consecutive 90° block turns with set-down + retract.
  2. Frame Capture Hook: Snap inspection photo of each resting face.
  3. Decision Gate: Anomalib OpenVINO PatchCore inference on Intel Xe3 iGPU / NPU.
  4. Routing Dispatch:
       - Defect Detected (True)  -> Execute Action: Take Block to Pink Plate
       - Normal / Good (False)   -> Execute Action: Take Block to Green Plate
  5. Telemetry & Summary Logging (scores, latency, Intel hardware offload stats).

Usage:
  # Dry-run test (simulates arm and camera, tests state machine):
  /home/intel/anomalib-env/bin/python3 scripts/sorting_middleware.py --dry-run

  # Run with OpenVINO PatchCore model on Intel Xe3 iGPU:
  /home/intel/anomalib-env/bin/python3 scripts/sorting_middleware.py \
      --openvino-model /home/intel/models/anomalib_patchcore_claw/weights/openvino/patchcore_claw.xml \
      --device GPU
"""

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure formatted logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sorting_middleware")


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class InspectionResult:
    """Holds the multi-view inspection metrics and final classification decision."""
    is_defective: bool
    max_score: float
    decision_threshold: float
    per_side_scores: List[float]
    inference_latencies_ms: List[float]
    total_ai_latency_ms: float
    device_used: str
    timestamp: str


# ==============================================================================
# 1. CAMERA INTERFACE & HOOKS
# ==============================================================================

class BaseCameraManager:
    """Abstract interface for capturing camera frames after each turn."""

    def initialize(self):
        """Open camera feeds or initialize IPC streaming."""
        pass

    def capture_frame(self, side_index: int) -> Optional[np.ndarray]:
        """
        Snap a frame for the given inspection side (0 to 3).
        Returns an RGB image as a numpy array with shape (H, W, 3), uint8.
        """
        raise NotImplementedError

    def release(self):
        """Clean up camera resources."""
        pass


class MockCameraManager(BaseCameraManager):
    """Synthetic camera used for testing and dry-run validation."""

    def __init__(self, height: int = 240, width: int = 320):
        self.height = height
        self.width = width
        logger.info(f"[CAMERA] Initialized Mock Camera ({width}x{height})")

    def capture_frame(self, side_index: int) -> np.ndarray:
        logger.info(f"  [CAMERA HOOK] Snapped frame for Side {side_index + 1}/4 (Synthetic RGB)")
        # Returns a dummy uint8 image (simulated block)
        dummy_frame = (np.random.rand(self.height, self.width, 3) * 255).astype(np.uint8)
        return dummy_frame


class OpenCVCameraManager(BaseCameraManager):
    """
    OpenCV implementation for live webcam capture (Tabletop or Claw Cam).
    Plug in device index (e.g., /dev/video32 or /dev/video40).
    """

    def __init__(self, device_path: str = "/dev/video32", width: int = 320, height: int = 240):
        self.device_path = device_path
        self.width = width
        self.height = height
        self.cap = None

    def initialize(self):
        try:
            import cv2
            logger.info(f"[CAMERA] Opening camera device: {self.device_path}")
            self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                logger.warning(f"[CAMERA] Could not open {self.device_path}. Falling back to synthetic.")
        except Exception as e:
            logger.warning(f"[CAMERA] OpenCV init failed: {e}")

    def capture_frame(self, side_index: int) -> Optional[np.ndarray]:
        if self.cap and self.cap.isOpened():
            import cv2
            # Flush buffer to get latest frame
            for _ in range(3):
                ret, frame = self.cap.read()
            if ret and frame is not None:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                logger.info(f"  [CAMERA HOOK] Live frame captured for Side {side_index + 1} ({rgb_frame.shape})")
                return rgb_frame
        logger.warning(f"  [CAMERA HOOK] Live frame unavailable; generating synthetic frame.")
        return (np.random.rand(self.height, self.width, 3) * 255).astype(np.uint8)

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("[CAMERA] Released camera device.")


# ==============================================================================
# 2. DEFECT DETECTOR INTERFACE & OPENVINO INFERENCE
# ==============================================================================

class BaseDefectDetector:
    """Abstract interface for multi-view defect classification."""

    def evaluate(self, frames: List[np.ndarray]) -> InspectionResult:
        raise NotImplementedError


class MockDefectDetector(BaseDefectDetector):
    """Mock detector for dry-run testing with force flags or simulated scores."""

    def __init__(self, force_defect: Optional[bool] = None, threshold: float = 0.4781):
        self.force_defect = force_defect
        self.threshold = threshold

    def evaluate(self, frames: List[np.ndarray]) -> InspectionResult:
        logger.info("  [AI HOOK] Running Mock Defect Detector on %d frames...", len(frames))
        t0 = time.perf_counter()

        if self.force_defect is True:
            scores = [0.12, 0.25, 0.82, 0.31]  # Side 3 is defective
        elif self.force_defect is False:
            scores = [0.11, 0.14, 0.18, 0.15]  # All normal
        else:
            # Random scores around threshold
            scores = [float(np.random.uniform(0.08, 0.45)) for _ in frames]

        max_score = max(scores)
        is_defective = max_score >= self.threshold
        elapsed = (time.perf_counter() - t0) * 1000.0

        return InspectionResult(
            is_defective=is_defective,
            max_score=max_score,
            decision_threshold=self.threshold,
            per_side_scores=scores,
            inference_latencies_ms=[elapsed / len(frames)] * len(frames),
            total_ai_latency_ms=elapsed,
            device_used="MOCK_CPU",
            timestamp=datetime.now().isoformat(),
        )


class OpenVINODefectDetector(BaseDefectDetector):
    """
    OpenVINO PatchCore Inferencer running on Intel Xe3 iGPU or NPU.
    Directly loads the exported OpenVINO IR model (.xml / .bin).
    """

    def __init__(
        self,
        model_xml_path: str,
        device: str = "GPU",
        threshold: float = 0.4781,
        input_size: Tuple[int, int] = (240, 320),  # (H, W)
    ):
        self.model_xml_path = Path(model_xml_path)
        self.device = device
        self.threshold = threshold
        self.input_size = input_size
        self.compiled_model = None
        self._init_model()

    def _init_model(self):
        import openvino as ov
        core = ov.Core()
        available_devices = core.available_devices
        logger.info(f"[INTEL OPENVINO] Available devices: {available_devices}")

        target_dev = self.device
        if target_dev not in available_devices:
            logger.warning(
                f"[INTEL OPENVINO] Requested device '{target_dev}' not in available list. "
                f"Falling back to CPU."
            )
            target_dev = "CPU"

        logger.info(f"[INTEL OPENVINO] Reading model: {self.model_xml_path}")
        model = core.read_model(str(self.model_xml_path))

        # Enforce static shape for Intel NPU Level Zero compiler compatibility
        h, w = self.input_size
        try:
            model.reshape([1, 3, h, w])
        except Exception as reshape_err:
            logger.debug(f"[INTEL OPENVINO] Model reshape skipped: {reshape_err}")

        logger.info(f"[INTEL OPENVINO] Compiling model for target: {target_dev} (Core Ultra Xe3/NPU)")
        self.compiled_model = core.compile_model(model, target_dev)
        self.actual_device = target_dev

        # Warmup model
        dummy = np.random.randn(1, 3, self.input_size[0], self.input_size[1]).astype(np.float32)
        for _ in range(3):
            _ = self.compiled_model([dummy])
        logger.info("[INTEL OPENVINO] Model compiled and warmed up successfully.")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resizes and normalizes frame to shape (1, 3, H, W), float32 in [0, 1]."""
        # Simple resize if needed
        h, w = self.input_size
        if frame.shape[0] != h or frame.shape[1] != w:
            import cv2
            frame = cv2.resize(frame, (w, h))

        # (H, W, C) -> (C, H, W)
        chw = frame.transpose((2, 0, 1)).astype(np.float32)
        if chw.max() > 1.0:
            chw = chw / 255.0

        # Add batch dimension: (1, 3, H, W)
        return np.expand_dims(chw, axis=0)

    def evaluate(self, frames: List[np.ndarray]) -> InspectionResult:
        logger.info(f"[AI DECISION GATE] Running OpenVINO inference on {len(frames)} views ({self.actual_device})...")
        scores = []
        latencies = []

        for i, frame in enumerate(frames):
            inp = self._preprocess_frame(frame)
            t0 = time.perf_counter()
            outputs = self.compiled_model([inp])
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)

            # Output 'pred_score' is the continuous anomaly score
            # OpenVINO output key can be accessed via output tensor name or index
            score = None
            for key, val in outputs.items():
                if key.get_any_name() == "pred_score":
                    score = float(val.flat[0])
                    break
            if score is None:
                # Fallback to first output
                first_val = list(outputs.values())[0]
                score = float(first_val.flat[0])

            scores.append(score)
            logger.info(
                f"  Side {i + 1}/4 -> Anomaly Score: {score:.4f} | Latency: {dt_ms:.2f} ms "
                f"({'ABNORMAL' if score >= self.threshold else 'NORMAL'})"
            )

        max_score = max(scores)
        is_defective = max_score >= self.threshold
        total_ai_latency = sum(latencies)

        logger.info(
            f"[AI RESULT] Max Score: {max_score:.4f} (Threshold: {self.threshold:.4f}) | "
            f"Decision: {'DEFECT DETECTED -> PINK' if is_defective else 'GOOD BRICK -> GREEN'} | "
            f"Total AI Time: {total_ai_latency:.2f} ms"
        )

        return InspectionResult(
            is_defective=is_defective,
            max_score=max_score,
            decision_threshold=self.threshold,
            per_side_scores=scores,
            inference_latencies_ms=latencies,
            total_ai_latency_ms=total_ai_latency,
            device_used=self.actual_device,
            timestamp=datetime.now().isoformat(),
        )


# ==============================================================================
# 3. ROBOT ACTION DRIVER & HOOKS
# ==============================================================================

class BaseRobotDriver:
    """Abstract interface for triggering robotic arm motions."""

    def turn_90_degrees(self, step_index: int):
        """Execute a single 90° pick-rotate-place-retract cycle."""
        raise NotImplementedError

    def place_pink(self):
        """Pick up block from inspection position and place on pink plate (Defect)."""
        raise NotImplementedError

    def place_green(self):
        """Pick up block from inspection position and place on green plate (Good)."""
        raise NotImplementedError

    def goto_home(self):
        """Move arm to safe home/inspection clearance pose."""
        raise NotImplementedError

    def emergency_stop(self):
        """Immediate safe stop of all motors."""
        raise NotImplementedError


class MockRobotDriver(BaseRobotDriver):
    """Simulated robot driver for testing control flow and state transitions."""

    def __init__(self, step_delay: float = 1.0):
        self.step_delay = step_delay
        logger.info(f"[ROBOT] Initialized Mock Robot Driver (simulated step delay: {step_delay}s)")

    def turn_90_degrees(self, step_index: int):
        logger.info(f"  [ROBOT HOOK] Starting 90° Turn #{step_index + 1}: Gripping -> Turning -> Setting down -> Retracting...")
        time.sleep(self.step_delay)
        logger.info(f"  [ROBOT HOOK] 90° Turn #{step_index + 1} COMPLETED. Arm in retracted home pose.")

    def place_pink(self):
        logger.info("  [ROBOT HOOK] Executing Action: Take Block to PINK Plate (Defect Sorting)...")
        time.sleep(self.step_delay * 1.5)
        logger.info("  [ROBOT HOOK] Action Take Block to PINK Plate COMPLETED.")

    def place_green(self):
        logger.info("  [ROBOT HOOK] Executing Action: Take Block to GREEN Plate (Good Sorting)...")
        time.sleep(self.step_delay * 1.5)
        logger.info("  [ROBOT HOOK] Action Take Block to GREEN Plate COMPLETED.")

    def goto_home(self):
        logger.info("  [ROBOT HOOK] Arm returning to HOME neutral position.")
        time.sleep(self.step_delay * 0.5)

    def emergency_stop(self):
        logger.warning("  [EMERGENCY STOP] Mock Robot motors stopped immediately.")


class StudioTrajectoryPlayerDriver(BaseRobotDriver):
    """
    Robot Driver that executes actions via Physical AI Studio API or Trajectory Replay.
    Plug in your endpoint or LeRobot execution calls here.
    """

    def __init__(self, studio_url: str = "http://localhost:7860"):
        self.studio_url = studio_url
        logger.info(f"[ROBOT] Initialized Studio Driver connecting to {studio_url}")

    def turn_90_degrees(self, step_index: int):
        # ------------------------------------------------------------------
        # HOOK: Insert your Studio API call or trajectory replay here:
        # e.g.: requests.post(f"{self.studio_url}/api/actions/run", json={"action": "90_degree_turn"})
        # ------------------------------------------------------------------
        logger.info(f"  [STUDIO ACTION] Triggering action '90_degree_turn' (Step {step_index + 1})...")
        # Placeholder delay until Studio API integration is finalized
        time.sleep(2.0)
        logger.info(f"  [STUDIO ACTION] '90_degree_turn' completed.")

    def place_pink(self):
        # ------------------------------------------------------------------
        # HOOK: requests.post(f"{self.studio_url}/api/actions/run", json={"action": "place_block_to_pink"})
        # ------------------------------------------------------------------
        logger.info("  [STUDIO ACTION] Triggering action 'place_block_to_pink'...")
        time.sleep(3.0)
        logger.info("  [STUDIO ACTION] 'place_block_to_pink' completed.")

    def place_green(self):
        # ------------------------------------------------------------------
        # HOOK: requests.post(f"{self.studio_url}/api/actions/run", json={"action": "place_block_to_green"})
        # ------------------------------------------------------------------
        logger.info("  [STUDIO ACTION] Triggering action 'place_block_to_green'...")
        time.sleep(3.0)
        logger.info("  [STUDIO ACTION] 'place_block_to_green' completed.")

    def goto_home(self):
        logger.info("  [STUDIO ACTION] Triggering action 'goto_home'...")
        time.sleep(1.0)

    def emergency_stop(self):
        logger.warning("  [EMERGENCY STOP] Sending abort signal to robot backend...")


class PhysicalSO101RobotDriver(BaseRobotDriver):
    """
    Direct hardware driver for LeRobot SO-101 Follower Arm via Feetech bus serial port.
    Replays calibrated 30 Hz joint trajectories with smooth cosine lead-in interpolation.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        calibration_path: str = "/home/intel/so101_calibration_backup.json",
        dataset_root: Path = Path("/home/intel/Junction/datasets"),
    ):
        self.port = port
        self.calibration_path = Path(calibration_path)
        self.dataset_root = dataset_root
        self.robot = None
        self._init_hardware()

    def _init_hardware(self):
        try:
            import json
            from physicalai_so101_plugin.so101 import SO101

            logger.info(f"[SO-101 HARDWARE] Loading calibration from {self.calibration_path}...")
            with open(self.calibration_path) as f:
                calib_data = json.load(f)["SO101 Follower"]

            logger.info(f"[SO-101 HARDWARE] Connecting to Follower Arm on {self.port}...")
            self.robot = SO101(
                port=self.port,
                calibration=calib_data,
                role="follower",
                unit="degrees",
            )
            self.robot.connect()
            obs = self.robot.get_observation()
            logger.info(
                f"[SO-101 HARDWARE] Connected! Current joint positions: "
                f"{[round(float(p), 1) for p in obs.joint_positions]}"
            )
        except Exception as e:
            logger.error(f"[SO-101 HARDWARE] Connection failed: {e}")
            raise

    def _smooth_lead_in(self, target_pose: np.ndarray, duration_sec: float = 1.5):
        """Smoothly interpolates arm from current joint angles to target_pose using cosine ease."""
        if not self.robot or not self.robot.is_connected:
            return
        obs = self.robot.get_observation()
        current_pose = np.array(obs.joint_positions, dtype=np.float32)

        steps = max(1, int(duration_sec * 30.0))
        dt = duration_sec / steps
        for step in range(steps + 1):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (step / steps)))
            interp_pose = (1.0 - alpha) * current_pose + alpha * target_pose
            self.robot.send_action(interp_pose.astype(np.float32))
            time.sleep(dt)

    def _replay_csv(self, csv_path: Path):
        """Loads and replays 30 Hz joint trajectory CSV on physical SO-101 arm."""
        import pandas as pd
        from physicalai_so101_plugin.so101 import SO101

        logger.info(f"[SO-101 TRAJECTORY] Loading {csv_path.name}...")
        df = pd.read_csv(csv_path)
        action_cols = [f"action_{j}" for j in SO101.JOINT_ORDER]
        actions = df[action_cols].values.astype(np.float32)

        # 1. Smooth lead-in to starting frame
        logger.info(f"  [SO-101 LEAD-IN] Moving smoothly to initial trajectory pose...")
        self._smooth_lead_in(actions[0], duration_sec=1.5)

        # 2. Stream actions at 30 Hz (33.3 ms per frame)
        logger.info(f"  [SO-101 STREAM] Streaming {len(actions)} trajectory frames at 30 Hz...")
        t_prev = time.perf_counter()
        target_dt = 1.0 / 30.0
        for action in actions:
            self.robot.send_action(action)
            # Maintain 30 Hz timing
            t_now = time.perf_counter()
            elapsed = t_now - t_prev
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            t_prev = time.perf_counter()

        logger.info(f"[SO-101 TRAJECTORY] Trajectory {csv_path.name} completed successfully.")

    def turn_90_degrees(self, step_index: int):
        csv_path = self.dataset_root / "90_degree_turn" / "trajectories" / "episode_00_joints.csv"
        logger.info(f"  [SO-101 ACTION] Executing 90° Turn #{step_index + 1} on physical follower arm...")
        self._replay_csv(csv_path)

    def place_pink(self):
        csv_path = self.dataset_root / "block_to_pink_v5" / "trajectories" / "episode_00_joints.csv"
        logger.info("  [SO-101 ACTION] Executing Block -> Pink Plate on physical follower arm...")
        self._replay_csv(csv_path)

    def place_green(self):
        csv_path = self.dataset_root / "block_to_green_v4" / "trajectories" / "episode_00_joints.csv"
        logger.info("  [SO-101 ACTION] Executing Block -> Green Plate on physical follower arm...")
        self._replay_csv(csv_path)

    def goto_home(self):
        logger.info("  [SO-101 ACTION] Arm moving to resting Home position...")
        home_pose = np.array([-5.0, -100.0, 98.0, 60.0, -50.0, 2.0], dtype=np.float32)
        self._smooth_lead_in(home_pose, duration_sec=1.5)

    def emergency_stop(self):
        if self.robot and self.robot.is_connected:
            logger.warning("  [EMERGENCY STOP] Disconnecting SO-101 follower arm and releasing torque!")
            try:
                self.robot.disconnect()
            except Exception:
                pass


# ==============================================================================
# 4. SORTING MIDDLEWARE ORCHESTRATOR (FINITE STATE MACHINE)
# ==============================================================================

class SortingMiddleware:
    """
    Finite State Machine orchestrating the full closed-loop inspection:
      1. Active Inspection (4x 90° rotations & set-downs)
      2. Frame Capture Hook (4 photos)
      3. OpenVINO Defect Decision Gate
      4. Routing Dispatch (Pink vs. Green)
    """

    def __init__(
        self,
        robot: BaseRobotDriver,
        camera: BaseCameraManager,
        detector: BaseDefectDetector,
        settle_delay_sec: float = 0.2,
        save_artifacts: bool = True,
        output_dir: Path = Path("/home/intel/Junction/runs"),
    ):
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.settle_delay_sec = settle_delay_sec
        self.save_artifacts = save_artifacts
        self.output_dir = output_dir
        self.is_aborted = False

        # Setup graceful SIGINT (Ctrl+C) handling
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        logger.warning("\n[INTERRUPT] SIGINT detected! Halting middleware safely...")
        self.is_aborted = True
        self.robot.emergency_stop()
        sys.exit(0)

    def run_cycle(self) -> Optional[InspectionResult]:
        cycle_start_time = time.perf_counter()
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = self.output_dir / f"run_{run_timestamp}"

        if self.save_artifacts:
            run_folder.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info(f" STARTING AUTONOMOUS INSPECTION & SORTING CYCLE [{run_timestamp}]")
        logger.info("=" * 70)

        # ----------------------------------------------------------------------
        # PHASE 1: 4-SIDE ACTIVE INSPECTION (4 x 90° TURNS)
        # ----------------------------------------------------------------------
        logger.info("\n>>> PHASE 1: ACTIVE INSPECTION (4 x 90° TURNS)")
        captured_frames: List[np.ndarray] = []

        for step in range(4):
            if self.is_aborted:
                return None

            logger.info(f"\n[TURN {step + 1}/4] Commanding 90° turn and set-down...")
            self.robot.turn_90_degrees(step_index=step)

            # Wait for physical vibrations to damp out before snapping
            if self.settle_delay_sec > 0:
                time.sleep(self.settle_delay_sec)

            # Frame Capture Hook
            frame = self.camera.capture_frame(side_index=step)
            if frame is not None:
                captured_frames.append(frame)
                if self.save_artifacts:
                    # Save frame for debugging and demonstration
                    try:
                        import cv2
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(run_folder / f"side_{step + 1}.png"), bgr)
                    except Exception:
                        pass

        logger.info("\n>>> Active inspection completed. All 4 sides captured.")
        logger.info(f">>> Block has completed 360° turn and is back in initial orientation.")

        # ----------------------------------------------------------------------
        # PHASE 2: ANOMALIB OPENVINO DECISION GATE
        # ----------------------------------------------------------------------
        logger.info("\n>>> PHASE 2: DEFECT DETECTION GATE")
        result = self.detector.evaluate(captured_frames)

        # ----------------------------------------------------------------------
        # PHASE 3: ROUTING DISPATCH (PINK VS GREEN)
        # ----------------------------------------------------------------------
        logger.info("\n>>> PHASE 3: ROUTING DISPATCH")
        if result.is_defective:
            logger.info(">>> DECISION: DEFECT DETECTED -> ROUTING TO PINK PLATE")
            self.robot.place_pink()
        else:
            logger.info(">>> DECISION: NORMAL BLOCK -> ROUTING TO GREEN PLATE")
            self.robot.place_green()

        # ----------------------------------------------------------------------
        # PHASE 4: RETURN HOME & LOG TELEMETRY
        # ----------------------------------------------------------------------
        self.robot.goto_home()
        total_cycle_time = time.perf_counter() - cycle_start_time

        logger.info("\n" + "=" * 70)
        logger.info(" CYCLE SUMMARY & INTEL HARDWARE TELEMETRY")
        logger.info("=" * 70)
        logger.info(f"Final Decision:         {'DEFECT (Pink Plate)' if result.is_defective else 'GOOD (Green Plate)'}")
        logger.info(f"Max Anomaly Score:      {result.max_score:.4f} (Threshold: {result.decision_threshold:.4f})")
        logger.info(f"Per-Side Scores:        {[round(s, 4) for s in result.per_side_scores]}")
        logger.info(f"OpenVINO Inference Dev: {result.device_used}")
        logger.info(f"AI Decision Time:       {result.total_ai_latency_ms:.2f} ms")
        logger.info(f"Total Cycle Time:       {total_cycle_time:.2f} s")
        logger.info("=" * 70)

        # Save run summary report
        if self.save_artifacts:
            report_data = {
                "run_timestamp": run_timestamp,
                "total_cycle_time_seconds": total_cycle_time,
                "inspection_result": asdict(result),
            }
            report_path = run_folder / "summary.json"
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"Run artifacts and metrics saved to: {run_folder}")

        return result


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Sorting Middleware with OpenVINO Defect Gate (Intel Physical AI Challenge)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in mock/dry-run mode without requiring physical robot hardware or live cameras.",
    )
    parser.add_argument(
        "--mock-robot",
        action="store_true",
        help="Simulate robot arm movements without commanding physical hardware.",
    )
    parser.add_argument(
        "--mock-camera",
        action="store_true",
        help="Simulate camera captures without opening physical video device.",
    )
    parser.add_argument(
        "--use-openvino",
        action="store_true",
        help="Force loading and running real OpenVINO model even in dry-run mode.",
    )
    parser.add_argument(
        "--openvino-model",
        type=str,
        default="/home/intel/models/anomalib_patchcore_claw/weights/openvino/patchcore_claw.xml",
        help="Path to exported OpenVINO PatchCore .xml model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["GPU", "NPU", "CPU"],
        help="Intel hardware target for OpenVINO inference (default: GPU for Xe3 iGPU).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4781,
        help="Anomalib anomaly score decision threshold (default: 0.4781).",
    )
    parser.add_argument(
        "--force-defect",
        action="store_true",
        help="Simulate a defective brick (forces branch to Pink Plate in dry-run).",
    )
    parser.add_argument(
        "--force-normal",
        action="store_true",
        help="Simulate a normal brick (forces branch to Green Plate in dry-run).",
    )
    parser.add_argument(
        "--camera-device",
        type=str,
        default="/dev/video32",
        help="V4L2 device path for inspection camera (default: /dev/video32).",
    )
    parser.add_argument(
        "--physical-robot",
        action="store_true",
        help="Command the real physical SO-101 Follower arm over serial (/dev/ttyACM0).",
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for physical SO-101 Follower arm (default: /dev/ttyACM0).",
    )
    parser.add_argument(
        "--settle-delay",
        type=float,
        default=0.2,
        help="Delay in seconds to let arm settle before snapping photos (default: 0.2s).",
    )
    parser.add_argument(
        "--test-action",
        type=str,
        choices=["turn", "pink", "green", "home"],
        help="Test a single isolated robot action (e.g. 'turn' for 90° turn) and exit safely.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine mock status
    mock_robot = (args.dry_run or args.mock_robot) and not args.physical_robot
    mock_camera = args.dry_run or args.mock_camera
    use_real_openvino = args.use_openvino or (not args.dry_run and not args.force_defect and not args.force_normal)

    # 1. Select Robot Driver
    if args.physical_robot:
        logger.info(f"[ROBOT DRIVER] Initializing PHYSICAL SO-101 Follower Arm on {args.robot_port}...")
        robot_driver = PhysicalSO101RobotDriver(port=args.robot_port)
    elif mock_robot:
        robot_driver = MockRobotDriver(step_delay=0.3)
    else:
        # Uses Studio trajectory player driver (easily plugged into Physical AI Studio)
        robot_driver = StudioTrajectoryPlayerDriver()

    # If testing a single isolated action, execute it and exit immediately
    if args.test_action:
        try:
            logger.info(f"[TEST ACTION] Running single action: '{args.test_action}'...")
            if args.test_action == "turn":
                robot_driver.turn_90_degrees(0)
            elif args.test_action == "pink":
                robot_driver.place_pink()
            elif args.test_action == "green":
                robot_driver.place_green()
            elif args.test_action == "home":
                robot_driver.goto_home()
            logger.info(f"[TEST ACTION] Single action '{args.test_action}' finished successfully.")
        finally:
            if hasattr(robot_driver, "robot") and robot_driver.robot and robot_driver.robot.is_connected:
                try:
                    robot_driver.robot.disconnect()
                    logger.info("[ROBOT DRIVER] Follower arm disconnected safely.")
                except Exception:
                    pass
        return

    # 2. Select Camera Manager
    if mock_camera:
        camera_manager = MockCameraManager()
    else:
        camera_manager = OpenCVCameraManager(device_path=args.camera_device)
        camera_manager.initialize()

    # 3. Select Defect Detector
    model_path = Path(args.openvino_model)
    if use_real_openvino and model_path.exists():
        try:
            defect_detector = OpenVINODefectDetector(
                model_xml_path=str(model_path),
                device=args.device,
                threshold=args.threshold,
            )
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model ({e}). Falling back to Mock detector.")
            defect_detector = MockDefectDetector(threshold=args.threshold)
    else:
        force = True if args.force_defect else (False if args.force_normal else None)
        defect_detector = MockDefectDetector(force_defect=force, threshold=args.threshold)

    # 4. Initialize and Run Middleware Orchestrator
    middleware = SortingMiddleware(
        robot=robot_driver,
        camera=camera_manager,
        detector=defect_detector,
        settle_delay_sec=args.settle_delay,
        save_artifacts=True,
    )

    try:
        middleware.run_cycle()
    finally:
        camera_manager.release()
        if hasattr(robot_driver, "robot") and robot_driver.robot and robot_driver.robot.is_connected:
            try:
                robot_driver.robot.disconnect()
                logger.info("[ROBOT DRIVER] Follower arm disconnected safely.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
