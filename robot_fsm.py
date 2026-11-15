import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QThread, Signal

from config import CONFIG
from ov_models import ACTOpenVINOModel, AnomalibOpenVINOModel

logger = logging.getLogger("robot_fsm")


class FSMState(Enum):
    IDLE = "IDLE"
    ROTATE_90 = "ROTATING"
    SETTLE_AND_INSPECT = "SCANNING"
    SORT_BRICK = "SORTING"
    RETURN_HOME = "RETURNING HOME"
    PAUSED = "PAUSED"
    ESTOP = "E-STOP"


class LatestObservationBuffer:
    """Thread-safe latest observation buffer (overwrite-latest, never unbounded)."""

    def __init__(self):
        self.mutex = QMutex()
        self.top_frame: Optional[np.ndarray] = None
        self.side_frame: Optional[np.ndarray] = None
        self.claw_frame: Optional[np.ndarray] = None
        self.joints = np.array(CONFIG.robot.home_inspect_pose, dtype=np.float32)
        self.gripper_closed = False

    def update(self, top: Optional[np.ndarray], side: Optional[np.ndarray], claw: Optional[np.ndarray], joints: np.ndarray):
        with QMutexLocker(self.mutex):
            if top is not None:
                self.top_frame = top.copy()
            if side is not None:
                self.side_frame = side.copy()
            if claw is not None:
                self.claw_frame = claw.copy()
            self.joints = joints.copy()
            self.gripper_closed = bool(joints[-1] > CONFIG.robot.grasp_threshold)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray, bool]:
        with QMutexLocker(self.mutex):
            top = self.top_frame.copy() if self.top_frame is not None else None
            side = self.side_frame.copy() if self.side_frame is not None else None
            claw = self.claw_frame.copy() if self.claw_frame is not None else None
            return top, side, claw, self.joints.copy(), self.gripper_closed


class RobotControlWorker(QThread):
    """
    Dedicated background worker running at control frequency (30 Hz).
    Single owner of robot hardware and camera streams.
    """

    # UI Signals
    camera_update = Signal(object, object, object)  # top, side, claw
    telemetry_update = Signal(object, bool)         # joints (np.ndarray), is_gripper_closed (bool)
    state_update = Signal(str, str, int, int)       # state_name, detail_str, current_step, max_step
    active_model_update = Signal(str, str)          # model_name, device
    inspection_result = Signal(int, bool, float, object, object)  # face_idx, is_defect, score, raw_crop, annotated_crop
    session_stats_update = Signal(int, int, int, float)           # total, green, pink, pass_rate_pct
    status_message = Signal(str)                                  # info log string for status bar

    def __init__(self, mock_mode: bool = False):
        super().__init__()
        self.mock_mode = mock_mode
        self._running = True
        self.is_paused = False
        self.obs_buffer = LatestObservationBuffer()

        # FSM variables
        self.state = FSMState.IDLE
        self.pre_pause_state = FSMState.IDLE
        self.face_index = 0
        self.defect_found = False
        self.force_next_step_flag = False

        # Session Metrics
        self.total_inspected = 0
        self.green_count = 0
        self.pink_count = 0

        # Physical Hardware Handles
        self.robot = None
        self.cap_top = None
        self.cap_side = None
        self.cap_claw = None

        # Preloaded OpenVINO Models
        self.act_rotate = None
        self.act_place_green = None
        self.act_place_pink = None
        self.anomalib_detector = None

        # Simulated trajectory buffers for mock mode
        self.mock_step = 0
        self.mock_sample_images = {}
        self._load_mock_assets()

    def _load_mock_assets(self):
        """Loads representative images from datasets for realistic mock camera rendering."""
        base_dir = Path("/home/intel/Junction/anomalib/block_to_green")
        for cam, sub in [("side", "side_view"), ("top", "top_view"), ("claw", "claw_cam")]:
            img_path = base_dir / sub / "green_ep00.png"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                self.mock_sample_images[cam] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                self.mock_sample_images[cam] = None

    def initialize_hardware(self):
        logger.info("[INIT] Preloading all 4 OpenVINO models into memory...")
        self.status_message.emit("Preloading OpenVINO models (Xe3 iGPU / CPU)...")

        # 1. Preload ACT Policies on CPU / GPU
        self.act_rotate = ACTOpenVINOModel(
            CONFIG.ai.act_rotate_90_xml,
            model_name="ACT_90_turn",
            device="CPU"
        )
        self.act_place_green = ACTOpenVINOModel(
            CONFIG.ai.act_place_green_xml,
            model_name="ACT_Place_Green",
            device="CPU"
        )
        self.act_place_pink = ACTOpenVINOModel(
            CONFIG.ai.act_place_pink_xml,
            model_name="ACT_Place_Pink",
            device="CPU"
        )

        # 2. Preload Intel Anomalib PatchCore Model on Xe3 iGPU
        self.anomalib_detector = AnomalibOpenVINOModel(
            CONFIG.ai.anomalib_xml,
            input_size=CONFIG.ai.anomalib_input_size,
            device=CONFIG.ai.openvino_device,
        )
        logger.info("[INIT] All 4 OpenVINO models compiled and ready for zero-latency switching.")

        if not self.mock_mode:
            self.status_message.emit("Connecting to physical cameras and SO-101 robot...")
            # Initialize Physical Cameras
            self.cap_side = cv2.VideoCapture(CONFIG.camera.primary_side_index, cv2.CAP_V4L2)
            self.cap_side.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera.width)
            self.cap_side.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera.height)

            self.cap_top = cv2.VideoCapture(CONFIG.camera.top_index, cv2.CAP_V4L2)
            self.cap_top.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera.width)
            self.cap_top.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera.height)

            self.cap_claw = cv2.VideoCapture(CONFIG.camera.claw_index, cv2.CAP_V4L2)
            self.cap_claw.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera.width)
            self.cap_claw.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera.height)

            # Initialize Follower Arm
            try:
                from physicalai_so101_plugin.so101 import SO101
                with open(CONFIG.robot.calibration_path) as f:
                    calib_data = json.load(f)["SO101 Follower"]

                logger.info(f"[SO-101] Connecting to Follower on {CONFIG.robot.port}...")
                self.robot = SO101(
                    port=CONFIG.robot.port,
                    calibration=calib_data,
                    role="follower",
                    unit="degrees",
                )
                self.robot.connect()
                logger.info("[SO-101] Connected successfully to physical follower arm!")
            except Exception as e:
                logger.error(f"[SO-101] Connection failed: {e}. Switching to mock robot driver.")
                self.robot = None
        else:
            logger.info("[INIT] Running in MOCK hardware mode (simulated servo bus & cameras).")
            self.status_message.emit("Station ready in DRY RUN (MOCK) mode.")

    def _read_cameras(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.mock_mode or not (self.cap_side and self.cap_side.isOpened()):
            # Use base dataset images or synthetic frames
            h, w = CONFIG.camera.height, CONFIG.camera.width
            top = self.mock_sample_images.get("top")
            side = self.mock_sample_images.get("side")
            claw = self.mock_sample_images.get("claw")

            if top is None:
                top = np.full((h, w, 3), 40, dtype=np.uint8)
            else:
                top = top.copy()
            if side is None:
                side = np.full((h, w, 3), 70, dtype=np.uint8)
            else:
                side = side.copy()
            if claw is None:
                claw = np.full((h, w, 3), 50, dtype=np.uint8)
            else:
                claw = claw.copy()

            # Add subtle dynamic noise to simulate live camera streaming
            noise = np.random.randint(-2, 3, side.shape, dtype=np.int16)
            side = np.clip(side.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Simulate black marker dot defect on Face 3 (face_index == 2) inside the ROI
            if self.face_index == 2:
                rx, ry, rw, rh = CONFIG.ai.roi_x, CONFIG.ai.roi_y, CONFIG.ai.roi_w, CONFIG.ai.roi_h
                cx = rx + rw // 2
                cy = ry + rh // 2
                # Draw black marker dot defect on side camera
                cv2.circle(side, (cx, cy), 9, (20, 20, 20), -1)

            return top, side, claw

        # Physical Camera Capture
        top, side, claw = None, None, None
        w_t, h_t = CONFIG.camera.width, CONFIG.camera.height
        if self.cap_top and self.cap_top.isOpened():
            ret, frame = self.cap_top.read()
            if ret and frame is not None:
                if frame.shape[1] != w_t or frame.shape[0] != h_t:
                    frame = cv2.resize(frame, (w_t, h_t))
                top = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.cap_side and self.cap_side.isOpened():
            ret, frame = self.cap_side.read()
            if ret and frame is not None:
                if frame.shape[1] != w_t or frame.shape[0] != h_t:
                    frame = cv2.resize(frame, (w_t, h_t))
                side = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.cap_claw and self.cap_claw.isOpened():
            ret, frame = self.cap_claw.read()
            if ret and frame is not None:
                if frame.shape[1] != w_t or frame.shape[0] != h_t:
                    frame = cv2.resize(frame, (w_t, h_t))
                claw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return top, side, claw

    def _read_robot_state(self) -> np.ndarray:
        if self.mock_mode or not self.robot or not getattr(self.robot, "is_connected", False):
            # Generate realistic simulated joint trajectory based on current FSM state
            base_home = np.array(CONFIG.robot.home_inspect_pose, dtype=np.float32)
            if self.state == FSMState.IDLE or self.state == FSMState.SETTLE_AND_INSPECT:
                return base_home

            if self.state == FSMState.ROTATE_90:
                # Simulate smooth rotation cycle:
                # Step 0-25: Approach & Open
                # Step 25-50: Grasp block (gripper rises to 5.0)
                # Step 50-80: Rotate (shoulder pan rotates 90 deg)
                # Step 80-100: Release block (gripper drops to 1.8)
                # Step 100+: Retract back
                step = self.mock_step
                sim_joints = base_home.copy()
                if step < 25:
                    sim_joints[-1] = 1.6
                elif step < 65:
                    sim_joints[-1] = 4.8  # Grasped!
                    sim_joints[0] = -5.0 + ((step - 25) / 40.0) * 30.0
                elif step < 95:
                    sim_joints[-1] = 1.7  # Released!
                    sim_joints[0] = 25.0 - ((step - 65) / 30.0) * 30.0
                else:
                    sim_joints = base_home
                return sim_joints

            if self.state == FSMState.SORT_BRICK:
                # Simulate pick and place to green or pink
                step = self.mock_step
                sim_joints = base_home.copy()
                target_pan = -65.0 if self.defect_found else 65.0
                if step < 30:
                    sim_joints[-1] = 1.6
                elif step < 75:
                    sim_joints[-1] = 4.8  # Grasp
                    sim_joints[0] = target_pan * ((step - 30) / 45.0)
                elif step < 110:
                    sim_joints[-1] = 1.7  # Release over bowl
                    sim_joints[0] = target_pan
                else:
                    sim_joints = base_home
                return sim_joints

            return base_home

        # Read Physical Robot Telemetry
        obs = self.robot.get_observation()
        return np.array(obs.joint_positions, dtype=np.float32)

    def _send_robot_action(self, target_joints: np.ndarray):
        if not self.mock_mode and self.robot and getattr(self.robot, "is_connected", False):
            self.robot.send_action(target_joints.astype(np.float32))

    def run(self):
        self.initialize_hardware()
        target_dt = 1.0 / CONFIG.robot.control_hz

        # State machine tracking variables
        action_step = 0
        grasp_detected = False
        consecutive_grasps = 0
        state_timer_start = 0.0
        return_home_start_pose = None

        while self._running:
            loop_start = time.perf_counter()

            # 1. Observation Capture (Single Hardware Owner)
            top, side, claw = self._read_cameras()
            joints = self._read_robot_state()
            self.obs_buffer.update(top, side, claw, joints)

            # Emit live video & joint telemetry to UI
            self.camera_update.emit(top, side, claw)
            self.telemetry_update.emit(joints, bool(joints[-1] > CONFIG.robot.grasp_threshold))

            # Handle Paused Mode
            if self.is_paused:
                self.state_update.emit("PAUSED", f"Holding state: {self.pre_pause_state.value}", action_step, 100)
                time.sleep(target_dt)
                continue

            # Check manual force next step trigger
            if self.force_next_step_flag:
                self.force_next_step_flag = False
                logger.info(f"[FSM] Manual 'Force Next Step' triggered from state {self.state.value}.")
                if self.state == FSMState.ROTATE_90:
                    self.state = FSMState.RETURN_HOME
                    state_timer_start = time.time()
                elif self.state == FSMState.SETTLE_AND_INSPECT:
                    state_timer_start = 0.0  # Force settle finish
                elif self.state == FSMState.SORT_BRICK:
                    self.state = FSMState.RETURN_HOME
                    state_timer_start = time.time()

            # ==================================================================
            # 2. DETERMINISTIC FINITE STATE MACHINE (FSM)
            # ==================================================================

            if self.state == FSMState.IDLE:
                self._send_robot_action(np.array(CONFIG.robot.home_inspect_pose, dtype=np.float32))
                self.state_update.emit("IDLE", "Awaiting operator command", 0, 100)
                self.active_model_update.emit("None [Idle]", "Standby")

            elif self.state == FSMState.ROTATE_90:
                action_step += 1
                self.mock_step = action_step
                max_steps = CONFIG.robot.max_rotate_steps
                self.state_update.emit(
                    "ROTATING",
                    f"Turn {self.face_index + 1}/4 [{(self.face_index) * 90}° -> {(self.face_index + 1) * 90}°]",
                    action_step,
                    max_steps
                )
                self.active_model_update.emit("act_rotate_90 (OpenVINO IR)", "Intel CPU")

                # Query ACT policy
                images = {"top": top, "side": side, "claw": claw}
                action = self.act_rotate.infer_step(joints, images)
                self._send_robot_action(action)

                # Gripper-Cycle Gate:
                # Phase A: Wait until MIN_GRASP_STEPS elapsed AND gripper crosses grasp threshold for N consecutive frames
                # Phase B: Wait until gripper crosses release threshold AND arm retracts (or safety timeout)
                gripper_val = float(joints[-1])
                if not grasp_detected:
                    if action_step >= CONFIG.robot.min_grasp_steps and gripper_val >= CONFIG.robot.grasp_threshold:
                        consecutive_grasps += 1
                        if consecutive_grasps >= CONFIG.robot.consecutive_grasp_req:
                            grasp_detected = True
                            logger.info(f"[GATE] Phase A Satisfied: Grasp verified on Face {self.face_index + 1} (val={gripper_val:.2f}).")
                    else:
                        consecutive_grasps = 0
                else:
                    # Phase B: Wait for release
                    if gripper_val <= CONFIG.robot.release_threshold:
                        logger.info(f"[GATE] Phase B Satisfied: Block released on Face {self.face_index + 1} (val={gripper_val:.2f}).")
                        self.state = FSMState.RETURN_HOME
                        state_timer_start = time.time()

                # Timeout fallback
                if action_step >= max_steps:
                    logger.warning(f"[TIMEOUT] Rotate 90 safety timeout reached ({max_steps} steps). Moving to inspect.")
                    self.state = FSMState.RETURN_HOME
                    state_timer_start = time.time()

            elif self.state == FSMState.RETURN_HOME:
                if return_home_start_pose is None:
                    return_home_start_pose = joints.copy()
                elapsed = time.time() - state_timer_start
                duration = 1.2
                alpha = min(1.0, max(0.0, elapsed / duration))
                # Cosine ease interpolation
                t = 0.5 * (1.0 - np.cos(np.pi * alpha))
                target_home = np.array(CONFIG.robot.home_inspect_pose, dtype=np.float32)
                interp_action = (1.0 - t) * return_home_start_pose + t * target_home
                self._send_robot_action(interp_action)
                self.state_update.emit("RETURNING HOME", f"Settling arm clear of camera ({elapsed:.1f}s)", int(alpha * 100), 100)

                if elapsed >= duration:
                    return_home_start_pose = None
                    if self.face_index < 4 and self.defect_found is False and action_step > 0:
                        self.state = FSMState.SETTLE_AND_INSPECT
                        state_timer_start = time.time()
                    elif self.face_index == 4:
                        # Sorting complete, return to idle
                        self.state = FSMState.IDLE
                    else:
                        if self.face_index < 4:
                            self.state = FSMState.SETTLE_AND_INSPECT
                            state_timer_start = time.time()
                        else:
                            self.state = FSMState.IDLE

            elif self.state == FSMState.SETTLE_AND_INSPECT:
                elapsed = time.time() - state_timer_start
                self.state_update.emit("SCANNING", f"Vibration damping & AI inspection ({elapsed:.1f}s / 0.5s)", int(elapsed * 200), 100)
                self.active_model_update.emit("patchcore_claw.xml (Anomalib)", f"Intel {CONFIG.ai.openvino_device}")

                if elapsed >= 0.5:  # 0.5s settle time elapsed
                    # Crop ROI from clean Side Camera frame
                    if side is not None:
                        rx, ry, rw, rh = CONFIG.ai.roi_x, CONFIG.ai.roi_y, CONFIG.ai.roi_w, CONFIG.ai.roi_h
                        # Clamp ROI coordinates to frame
                        h_f, w_f = side.shape[:2]
                        rx = max(0, min(rx, w_f - 10))
                        ry = max(0, min(ry, h_f - 10))
                        rw = max(10, min(rw, w_f - rx))
                        rh = max(10, min(rh, h_f - ry))

                        raw_roi = side[ry:ry + rh, rx:rx + rw].copy()

                        # Run Intel Anomalib OpenVINO Inference on Cropped ROI
                        t_ai = time.perf_counter()
                        score, mask, amap = self.anomalib_detector.infer(raw_roi)
                        ai_time_ms = (time.perf_counter() - t_ai) * 1000.0

                        is_defect = bool(score >= CONFIG.ai.anomalib_threshold)
                        if is_defect:
                            self.defect_found = True

                        # Annotate Defect on ROI
                        annotated_roi = raw_roi.copy()
                        if is_defect and mask is not None and np.any(mask > 0):
                            # Find OpenCV contours on thresholded mask
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            # Highlight semi-transparent magenta overlay
                            overlay = annotated_roi.copy()
                            overlay[mask > 0] = [255, 0, 128]  # Magenta
                            cv2.addWeighted(overlay, 0.45, annotated_roi, 0.55, 0, annotated_roi)

                            # Draw bold red/magenta contours & bounding box around black marker dot
                            cv2.drawContours(annotated_roi, contours, -1, (255, 0, 80), 2)
                            for cnt in contours:
                                if cv2.contourArea(cnt) > 8:
                                    bx, by, bw, bh = cv2.boundingRect(cnt)
                                    cv2.rectangle(annotated_roi, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                                    cv2.putText(annotated_roi, "DEFECT", (bx, max(12, by - 4)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                        logger.info(
                            f"[INSPECT] Face {self.face_index + 1}/4: Score={score:.4f} "
                            f"(Thresh={CONFIG.ai.anomalib_threshold:.4f}) -> "
                            f"{'DEFECT DETECTED' if is_defect else 'PASS'} [AI Time: {ai_time_ms:.1f}ms]"
                        )

                        # Emit to UI Filmstrip Slot (face_index 0 -> Face 1)
                        self.inspection_result.emit(self.face_index, is_defect, score, raw_roi, annotated_roi)

                    self.face_index += 1
                    if self.face_index < 4:
                        # Prepare for next 90° rotation
                        self.state = FSMState.ROTATE_90
                        self.act_rotate.reset()
                        action_step = 0
                        grasp_detected = False
                        consecutive_grasps = 0
                    else:
                        # All 4 faces inspected!
                        self.state = FSMState.SORT_BRICK
                        action_step = 0
                        grasp_detected = False
                        consecutive_grasps = 0
                        chosen_policy = self.act_place_pink if self.defect_found else self.act_place_green
                        chosen_policy.reset()

            elif self.state == FSMState.SORT_BRICK:
                action_step += 1
                self.mock_step = action_step
                max_steps = CONFIG.robot.max_sort_steps
                dest_name = "PINK (DEFECT)" if self.defect_found else "GREEN (PASS)"
                self.state_update.emit(
                    f"SORTING: {dest_name}",
                    f"Delivering brick to {dest_name} plate",
                    action_step,
                    max_steps
                )
                model_name = "act_place_pink" if self.defect_found else "act_place_green"
                self.active_model_update.emit(f"{model_name} (OpenVINO IR)", "Intel CPU")

                # Query Placement Policy
                policy = self.act_place_pink if self.defect_found else self.act_place_green
                images = {"top": top, "side": side, "claw": claw}
                action = policy.infer_step(joints, images)
                self._send_robot_action(action)

                # Gripper-Cycle Gate for Placement
                gripper_val = float(joints[-1])
                if not grasp_detected:
                    if action_step >= CONFIG.robot.min_grasp_steps and gripper_val >= CONFIG.robot.grasp_threshold:
                        consecutive_grasps += 1
                        if consecutive_grasps >= CONFIG.robot.consecutive_grasp_req:
                            grasp_detected = True
                            logger.info(f"[GATE] Sort grasp verified on center stage (val={gripper_val:.2f}).")
                    else:
                        consecutive_grasps = 0
                else:
                    if gripper_val <= CONFIG.robot.release_threshold:
                        logger.info(f"[GATE] Sort release verified over {dest_name} plate.")
                        self._finish_cycle()
                        self.state = FSMState.RETURN_HOME
                        state_timer_start = time.time()

                if action_step >= max_steps:
                    logger.warning(f"[TIMEOUT] Sort placement timeout reached ({max_steps} steps). Finishing cycle.")
                    self._finish_cycle()
                    self.state = FSMState.RETURN_HOME
                    state_timer_start = time.time()

            elif self.state == FSMState.ESTOP:
                self.state_update.emit("E-STOP", "EMERGENCY STOP ENGAGED — MOTORS HALTED", 0, 100)
                self.active_model_update.emit("Disabled", "Halted")

            # 3. Frequency Control Loop (30 Hz)
            elapsed_loop = time.perf_counter() - loop_start
            sleep_sec = target_dt - elapsed_loop
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    def _finish_cycle(self):
        """Updates session counters after completion of sorting."""
        self.total_inspected += 1
        if self.defect_found:
            self.pink_count += 1
        else:
            self.green_count += 1

        rate = (self.green_count / self.total_inspected) * 100.0 if self.total_inspected > 0 else 0.0
        self.session_stats_update.emit(self.total_inspected, self.green_count, self.pink_count, rate)
        self.face_index = 4  # marks cycle finished

    # ==========================================================================
    # OPERATOR CONTROLS (THREAD-SAFE SLOTS)
    # ==========================================================================

    def trigger_start(self):
        if self.state in [FSMState.IDLE, FSMState.RETURN_HOME]:
            logger.info("[OPERATOR] Starting autonomous multi-view inspection cycle...")
            self.defect_found = False
            self.face_index = 0
            self.mock_step = 0
            self.act_rotate.reset()
            self.state = FSMState.ROTATE_90
            self.status_message.emit("Cycle started: Beginning 4-Face rotation and inspection.")

    def trigger_pause_resume(self):
        if self.is_paused:
            self.is_paused = False
            self.state = self.pre_pause_state
            logger.info("[OPERATOR] Cycle RESUMED.")
            self.status_message.emit("Cycle resumed.")
        else:
            self.is_paused = True
            self.pre_pause_state = self.state
            logger.info("[OPERATOR] Cycle PAUSED.")
            self.status_message.emit("Cycle paused by operator.")

    def trigger_estop(self):
        logger.warning("[OPERATOR] EMERGENCY STOP TRIGGERED!")
        self.state = FSMState.ESTOP
        self.is_paused = False
        if not self.mock_mode and self.robot:
            try:
                self.robot.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting robot on E-STOP: {e}")
        self.status_message.emit("EMERGENCY STOP ACTIVATED! All motor torques disabled.")

    def trigger_home(self):
        logger.info("[OPERATOR] Commanding arm to HOME pose...")
        self.state = FSMState.IDLE
        self.is_paused = False
        self.face_index = 0
        self._send_robot_action(np.array(CONFIG.robot.home_inspect_pose, dtype=np.float32))
        self.status_message.emit("Arm moved to home inspection pose.")

    def trigger_force_next_step(self):
        self.force_next_step_flag = True

    def test_anomalib_current_frame(self):
        """Immediately runs Anomalib on the current side camera ROI and emits result."""
        _, side, _, _, _ = self.obs_buffer.get_latest()
        if side is not None:
            rx, ry, rw, rh = CONFIG.ai.roi_x, CONFIG.ai.roi_y, CONFIG.ai.roi_w, CONFIG.ai.roi_h
            roi = side[ry:ry + rh, rx:rx + rw].copy()
            score, mask, _ = self.anomalib_detector.infer(roi)
            is_def = bool(score >= CONFIG.ai.anomalib_threshold)
            logger.info(f"[TEST ANOMALIB] Score: {score:.4f} | Defect: {is_def}")
            self.status_message.emit(f"Manual Test Score: {score:.4f} ({'DEFECT' if is_def else 'PASS'})")
            # Populate slot 0 as test preview
            self.inspection_result.emit(0, is_def, score, roi, roi)

    def capture_current_pose_as_home(self):
        """Captures follower's current joint positions and saves to config."""
        _, _, _, joints, _ = self.obs_buffer.get_latest()
        CONFIG.robot.home_inspect_pose = [round(float(j), 2) for j in joints]
        CONFIG.save()
        logger.info(f"[CONFIG] Saved current arm pose as HOME: {CONFIG.robot.home_inspect_pose}")
        self.status_message.emit(f"Saved new HOME pose: {CONFIG.robot.home_inspect_pose}")

    def set_anomalib_threshold(self, val: float):
        CONFIG.ai.anomalib_threshold = float(val)
        CONFIG.save()

    def set_roi(self, x: int, y: int, w: int, h: int):
        CONFIG.ai.roi_x = x
        CONFIG.ai.roi_y = y
        CONFIG.ai.roi_w = w
        CONFIG.ai.roi_h = h
        CONFIG.save()

    def set_grasp_threshold(self, val: float):
        CONFIG.robot.grasp_threshold = float(val)
        CONFIG.save()

    def set_release_threshold(self, val: float):
        CONFIG.robot.release_threshold = float(val)
        CONFIG.save()

    def stop(self):
        self._running = False
        self.wait(1500)
        if self.cap_side: self.cap_side.release()
        if self.cap_top: self.cap_top.release()
        if self.cap_claw: self.cap_claw.release()
        if self.robot and getattr(self.robot, "is_connected", False):
            try: self.robot.disconnect()
            except: pass
