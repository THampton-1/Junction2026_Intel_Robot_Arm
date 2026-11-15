import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import openvino as ov

logger = logging.getLogger("ov_models")


class ACTOpenVINOModel:
    """
    OpenVINO IR ACT Policy wrapper with chunk queue execution.
    Loads and compiles the exported act.xml model into memory.
    """

    def __init__(self, xml_path: str, model_name: str, device: str = "CPU"):
        self.xml_path = Path(xml_path)
        self.model_name = model_name
        self.device = device
        self.core = ov.Core()
        self.compiled_model = None
        self.action_queue: deque = deque()
        self.target_image_size = (384, 384)  # (W, H)
        self.compile()

    def compile(self):
        if not self.xml_path.exists():
            logger.warning(f"[ACT {self.model_name}] XML not found at {self.xml_path}. Running in mock mode.")
            self.compiled_model = None
            return

        try:
            available = self.core.available_devices
            dev = self.device if self.device in available else "CPU"
            logger.info(f"[ACT {self.model_name}] Loading from {self.xml_path}...")
            model = self.core.read_model(str(self.xml_path))
            self.compiled_model = self.core.compile_model(model, dev)
            self.actual_device = dev
            logger.info(f"[ACT {self.model_name}] Compiled successfully on {dev}!")

            # Warmup
            dummy_state = np.zeros((1, 6), dtype=np.float32)
            dummy_img = np.zeros((1, 3, 384, 384), dtype=np.float32)
            _ = self.compiled_model({
                "state": dummy_state,
                "images.claw_cam": dummy_img,
                "images.top_view": dummy_img,
                "images.side_view": dummy_img,
            })
            logger.info(f"[ACT {self.model_name}] Warmed up and ready for 0ms switching.")
        except Exception as e:
            logger.exception(f"[ACT {self.model_name}] Compilation error: {e}")
            self.compiled_model = None

    def reset(self):
        """Clears the predicted action chunk buffer on each state transition."""
        self.action_queue.clear()
        logger.debug(f"[ACT {self.model_name}] Action queue reset.")

    def _preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesses frame to [1, 3, 384, 384] float32 in [0, 1]."""
        if frame is None:
            return np.zeros((1, 3, 384, 384), dtype=np.float32)

        # Resize to 384x384
        resized = cv2.resize(frame, self.target_image_size, interpolation=cv2.INTER_LINEAR)
        # Convert HWC uint8 -> CHW float32 [0, 1]
        chw = resized.transpose((2, 0, 1)).astype(np.float32)
        if chw.max() > 1.0:
            chw = chw / 255.0
        return np.expand_dims(chw, axis=0)

    def infer_step(self, current_state: np.ndarray, images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Returns the next 6-DOF joint target action from the chunk queue.
        If the queue is empty, runs forward inference to predict a new 100-step action chunk.
        """
        if len(self.action_queue) == 0:
            if self.compiled_model is not None:
                try:
                    state_tensor = np.expand_dims(current_state.astype(np.float32), axis=0)
                    claw_tensor = self._preprocess_image(images.get("claw"))
                    top_tensor = self._preprocess_image(images.get("top"))
                    side_tensor = self._preprocess_image(images.get("side"))

                    inputs = {
                        "state": state_tensor,
                        "images.claw_cam": claw_tensor,
                        "images.top_view": top_tensor,
                        "images.side_view": side_tensor,
                    }
                    outputs = self.compiled_model(inputs)
                    # Output shape is [1, 100, 6]
                    action_chunk = list(outputs.values())[0][0]  # shape: [100, 6]
                    for action in action_chunk:
                        self.action_queue.append(action)
                except Exception as e:
                    logger.error(f"[ACT {self.model_name}] Inference error: {e}")
                    # Fallback to current state
                    return current_state
            else:
                # Mock action generation (slight progression toward target)
                return current_state

        if len(self.action_queue) > 0:
            return self.action_queue.popleft()
        return current_state


class AnomalibOpenVINOModel:
    """
    OpenVINO PatchCore defect detector offloaded to Intel Xe3 iGPU / NPU.
    Directly loads exported patchcore_claw.xml and computes continuous anomaly scores and masks.
    """

    def __init__(self, xml_path: str, input_size: Tuple[int, int] = (240, 320), device: str = "GPU"):
        self.xml_path = Path(xml_path)
        self.input_size = input_size
        self.device = device
        self.core = ov.Core()
        self.compiled_model = None
        self.compile()

    def compile(self):
        if not self.xml_path.exists():
            logger.warning(f"[ANOMALIB] XML not found at {self.xml_path}. Mocking defect detector.")
            return

        try:
            available = self.core.available_devices
            logger.info(f"[ANOMALIB] OpenVINO Available Devices: {available}")
            dev = self.device if self.device in available else "CPU"

            logger.info(f"[ANOMALIB] Reading model from {self.xml_path}...")
            model = self.core.read_model(str(self.xml_path))

            # Intel Xe3 iGPU / NPU static shape enforcement
            h, w = self.input_size
            try:
                model.reshape([1, 3, h, w])
            except Exception as e:
                logger.debug(f"[ANOMALIB] Reshape note: {e}")

            logger.info(f"[ANOMALIB] Compiling model for {dev}...")
            self.compiled_model = self.core.compile_model(model, dev)
            self.actual_device = dev
            logger.info(f"[ANOMALIB] Compiled successfully on {dev}!")

            # Warmup
            dummy = np.random.randn(1, 3, h, w).astype(np.float32)
            for _ in range(3):
                _ = self.compiled_model([dummy])
            logger.info("[ANOMALIB] Model warmed up and ready.")
        except Exception as e:
            logger.exception(f"[ANOMALIB] Compilation error: {e}")
            self.compiled_model = None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resizes to (320, 240) and converts to (1, 3, H, W) float32 [0, 1]."""
        h, w = self.input_size
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        chw = frame.transpose((2, 0, 1)).astype(np.float32)
        if chw.max() > 1.0:
            chw = chw / 255.0
        return np.expand_dims(chw, axis=0)

    def infer(self, frame: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Runs defect detection on the given frame (e.g. cropped ROI).
        Returns:
            pred_score: continuous anomaly score (float)
            pred_mask: binary defect mask resized to frame's original (H, W) (uint8 0 or 255)
            anomaly_map: normalized float anomaly heatmap [0, 1]
        """
        orig_h, orig_w = frame.shape[:2]

        if self.compiled_model is None:
            # Mock detector behavior: check for artificial dark spot (black marker dot)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            min_val, _, min_loc, _ = cv2.minMaxLoc(gray)
            if min_val < 45:  # Black dot detected
                mock_score = 0.88
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                cv2.circle(mask, min_loc, 14, 255, -1)
                return mock_score, mask, mask.astype(np.float32) / 255.0
            else:
                mock_score = float(np.random.uniform(0.10, 0.28))
                return mock_score, np.zeros((orig_h, orig_w), dtype=np.uint8), np.zeros((orig_h, orig_w), dtype=np.float32)

        inp = self.preprocess(frame)
        outputs = self.compiled_model([inp])

        score = 0.0
        mask_raw = None
        amap_raw = None

        for key, val in outputs.items():
            name = key.get_any_name()
            if name == "pred_score":
                score = float(val.flat[0])
            elif name == "pred_mask":
                mask_raw = val[0, 0]
            elif name == "anomaly_map":
                amap_raw = val[0, 0]

        # Process binary mask to match input ROI dimensions
        if mask_raw is not None:
            mask_uint8 = (mask_raw.astype(np.uint8) * 255)
            mask_resized = cv2.resize(mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Process anomaly heatmap
        if amap_raw is not None:
            amap_norm = (amap_raw - amap_raw.min()) / (amap_raw.max() - amap_raw.min() + 1e-6)
            amap_resized = cv2.resize(amap_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            amap_resized = np.zeros((orig_h, orig_w), dtype=np.float32)

        return score, mask_resized, amap_resized
