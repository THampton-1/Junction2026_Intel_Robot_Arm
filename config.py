import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List


@dataclass
class RobotConfig:
    port: str = "/dev/ttyACM0"
    calibration_path: str = "/home/intel/so101_calibration_backup.json"
    control_hz: int = 30
    grasp_threshold: float = 3.5    # Gripper angle threshold to detect grasp (closed on brick)
    release_threshold: float = 2.5  # Gripper angle threshold to detect release (opened)
    min_grasp_steps: int = 15       # Minimum steps before checking grasp
    consecutive_grasp_req: int = 3  # Required consecutive frames exceeding grasp threshold
    max_rotate_steps: int = 300     # Safety timeout for 90-degree turn (~10s)
    max_sort_steps: int = 400       # Safety timeout for sorting placement (~13s)
    home_inspect_pose: List[float] = field(default_factory=lambda: [
        -5.0, -100.0, 98.0, 60.0, -50.0, 2.0
    ])


@dataclass
class CameraConfig:
    primary_side_index: str = "/dev/video32"
    top_index: str = "/dev/video38"
    claw_index: str = "/dev/video34"
    width: int = 320
    height: int = 240


@dataclass
class AIConfig:
    # ACT Models (OpenVINO IR XML paths discovered from physicalai.db)
    act_rotate_90_xml: str = "/home/intel/.local/share/physicalai/models/fdbac383-0ca4-4392-bd94-979a59914248/exports/openvino/act.xml"
    act_place_green_xml: str = "/home/intel/.local/share/physicalai/models/b6c62b51-70bc-4bf4-906f-62f3b43ed3e9/exports/openvino/act.xml"
    act_place_pink_xml: str = "/home/intel/.local/share/physicalai/models/e41fb24f-85c6-47fc-9f26-6dfa4667943c/exports/openvino/act.xml"

    # Anomalib PatchCore Model
    anomalib_xml: str = "/home/intel/models/anomalib_patchcore_claw/weights/openvino/patchcore_claw.xml"
    anomalib_input_size: Tuple[int, int] = (240, 320)  # (H, W)
    anomalib_threshold: float = 0.4781
    openvino_device: str = "GPU"  # Xe3 iGPU acceleration

    # Inspection ROI on Side Camera centered on Lego brick
    roi_x: int = 60
    roi_y: int = 60
    roi_w: int = 180
    roi_h: int = 140


@dataclass
class AppConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    ai: AIConfig = field(default_factory=AIConfig)

    def save(self, filepath: str = "app_config.json"):
        data = {
            "robot": self.robot.__dict__,
            "camera": self.camera.__dict__,
            "ai": {
                **self.ai.__dict__,
                "anomalib_input_size": list(self.ai.anomalib_input_size),
            },
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str = "app_config.json") -> "AppConfig":
        cfg = cls()
        path = Path(filepath)
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if "robot" in data:
                    cfg.robot = RobotConfig(**data["robot"])
                if "camera" in data:
                    cfg.camera = CameraConfig(**data["camera"])
                if "ai" in data:
                    ai_data = data["ai"]
                    if "anomalib_input_size" in ai_data:
                        ai_data["anomalib_input_size"] = tuple(ai_data["anomalib_input_size"])
                    cfg.ai = AIConfig(**ai_data)
            except Exception as e:
                print(f"Failed to load config from {filepath}: {e}, using defaults.")
        return cfg


CONFIG = AppConfig.load()
