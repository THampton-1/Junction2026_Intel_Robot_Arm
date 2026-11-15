import sys
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from config import CONFIG
from robot_fsm import FSMState, RobotControlWorker

# ==============================================================================
# RETRO EARLY-2000s INTEL WEBSITE STYLESHEET (Centrino / Pentium 4 Era: 2001-2005)
# ==============================================================================

INTEL_RETRO_QSS = """
/* Global Window & Typography */
QMainWindow {
    background-color: #DCE4EC;
}
QWidget {
    font-family: "Tahoma", "Verdana", "Segoe UI", sans-serif;
    font-size: 11px;
    color: #222222;
}

/* Intel Classic Blue Top Header */
#topHeaderBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0071C5, stop:1 #004B87);
    border-bottom: 2px solid #003366;
    padding: 6px 12px;
}
#logoLabel {
    color: #FFFFFF;
    font-size: 18px;
    font-weight: bold;
    font-family: "Verdana", sans-serif;
}
#subLogoLabel {
    color: #BBE1FA;
    font-size: 11px;
    font-weight: normal;
}
#hardwareBadge {
    background-color: #002D54;
    color: #00FFCC;
    font-family: "Consolas", monospace;
    font-size: 11px;
    padding: 3px 8px;
    border: 1px solid #005A9C;
}

/* Panel Containers */
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #7F9DB9;
    margin-top: 18px;
    font-weight: bold;
    font-size: 11px;
    color: #004B87;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0071C5, stop:1 #004F8B);
    color: #FFFFFF;
    border: 1px solid #003366;
}

/* Classic 2000s Inset / Outset Beveled Buttons */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F6F8FA, stop:1 #D8E0E8);
    border: 1px solid #7F9DB9;
    border-right-color: #4A6572;
    border-bottom-color: #4A6572;
    padding: 5px 12px;
    font-weight: bold;
    color: #111111;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:1 #E4ECF4);
}
QPushButton:pressed {
    background: #CAD5E0;
    border: 1px solid #334455;
    border-right-color: #FFFFFF;
    border-bottom-color: #FFFFFF;
    padding-top: 6px;
    padding-left: 13px;
}
QPushButton:disabled {
    background: #EAEAEA;
    color: #888888;
    border: 1px solid #CCCCCC;
}

/* High-Profile Action Buttons */
#btnStart {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0082D9, stop:1 #005A9C);
    color: #FFFFFF;
    font-size: 12px;
    font-weight: bold;
    border: 1px solid #003366;
    border-right-color: #001A33;
    border-bottom-color: #001A33;
    min-height: 28px;
}
#btnStart:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1A96ED, stop:1 #006BB3);
}
#btnStart:disabled {
    background: #99C2E0;
    color: #E0E0E0;
}
#btnEstop {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #E52B50, stop:1 #B2182B);
    color: #FFFFFF;
    font-size: 12px;
    font-weight: bold;
    border: 1px solid #700;
    min-height: 28px;
}
#btnEstop:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FF3B60, stop:1 #D12030);
}

/* Status Badges */
#stateBadge {
    font-size: 13px;
    font-weight: bold;
    padding: 6px 10px;
    border: 1px solid #555555;
    color: #FFFFFF;
    background-color: #0071C5;
    text-align: center;
}

/* Monospace Telemetry Readout Box */
#telemetryBox {
    background-color: #F8FBFD;
    border: 1px inset #7F9DB9;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
    color: #003366;
    padding: 5px;
}

/* Filmstrip Frames */
.FilmstripCard {
    background-color: #F5F7FA;
    border: 1px solid #7F9DB9;
}
"""


class LiveCameraWidget(QWidget):
    """Viewport displaying a camera stream with optional live ROI targeting rectangle."""

    def __init__(self, camera_name: str, has_roi: bool = False, default_w: int = 320, default_h: int = 240):
        super().__init__()
        self.camera_name = camera_name
        self.has_roi = has_roi
        self.default_w = default_w
        self.default_h = default_h
        self._pixmap: Optional[QPixmap] = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(160, 120)

    def sizeHint(self) -> QSize:
        return QSize(self.default_w, self.default_h)

    def minimumSizeHint(self) -> QSize:
        return QSize(160, 120)

    def update_frame(self, frame_rgb: Optional[np.ndarray], roi_rect: Optional[Tuple[int, int, int, int]] = None):
        if frame_rgb is None:
            return

        render_img = frame_rgb.copy()
        # Draw live cyan targeting rectangle for ROI on side camera
        if self.has_roi and roi_rect is not None:
            rx, ry, rw, rh = roi_rect
            cv2.rectangle(render_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            cv2.putText(render_img, "ANOMALIB ROI", (rx, max(12, ry - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        h, w, ch = render_img.shape
        bytes_per_line = ch * w
        q_img = QImage(render_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_img)
        # Calling update() queues a repaint without modifying the widget's layout sizeHint
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        if self._pixmap is not None and not self._pixmap.isNull():
            # Render scaled pixmap centered within current widget geometry
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QPen(QColor(0, 255, 204)))
            painter.setFont(QFont("Consolas", 10))
            painter.drawText(self.rect(), Qt.AlignCenter, f"[ {self.camera_name} FEED: INITIALIZING ]")

        # Inset border
        painter.setPen(QPen(QColor(0x7F, 0x9D, 0xB9), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)


class FaceSnapshotWidget(QWidget):
    """Panel in the 4-Face Inspection Filmstrip showing inspected face, score, and Pass/Defect banner."""

    def __init__(self, face_label: str):
        super().__init__()
        self.face_label = face_label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # Header Title
        self.title_lbl = QLabel(face_label)
        self.title_lbl.setStyleSheet("font-weight: bold; color: #004B87; font-size: 10px;")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_lbl)

        # Thumbnail Image
        self.img_lbl = QLabel("WAITING")
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setFixedSize(110, 85)
        self.img_lbl.setStyleSheet("background-color: #E6EAEF; border: 1px inset #7F9DB9; color: #666666;")
        layout.addWidget(self.img_lbl, stretch=1)

        # Result Banner (PASS / DEFECT)
        self.banner_lbl = QLabel("STANDBY")
        self.banner_lbl.setAlignment(Qt.AlignCenter)
        self.banner_lbl.setStyleSheet("background-color: #DCE4EC; border: 1px solid #7F9DB9; font-weight: bold; font-size: 9px; padding: 2px;")
        layout.addWidget(self.banner_lbl)

        # Anomaly Score
        self.score_lbl = QLabel("Score: ---")
        self.score_lbl.setAlignment(Qt.AlignCenter)
        self.score_lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 10px; color: #333333;")
        layout.addWidget(self.score_lbl)

        self.setStyleSheet("background-color: #FFFFFF; border: 1px solid #7F9DB9;")

    def reset(self):
        self.img_lbl.setText("WAITING")
        self.img_lbl.setPixmap(QPixmap())
        self.img_lbl.setStyleSheet("background-color: #E6EAEF; border: 1px inset #7F9DB9; color: #666666;")
        self.banner_lbl.setText("STANDBY")
        self.banner_lbl.setStyleSheet("background-color: #DCE4EC; border: 1px solid #7F9DB9; font-weight: bold; font-size: 9px; padding: 2px;")
        self.score_lbl.setText("Score: ---")

    def set_result(self, is_defect: bool, score: float, image_rgb: Optional[np.ndarray]):
        if image_rgb is not None:
            h, w, ch = image_rgb.shape
            q_img = QImage(image_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(q_img).scaled(100, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_lbl.setPixmap(pix)

        self.score_lbl.setText(f"Score: {score:.4f}")

        if is_defect:
            self.banner_lbl.setText("DEFECT DETECTED")
            self.banner_lbl.setStyleSheet(
                "background-color: #FF0055; color: #FFFFFF; font-weight: bold; font-size: 9px; border: 1px solid #990033; padding: 2px;"
            )
            self.setStyleSheet("background-color: #FFF0F4; border: 2px solid #FF0055;")
        else:
            self.banner_lbl.setText("PASS (NORMAL)")
            self.banner_lbl.setStyleSheet(
                "background-color: #00AA44; color: #FFFFFF; font-weight: bold; font-size: 9px; border: 1px solid #006622; padding: 2px;"
            )
            self.setStyleSheet("background-color: #F0FFF4; border: 1px solid #00AA44;")


class Intel2000sMainWindow(QMainWindow):
    """Main Application Window styled like classic 2000s Intel portals."""

    def __init__(self, worker: RobotControlWorker):
        super().__init__()
        self.worker = worker
        self.setWindowTitle("intel // Physical AI Quality Assurance Station — SO-101 Robotics")
        self.resize(1280, 840)
        self.setMinimumSize(1024, 700)
        self.setStyleSheet(INTEL_RETRO_QSS)

        self._setup_ui()
        self._connect_signals()

        # UI Refresh Loop at 30 FPS to read observation buffer
        self.display_timer = QTimer(self)
        self.display_timer.timeout.connect(self._poll_live_observation)
        self.display_timer.start(33)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ----------------------------------------------------------------------
        # 1. TOP INTEL BANNER BAR (Centrino / Pentium 4 Era Aesthetic)
        # ----------------------------------------------------------------------
        header_bar = QWidget()
        header_bar.setObjectName("topHeaderBar")
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(12, 6, 12, 6)

        title_box = QVBoxLayout()
        logo_lbl = QLabel("intel // Physical AI Quality Assurance Station")
        logo_lbl.setObjectName("logoLabel")
        sub_lbl = QLabel("Autonomous Multi-View Active Inspection & Routing • OpenVINO™ • SO-101 Arm")
        sub_lbl.setObjectName("subLogoLabel")
        title_box.addWidget(logo_lbl)
        title_box.addWidget(sub_lbl)
        header_layout.addLayout(title_box, stretch=1)

        # Hardware & OpenVINO Status Badges
        self.hw_badge = QLabel("SO-101: CONNECTED [/dev/ttyACM0]" if not self.worker.mock_mode else "SO-101: SIMULATED [MOCK]")
        self.hw_badge.setObjectName("hardwareBadge")
        header_layout.addWidget(self.hw_badge)

        self.dev_badge = QLabel(f"OPENVINO: {CONFIG.ai.openvino_device} ACCELERATION")
        self.dev_badge.setObjectName("hardwareBadge")
        header_layout.addWidget(self.dev_badge)

        self.model_badge = QLabel("ACTIVE POLICY: None [Idle]")
        self.model_badge.setObjectName("hardwareBadge")
        header_layout.addWidget(self.model_badge)

        main_layout.addWidget(header_bar)

        # ----------------------------------------------------------------------
        # 2. BODY CONTENT (Cameras + Controls + Filmstrip + Calibration)
        # ----------------------------------------------------------------------
        body_widget = QWidget()
        body_layout = QHBoxLayout(body_widget)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(8)

        # === LEFT COLUMN: 3 LIVE CAMERAS ===
        left_column = QVBoxLayout()
        body_layout.addLayout(left_column, stretch=3)

        # Primary Side Camera (Large)
        cam_side_box = QGroupBox("Primary Inspection Camera [Side View - Brick Face & ROI]")
        cam_side_layout = QVBoxLayout(cam_side_box)
        cam_side_layout.setContentsMargins(4, 4, 4, 4)
        self.cam_side_view = LiveCameraWidget("PRIMARY SIDE", has_roi=True)
        cam_side_layout.addWidget(self.cam_side_view, stretch=2)
        left_column.addWidget(cam_side_box, stretch=2)

        # Secondary Cameras (Top RealSense & Claw Viewports)
        sec_cams_box = QGroupBox("Secondary Vision Viewports [Top RealSense & Claw Camera]")
        sec_cams_layout = QHBoxLayout(sec_cams_box)
        sec_cams_layout.setContentsMargins(4, 4, 4, 4)
        self.cam_top_view = LiveCameraWidget("TOP REALSENSE", has_roi=False)
        self.cam_claw_view = LiveCameraWidget("CLAW CAMERA", has_roi=False)
        sec_cams_layout.addWidget(self.cam_top_view)
        sec_cams_layout.addWidget(self.cam_claw_view)
        left_column.addWidget(sec_cams_box, stretch=1)

        # === RIGHT COLUMN: FILMSTRIP, CONTROLS, TELEMETRY, CALIBRATION ===
        right_column = QVBoxLayout()
        body_layout.addLayout(right_column, stretch=2)

        # --- A. PERSISTENT 4-FACE INSPECTION FILMSTRIP ---
        filmstrip_box = QGroupBox("Persistent 4-Face Inspection Filmstrip")
        filmstrip_layout = QHBoxLayout(filmstrip_box)
        filmstrip_layout.setContentsMargins(6, 6, 6, 6)
        filmstrip_layout.setSpacing(6)

        self.film_slots = []
        labels = ["FACE 1 [0°]", "FACE 2 [90°]", "FACE 3 [180°]", "FACE 4 [270°]"]
        for lbl_text in labels:
            slot = FaceSnapshotWidget(lbl_text)
            filmstrip_layout.addWidget(slot)
            self.film_slots.append(slot)
        right_column.addWidget(filmstrip_box)

        # --- B. FSM STATUS & PROGRESS BAR ---
        status_box = QGroupBox("Automation State & Cycle Progress")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(6, 6, 6, 6)

        self.state_badge = QLabel("IDLE")
        self.state_badge.setObjectName("stateBadge")
        self.state_badge.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.state_badge)

        self.state_detail_lbl = QLabel("Awaiting operator command")
        self.state_detail_lbl.setAlignment(Qt.AlignCenter)
        self.state_detail_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #004B87;")
        status_layout.addWidget(self.state_detail_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px inset #7F9DB9; text-align: center; height: 16px; }"
                                        "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0071C5, stop:1 #004F8B); }")
        status_layout.addWidget(self.progress_bar)
        right_column.addWidget(status_box)

        # --- C. OPERATOR CONTROL DECK ---
        deck_box = QGroupBox("Operator Tactical Deck")
        deck_layout = QGridLayout(deck_box)
        deck_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_start = QPushButton("[ ▶ START INSPECTION ]")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.worker.trigger_start)
        deck_layout.addWidget(self.btn_start, 0, 0, 1, 2)

        self.btn_pause = QPushButton("[ ⏸ PAUSE / RESUME ]")
        self.btn_pause.clicked.connect(self.worker.trigger_pause_resume)
        deck_layout.addWidget(self.btn_pause, 1, 0)

        self.btn_home = QPushButton("[ ↺ MOVE TO HOME POSE ]")
        self.btn_home.clicked.connect(self.worker.trigger_home)
        deck_layout.addWidget(self.btn_home, 1, 1)

        self.btn_force_step = QPushButton("[ ⏭ FORCE NEXT STEP ]")
        self.btn_force_step.clicked.connect(self.worker.trigger_force_next_step)
        deck_layout.addWidget(self.btn_force_step, 2, 0)

        self.btn_estop = QPushButton("[ 🛑 EMERGENCY STOP ]")
        self.btn_estop.setObjectName("btnEstop")
        self.btn_estop.clicked.connect(self.worker.trigger_estop)
        deck_layout.addWidget(self.btn_estop, 2, 1)

        right_column.addWidget(deck_box)

        # --- D. SESSION TELEMETRY & LIVE CALIBRATION DRAWER ---
        telemetry_box = QGroupBox("Session Telemetry & Live Hardware Calibration")
        tel_layout = QVBoxLayout(telemetry_box)
        tel_layout.setContentsMargins(6, 6, 6, 6)

        # Stats readout
        self.lbl_stats = QLabel("Total: 0 | Green Plate (Pass): 0 | Pink Plate (Defect): 0 | Pass Rate: 0.0%")
        self.lbl_stats.setObjectName("telemetryBox")
        tel_layout.addWidget(self.lbl_stats)

        # Arm Joint & Gripper telemetry
        self.lbl_joints = QLabel("SO-101 Joints [Pan, Lift, Elbow, W_Flex, W_Roll, Gripper]:\n---")
        self.lbl_joints.setObjectName("telemetryBox")
        tel_layout.addWidget(self.lbl_joints)

        # Calibration controls
        calib_grid = QGridLayout()

        # Anomalib Threshold
        calib_grid.addWidget(QLabel("Anomalib Threshold:"), 0, 0)
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.01, 1.00)
        self.thresh_spin.setSingleStep(0.02)
        self.thresh_spin.setDecimals(4)
        self.thresh_spin.setValue(CONFIG.ai.anomalib_threshold)
        self.thresh_spin.valueChanged.connect(self._on_threshold_changed)
        calib_grid.addWidget(self.thresh_spin, 0, 1)

        self.btn_test_anomalib = QPushButton("Test On Current Frame")
        self.btn_test_anomalib.clicked.connect(self.worker.test_anomalib_current_frame)
        calib_grid.addWidget(self.btn_test_anomalib, 0, 2)

        # ROI adjustment (X, Y, W, H)
        calib_grid.addWidget(QLabel("Side ROI (X, Y, W, H):"), 1, 0)
        roi_sub = QHBoxLayout()
        self.sp_roi_x = QSpinBox(); self.sp_roi_x.setRange(0, 320); self.sp_roi_x.setValue(CONFIG.ai.roi_x)
        self.sp_roi_y = QSpinBox(); self.sp_roi_y.setRange(0, 240); self.sp_roi_y.setValue(CONFIG.ai.roi_y)
        self.sp_roi_w = QSpinBox(); self.sp_roi_w.setRange(10, 320); self.sp_roi_w.setValue(CONFIG.ai.roi_w)
        self.sp_roi_h = QSpinBox(); self.sp_roi_h.setRange(10, 240); self.sp_roi_h.setValue(CONFIG.ai.roi_h)
        for sp in [self.sp_roi_x, self.sp_roi_y, self.sp_roi_w, self.sp_roi_h]:
            sp.valueChanged.connect(self._on_roi_changed)
            roi_sub.addWidget(sp)
        calib_grid.addLayout(roi_sub, 1, 1, 1, 2)

        # Gripper Grasp / Release Thresholds
        calib_grid.addWidget(QLabel("Grasp / Release Thresh:"), 2, 0)
        thresh_sub = QHBoxLayout()
        self.sp_grasp = QDoubleSpinBox(); self.sp_grasp.setRange(0.0, 50.0); self.sp_grasp.setValue(CONFIG.robot.grasp_threshold)
        self.sp_grasp.valueChanged.connect(self.worker.set_grasp_threshold)
        self.sp_release = QDoubleSpinBox(); self.sp_release.setRange(0.0, 50.0); self.sp_release.setValue(CONFIG.robot.release_threshold)
        self.sp_release.valueChanged.connect(self.worker.set_release_threshold)
        thresh_sub.addWidget(self.sp_grasp)
        thresh_sub.addWidget(self.sp_release)
        calib_grid.addLayout(thresh_sub, 2, 1, 1, 2)

        # Capture Pose Button
        self.btn_cap_home = QPushButton("[ Capture Current Arm Pose as Home/Inspect Pose ]")
        self.btn_cap_home.clicked.connect(self.worker.capture_current_pose_as_home)
        calib_grid.addWidget(self.btn_cap_home, 3, 0, 1, 3)

        tel_layout.addLayout(calib_grid)
        right_column.addWidget(telemetry_box)

        main_layout.addWidget(body_widget, stretch=1)

        # ----------------------------------------------------------------------
        # 3. STATUS BAR (Classic Windows 2000 / XP Grey Inset Status Bar)
        # ----------------------------------------------------------------------
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #DCE4EC; border-top: 1px solid #7F9DB9; font-size: 10px; color: #333333;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("System ready. Running on Intel® Core™ Ultra hardware.")

    def _connect_signals(self):
        self.worker.telemetry_update.connect(self._on_telemetry_update)
        self.worker.state_update.connect(self._on_state_update)
        self.worker.active_model_update.connect(self._on_active_model_update)
        self.worker.inspection_result.connect(self._on_inspection_result)
        self.worker.session_stats_update.connect(self._on_session_stats_update)
        self.worker.status_message.connect(self._on_status_message)

    # ==========================================================================
    # SLOTS
    # ==========================================================================

    @Slot()
    def _poll_live_observation(self):
        """Polls thread-safe LatestObservationBuffer at 30 FPS to update live viewports."""
        top, side, claw, _, _ = self.worker.obs_buffer.get_latest()
        roi = (CONFIG.ai.roi_x, CONFIG.ai.roi_y, CONFIG.ai.roi_w, CONFIG.ai.roi_h)
        self.cam_side_view.update_frame(side, roi_rect=roi)
        self.cam_top_view.update_frame(top)
        self.cam_claw_view.update_frame(claw)

    @Slot(object, bool)
    def _on_telemetry_update(self, joints: np.ndarray, is_gripper_closed: bool):
        j_text = ", ".join([f"{val:6.1f}°" for val in joints])
        grip_state = "CLOSED (GRASPED)" if is_gripper_closed else "OPEN (RELEASED)"
        self.lbl_joints.setText(f"SO-101 Joints: [{j_text}]\nGripper State: {grip_state} (Val: {joints[-1]:.2f})")

    @Slot(str, str, int, int)
    def _on_state_update(self, state_name: str, detail_str: str, current_step: int, max_step: int):
        self.state_badge.setText(state_name)
        self.state_detail_lbl.setText(detail_str)
        if max_step > 0:
            pct = int(min(100, max(0, (current_step / max_step) * 100)))
            self.progress_bar.setValue(pct)

        self.btn_start.setEnabled(state_name in ["IDLE", "RETURNING HOME"])

        # Dynamically color state badge
        if state_name == "IDLE":
            self.state_badge.setStyleSheet("background-color: #0071C5; color: #FFFFFF; font-weight: bold; padding: 6px;")
        elif state_name == "ROTATING":
            self.state_badge.setStyleSheet("background-color: #E68A00; color: #FFFFFF; font-weight: bold; padding: 6px;")
        elif state_name == "SCANNING":
            self.state_badge.setStyleSheet("background-color: #0088CC; color: #FFFFFF; font-weight: bold; padding: 6px;")
        elif "SORTING" in state_name:
            if "PINK" in state_name or self.worker.defect_found:
                self.state_badge.setStyleSheet("background-color: #FF0055; color: #FFFFFF; font-weight: bold; padding: 6px;")
            else:
                self.state_badge.setStyleSheet("background-color: #00AA44; color: #FFFFFF; font-weight: bold; padding: 6px;")
        elif state_name == "E-STOP":
            self.state_badge.setStyleSheet("background-color: #CC0000; color: #FFFFFF; font-weight: bold; padding: 6px;")

        # Clear filmstrip slots on beginning of new rotation sequence
        if state_name == "ROTATING" and self.worker.face_index == 0 and current_step == 1:
            for slot in self.film_slots:
                slot.reset()

    @Slot(str, str)
    def _on_active_model_update(self, model_name: str, device: str):
        self.model_badge.setText(f"ACTIVE: {model_name} [{device}]")

    @Slot(int, bool, float, object, object)
    def _on_inspection_result(self, face_idx: int, is_defect: bool, score: float, raw_crop: np.ndarray, annotated_crop: np.ndarray):
        if 0 <= face_idx < 4:
            self.film_slots[face_idx].set_result(is_defect, score, annotated_crop)

    @Slot(int, int, int, float)
    def _on_session_stats_update(self, total: int, green: int, pink: int, rate: float):
        self.lbl_stats.setText(
            f"Total Bricks Inspected: {total} | Green Plate (Pass): {green} | "
            f"Pink Plate (Defect): {pink} | Pass Rate: {rate:.1f}%"
        )

    @Slot(str)
    def _on_status_message(self, message: str):
        self.status_bar.showMessage(message, 5000)

    def _on_threshold_changed(self, val: float):
        self.worker.set_anomalib_threshold(val)

    def _on_roi_changed(self):
        self.worker.set_roi(
            self.sp_roi_x.value(),
            self.sp_roi_y.value(),
            self.sp_roi_w.value(),
            self.sp_roi_h.value(),
        )
