#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Automated Dry-Run Verification Script.
Validates the complete closed-loop inspection and sorting FSM in mock mode.
"""

import sys
import time
import logging
from PySide6.QtCore import QCoreApplication

from robot_fsm import RobotControlWorker, FSMState

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("dry_test")


def main():
    app = QCoreApplication([])

    logger.info("Initializing RobotControlWorker in MOCK mode...")
    worker = RobotControlWorker(mock_mode=True)
    worker.start()

    time.sleep(1.0)
    logger.info("Triggering Start Inspection...")
    worker.trigger_start()

    start_time = time.time()
    last_state = None
    inspected_faces = set()

    def on_inspection(face_idx, is_defect, score, raw, annotated):
        logger.info(f"===> INSPECTION RESULT: Face {face_idx + 1} | Defect: {is_defect} | Score: {score:.4f}")
        inspected_faces.add(face_idx)

    def on_stats(total, green, pink, rate):
        logger.info(f"===> SESSION STATS: Total={total}, Green={green}, Pink={pink}, PassRate={rate:.1f}%")

    worker.inspection_result.connect(on_inspection)
    worker.session_stats_update.connect(on_stats)

    # Monitor FSM execution for up to 30 seconds
    while time.time() - start_time < 30.0:
        app.processEvents()
        current_state = worker.state
        if current_state != last_state:
            logger.info(f"[FSM STATE CHANGE] -> {current_state.value} (Face: {worker.face_index})")
            last_state = current_state

        if worker.total_inspected >= 1:
            logger.info("Inspection & sorting cycle finished successfully!")
            break

        time.sleep(0.05)

    worker.stop()

    assert len(inspected_faces) == 4, f"Expected 4 faces inspected, got {len(inspected_faces)}"
    assert worker.defect_found is True, "Expected defect to be detected on simulated defective face!"
    assert worker.pink_count == 1, "Expected block to be routed to Pink plate!"
    logger.info("ALL DRY-RUN VERIFICATION ASSERTIONS PASSED! 100% SUCCESS.")


if __name__ == "__main__":
    main()
