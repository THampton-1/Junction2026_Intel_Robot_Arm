#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Main Entry Point for Intel Physical AI Quality Assurance Station.
Launches the PySide6 Retro 2000s Intel Native Application.
"""

import argparse
import logging
import signal
import sys
from PySide6.QtWidgets import QApplication

from robot_fsm import RobotControlWorker
from ui_intel2000s import Intel2000sMainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app_main")


def parse_args():
    parser = argparse.ArgumentParser(description="Intel Physical AI QA Station")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in dry-run mode with synthetic camera frames and simulated SO-101 arm states.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Automatically trigger inspection cycle upon UI startup for dry-run verification.",
    )
    return parser.parse_args()


def main():
    # Graceful exit on Ctrl+C in terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    args = parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    logger.info("=" * 65)
    logger.info("  INTEL PHYSICAL AI QUALITY ASSURANCE STATION")
    logger.info("  SO-101 Robotics • OpenVINO™ ACT Policies • Anomalib PatchCore")
    logger.info("=" * 65)
    logger.info(f"Hardware Mode: {'SIMULATED (MOCK DRY RUN)' if args.mock else 'PHYSICAL HARDWARE'}")

    # 1. Start Control Worker (Background 30 Hz QThread)
    worker = RobotControlWorker(mock_mode=args.mock)
    worker.start()

    # 2. Start Intel 2000s PySide6 Main Window
    window = Intel2000sMainWindow(worker)
    window.show()

    # Optional auto-start for automated testing
    if args.auto_start:
        logger.info("[DRY TEST] Auto-start requested: Triggering inspection cycle in 1.5 seconds...")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1500, worker.trigger_start)

    logger.info("Application event loop started successfully.")
    exit_code = app.exec()

    # 3. Clean Shutdown
    logger.info("Shutting down worker and releasing hardware locks...")
    worker.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
