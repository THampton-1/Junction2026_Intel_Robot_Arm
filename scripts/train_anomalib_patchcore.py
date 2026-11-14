#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Train, evaluate, and export Anomalib PatchCore for Junction 2026 Intel Physical AI Challenge.
Target: Claw Camera (claw_cam) - 20 train normal, 8 test normal (out-of-sample), 16 test abnormal (big blocks).
Hardware: Intel Core Ultra Xe3 iGPU (PyTorch XPU + OpenVINO IR).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Patch PyTorch 2.5+ dynamo-based ONNX exporter to use TorchScript backend (prevents dynamic_axes error)
_orig_onnx_export = torch.onnx.export


def _patched_onnx_export(*args, **kwargs):
    if "dynamo" not in kwargs:
        kwargs["dynamo"] = False
    return _orig_onnx_export(*args, **kwargs)


torch.onnx.export = _patched_onnx_export

# Configure standard logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("anomalib_patchcore")

# Ensure anomalib imports resolve cleanly
from anomalib.data import Folder
from anomalib.data.utils.split import ValSplitMode
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Anomalib PatchCore on Intel Xe3 iGPU")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/intel/anomalib_data/anomalib_ready/claw_cam",
        help="Path to dataset root folder (contains train/ and test/)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="wide_resnet50_2",
        choices=["wide_resnet50_2", "resnet18", "resnet50"],
        help="Feature extractor backbone (default: wide_resnet50_2)",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.2,
        help="Coreset subsampling ratio for memory bank (default: 0.2)",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=9,
        help="Number of nearest neighbors for anomaly scoring (default: 9)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="/home/intel/models/anomalib_patchcore_claw",
        help="Output directory for OpenVINO IR model export",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/intel/models/anomalib_results",
        help="Directory to store training logs and checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Dataset root does not exist: {data_root}")
        sys.exit(1)

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hardware Detection
    has_xpu = torch.xpu.is_available()
    device_name = torch.xpu.get_device_name(0) if has_xpu else "CPU"
    logger.info("=" * 65)
    logger.info("  ANOMALIB PATCHCORE TRAINING & EVALUATION PIPELINE")
    logger.info("=" * 65)
    logger.info(f"Feature Extraction:  CPU (Fast multi-core feature aggregation)")
    logger.info(f"Target OpenVINO Acc: Intel Xe3 iGPU ({device_name})")
    logger.info(f"Dataset Root:        {data_root}")
    logger.info(f"Backbone:            {args.backbone}")
    logger.info(f"Coreset Ratio:       {args.coreset_ratio}")
    logger.info(f"Num Neighbors:       {args.num_neighbors}")
    logger.info(f"OpenVINO Export Dir: {export_dir}")

    # 2. Configure Folder DataModule
    logger.info("Setting up Claw Cam dataset...")
    datamodule = Folder(
        name="claw_cam",
        root=data_root,
        normal_dir="train/normal",
        normal_test_dir="test/normal",
        abnormal_dir="test/abnormal",
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=2,
    )
    datamodule.setup()
    logger.info(f"Dataset loaded: {len(datamodule.train_data)} train normal, {len(datamodule.test_data)} test total")

    # 3. Instantiate PatchCore Model
    logger.info(f"Initializing PatchCore model with {args.backbone} backbone...")
    model = Patchcore(
        backbone=args.backbone,
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=args.num_neighbors,
        pre_trained=True,
        visualizer=False,
    )

    # 4. Engine & Training (Building Normal Feature Memory Bank)
    engine = Engine(
        default_root_dir=Path(args.results_dir),
        accelerator="cpu",
        devices=1,
    )

    logger.info("Starting memory bank construction...")
    train_start = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    train_time = time.perf_counter() - train_start
    logger.info(f"PatchCore memory bank built in {train_time:.2f} seconds!")

    # 5. Evaluation on Test Set
    logger.info("Running evaluation on test set (8 Normal, 16 Abnormal)...")
    eval_start = time.perf_counter()
    test_results = engine.test(model=model, datamodule=datamodule)
    eval_time = time.perf_counter() - eval_start
    logger.info(f"Test set evaluation completed in {eval_time:.2f} seconds.")
    logger.info(f"Test Results Summary: {test_results}")

    # 6. Per-Image Score Analysis & Separation
    logger.info("Predicting detailed anomaly scores across test samples...")
    predictions = engine.predict(model=model, datamodule=datamodule)

    normal_scores = []
    abnormal_scores = []
    logger.info("-" * 75)
    logger.info(f"{'IMAGE NAME':<25} {'LABEL':<12} {'PRED':<12} {'ANOMALY SCORE':<15}")
    logger.info("-" * 75)

    for batch in predictions:
        paths = batch.image_path if hasattr(batch, "image_path") else batch.get("image_path", [])
        labels = batch.gt_label if hasattr(batch, "gt_label") else batch.get("gt_label", [])
        preds = batch.pred_label if hasattr(batch, "pred_label") else batch.get("pred_label", [])
        scores = batch.pred_score if hasattr(batch, "pred_score") else batch.get("pred_score", [])

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        for path_str, label, pred, score in zip(paths, labels, preds, scores):
            name = Path(path_str).name
            lbl_str = "Normal" if int(label) == 0 else "Abnormal"
            pred_str = "Normal" if int(pred) == 0 else "Abnormal"
            float_score = float(score)

            if int(label) == 0:
                normal_scores.append((name, float_score))
            else:
                abnormal_scores.append((name, float_score))

            logger.info(f"{name:<25} {lbl_str:<12} {pred_str:<12} {float_score:<15.4f}")

    logger.info("-" * 75)
    if normal_scores and abnormal_scores:
        norm_vals = np.array([s for _, s in normal_scores])
        abnorm_vals = np.array([s for _, s in abnormal_scores])
        logger.info(f"NORMAL SCORES   (Small Blocks, N={len(norm_vals)}):  min={norm_vals.min():.4f}, mean={norm_vals.mean():.4f}, max={norm_vals.max():.4f}")
        logger.info(f"ABNORMAL SCORES (Big Blocks,   N={len(abnorm_vals)}): min={abnorm_vals.min():.4f}, mean={abnorm_vals.mean():.4f}, max={abnorm_vals.max():.4f}")
        separation = abnorm_vals.min() - norm_vals.max()
        recommended_threshold = (norm_vals.max() + abnorm_vals.min()) / 2.0
        logger.info(f"SEPARATION GAP (Abnormal Min - Normal Max): {separation:.4f}")
        logger.info(f"RECOMMENDED THRESHOLD:                     {recommended_threshold:.4f}")
        if separation > 0:
            logger.info(">>> PERFECT SEPARATION ACHIEVED! Zero overlap between Normal and Abnormal classes. <<<")
        else:
            logger.warning(f"Overlap detected between classes (Margin: {separation:.4f}).")

    # 7. OpenVINO IR Export
    logger.info("Exporting PatchCore model to OpenVINO IR...")
    exported_model_path = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=export_dir,
        model_file_name="patchcore_claw",
        input_size=(240, 320),
    )
    logger.info(f"OpenVINO IR successfully exported to: {exported_model_path}")

    # 8. Test OpenVINO Inference Latency on Intel Hardware
    logger.info("Benchmarking OpenVINO runtime inference on Intel hardware...")
    try:
        import openvino as ov
        core = ov.Core()
        available_devices = core.available_devices
        logger.info(f"OpenVINO Available Devices: {available_devices}")

        xml_path = export_dir / "patchcore_claw.xml"
        if not xml_path.exists():
            candidates = list(export_dir.rglob("*.xml"))
            if candidates:
                xml_path = candidates[0]

        if xml_path.exists():
            logger.info(f"Loading OpenVINO model from: {xml_path}")
            ov_model = core.read_model(str(xml_path))

            target_device = "GPU" if "GPU" in available_devices else "CPU"
            logger.info(f"Compiling OpenVINO model for device: {target_device} (Intel Core Ultra Xe3)")
            compiled_model = core.compile_model(ov_model, target_device)

            dummy_input = np.random.randn(1, 3, 240, 320).astype(np.float32)

            # Warmup
            for _ in range(5):
                _ = compiled_model([dummy_input])

            # Measure latency over 30 iterations
            latencies = []
            for _ in range(30):
                t0 = time.perf_counter()
                _ = compiled_model([dummy_input])
                latencies.append((time.perf_counter() - t0) * 1000.0)

            avg_lat = np.mean(latencies)
            p95_lat = np.percentile(latencies, 95)
            logger.info(f"OpenVINO ({target_device}) Latency: Mean = {avg_lat:.2f} ms | P95 = {p95_lat:.2f} ms")
            logger.info("SUCCESS: Anomalib PatchCore ready for live closed-loop gate!")
        else:
            logger.warning(f"Could not find patchcore_claw.xml in {export_dir}")
    except Exception as exc:
        logger.exception(f"Error during OpenVINO benchmark: {exc}")

    logger.info("=" * 65)
    logger.info("  TRAINING, EVALUATION & EXPORT COMPLETE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
