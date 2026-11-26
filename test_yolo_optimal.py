"""
Optimized test: YOLO v8/v9/v10 with BoTSORT + ByteTrack
Tests all combinations with progress feedback
"""
import cv2
import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# Configuration
INPUT_VIDEO = "video01_botsort_once.mp4"
OUTPUT_DIR = "test_outputs"
CONF_THRESHOLD = 0.5

# Models to test - nano for speed
MODELS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv9n": "yolov9n.pt",
    "YOLOv10n": "yolov10n.pt",
}

TRACKERS = {
    "BoTSORT": "botsort.yaml",
    "ByteTrack": "bytetrack.yaml"
}


class ProgressBar:
    def __init__(self, total, label=""):
        self.total = total
        self.label = label
        self.current = 0

    def update(self, amount=1):
        self.current += amount
        pct = self.current / self.total * 100 if self.total > 0 else 0
        bar_len = int(pct / 2)  # 50 char bar
        bar = "█" * bar_len + "░" * (50 - bar_len)
        sys.stdout.write(f'\r{self.label} |{bar}| {pct:>5.1f}%')
        sys.stdout.flush()


def test_combination(model_name, model_path, tracker_name, tracker_path, output_dir):
    """Test one model + tracker combination"""

    # Load model
    print(f"\n  Loading {model_name}...", end="", flush=True)
    try:
        model = YOLO(model_path)
        print(" ✓")
    except Exception as e:
        print(f" ✗ Error: {e}")
        return None, None, 0

    # Setup output
    output_name = f"{model_name.replace('v', '')}_{tracker_name.lower()}.mp4"
    output_path = Path(output_dir) / output_name

    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"  ✗ Cannot open input video")
        return None, None, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Input video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"  ✗ Cannot create output video writer")
        cap.release()
        return None, None, 0

    start_time = time.time()
    frame_idx = 0
    det_count = 0

    progress = ProgressBar(total_frames, f"  Processing {tracker_name:<12}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Run detection with tracking
            results = model.track(
                frame,
                imgsz=640,
                conf=CONF_THRESHOLD,
                tracker=tracker_path,
                persist=True,
                verbose=False
            )

            # Get detection count
            det_count += len(results[0].boxes)

            # Annotate frame
            annotated = results[0].plot()

            # Write frame
            writer.write(annotated)

            progress.update(1)

        cap.release()
        writer.release()

        elapsed = time.time() - start_time

        print(f"\n  ✓ Completed in {elapsed:.1f}s")

        # Get output file size
        size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0

        return str(output_path), elapsed, size_mb

    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        cap.release()
        if writer.isOpened():
            writer.release()
        return None, None, 0


def main():
    print("\n" + "=" * 100)
    print(" " * 25 + "YOLO DETECTION & TRACKING - COMPREHENSIVE TEST")
    print("=" * 100)

    # Check input
    if not os.path.exists(INPUT_VIDEO):
        print(f"\n❌ Input video not found: {INPUT_VIDEO}\n")
        return

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print(f"\n📁 Input:  {INPUT_VIDEO}")
    print(f"📁 Output: {OUTPUT_DIR}/")

    # Test matrix
    combinations = []
    for model_name, model_path in MODELS.items():
        for tracker_name, tracker_path in TRACKERS.items():
            combinations.append((model_name, model_path, tracker_name, tracker_path))

    total_tests = len(combinations)

    print(f"\n📊 Test plan: {total_tests} combinations")
    print(f"   • Models:  {', '.join(MODELS.keys())}")
    print(f"   • Trackers: {', '.join(TRACKERS.keys())}")

    # Run tests
    results = []
    start_all = time.time()

    print(f"\n{'#':<4} {'Model':<10} {'Tracker':<12} {'Status':<8} {'Time (s)':<10} {'Size (MB)':<10}")
    print("-" * 100)

    for idx, (model_name, model_path, tracker_name, tracker_path) in enumerate(combinations, 1):
        try:
            output_path, elapsed, size_mb = test_combination(
                model_name, model_path, tracker_name, tracker_path, OUTPUT_DIR
            )

            if output_path:
                print(f"{idx:<4} {model_name:<10} {tracker_name:<12} ✓ PASS    {elapsed:>8.1f}  {size_mb:>8.2f}")
                results.append({
                    'model': model_name,
                    'tracker': tracker_name,
                    'status': 'PASS',
                    'time': elapsed,
                    'size_mb': size_mb,
                    'output': output_path
                })
            else:
                print(f"{idx:<4} {model_name:<10} {tracker_name:<12} ✗ FAIL")
                results.append({
                    'model': model_name,
                    'tracker': tracker_name,
                    'status': 'FAIL'
                })
        except Exception as e:
            print(f"{idx:<4} {model_name:<10} {tracker_name:<12} ✗ ERROR   {str(e)[:30]}")
            results.append({
                'model': model_name,
                'tracker': tracker_name,
                'status': 'ERROR',
                'error': str(e)
            })

    # Summary
    total_time = time.time() - start_all
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] != 'PASS')

    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)

    print(f"\n✓ Passed: {passed}/{total_tests}")
    print(f"✗ Failed: {failed}/{total_tests}")
    print(f"⏱ Total time: {total_time:.1f}s\n")

    # List output files
    if passed > 0:
        print("📹 Generated videos:")
        for r in sorted(results, key=lambda x: x['model']):
            if r['status'] == 'PASS':
                output_name = Path(r['output']).name
                print(f"   • {output_name:<40} {r['size_mb']:>8.2f}MB ({r['time']:>6.1f}s)")

    print("\n" + "=" * 100)
    print("✓ Testing complete!")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
