"""
Test script: YOLO v8/v9/v10 with BoTSORT + ByteTrack on demo video
Generates output videos with bounding boxes for each model+tracker combination
"""
import cv2
import os
import time
from pathlib import Path
from ultralytics import YOLO

# Configuration
INPUT_VIDEO = "video01_botsort_once.mp4"
OUTPUT_DIR = "test_outputs"
CONF_THRESHOLD = 0.5

# Models to test (nano size for speed)
MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov9n": "yolov9n.pt",
    "yolov10n": "yolov10n.pt",
}

# Trackers to test
TRACKERS = ["botsort.yaml", "bytetrack.yaml"]

TRACKER_NAMES = {
    "botsort.yaml": "BoTSORT",
    "bytetrack.yaml": "ByteTrack"
}

class YOLOTester:
    def __init__(self, model_name, model_path, tracker, output_dir):
        self.model_name = model_name
        self.model_path = model_path
        self.tracker = tracker
        self.tracker_name = TRACKER_NAMES[tracker]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load model
        print(f"  Loading model: {model_path}")
        self.model = YOLO(model_path)

        # Setup output video
        self.setup_output_video()

    def setup_output_video(self):
        """Setup video writer for output"""
        output_name = f"{self.model_name}_{self.tracker_name.lower()}.mp4"
        self.output_path = self.output_dir / output_name
        self.writer = None
        self.fps = None
        self.frame_count = 0

    def process_video(self, input_path):
        """Process video with YOLO + tracker"""
        print(f"\n  Processing: {input_path}")

        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  Video info: {w}x{h}, {self.fps:.1f}fps, {total_frames} frames")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (w, h)
        )

        start_time = time.time()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Run YOLO with tracker
            results = self.model.track(
                frame,
                imgsz=640,
                conf=CONF_THRESHOLD,
                tracker=self.tracker,
                persist=True,
                verbose=False
            )

            # Draw boxes and IDs
            annotated_frame = results[0].plot()

            # Write frame
            self.writer.write(annotated_frame)

            # Progress
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_real = frame_idx / elapsed
                eta = (total_frames - frame_idx) / fps_real if fps_real > 0 else 0
                print(f"    Frame {frame_idx}/{total_frames} | FPS: {fps_real:.1f} | ETA: {eta:.0f}s | Dets: {len(results[0].boxes)}")

        cap.release()
        self.writer.release()

        elapsed = time.time() - start_time
        print(f"  ✓ Output: {self.output_path}")
        print(f"    Processing time: {elapsed:.1f}s ({elapsed/total_frames*1000:.1f}ms per frame)")

        return str(self.output_path), elapsed, total_frames


def main():
    print("=" * 80)
    print("YOLO Object Detection & Tracking - Comprehensive Test")
    print("=" * 80)

    # Check input video
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return

    print(f"\nInput video: {INPUT_VIDEO}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Test results
    results = []

    # Test each combination
    total_combinations = len(MODELS) * len(TRACKERS)
    current = 0

    print(f"\n{'Model':<12} {'Tracker':<12} {'Status':<10} {'Time (s)':<10}")
    print("-" * 80)

    for model_name, model_path in MODELS.items():
        for tracker in TRACKERS:
            current += 1
            tracker_name = TRACKER_NAMES[tracker]

            # Check if model file exists, if not download it
            if not os.path.exists(model_path):
                print(f"\n[{current}/{total_combinations}] {model_name:<12} {tracker_name:<12} Downloading model...")
                # Ultralytics will auto-download
                model_path = model_name

            try:
                print(f"[{current}/{total_combinations}] {model_name:<12} {tracker_name:<12}", end=" ", flush=True)

                tester = YOLOTester(model_name, model_path, tracker, OUTPUT_DIR)
                output_file, elapsed, frames = tester.process_video(INPUT_VIDEO)

                print(f"✓ PASS     {elapsed:>8.1f}s")

                results.append({
                    'model': model_name,
                    'tracker': tracker_name,
                    'status': 'PASS',
                    'time': elapsed,
                    'frames': frames,
                    'output': output_file
                })

            except Exception as e:
                print(f"✗ FAIL     Error: {str(e)[:40]}")
                results.append({
                    'model': model_name,
                    'tracker': tracker_name,
                    'status': 'FAIL',
                    'error': str(e),
                    'output': None
                })

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')

    print(f"\n{'Model':<12} {'Tracker':<12} {'Status':<8} {'Time (s)':<12} {'Output File':<30}")
    print("-" * 80)

    for r in results:
        output_name = Path(r['output']).name if r['output'] else "N/A"
        time_str = f"{r['time']:.2f}" if 'time' in r else "ERROR"
        print(f"{r['model']:<12} {r['tracker']:<12} {r['status']:<8} {time_str:<12} {output_name:<30}")

    print("-" * 80)
    print(f"Total: {len(results)} | Passed: {passed} ✓ | Failed: {failed} ✗")

    # Print output locations
    print(f"\n📁 Output videos saved to: {Path(OUTPUT_DIR).absolute()}")
    print("\nGenerated videos:")
    for r in results:
        if r['status'] == 'PASS':
            print(f"  • {Path(r['output']).name}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
