"""
Quick test: YOLO v8/v9/v10 with BoTSORT + ByteTrack on first 100 frames
Faster verification before running full test
"""
import cv2
import os
import time
from pathlib import Path
from ultralytics import YOLO

INPUT_VIDEO = "video01_botsort_once.mp4"
OUTPUT_DIR = "quick_test_outputs"
CONF_THRESHOLD = 0.5
MAX_FRAMES = 100  # Test only first 100 frames

MODELS = {
    "yolov8n": "yolov8n.pt",
}

TRACKERS = ["botsort.yaml", "bytetrack.yaml"]

TRACKER_NAMES = {
    "botsort.yaml": "BoTSORT",
    "bytetrack.yaml": "ByteTrack"
}


def process_video_sample(model_name, model_path, tracker, output_dir, max_frames=100):
    """Process first N frames of video"""

    tracker_name = TRACKER_NAMES[tracker]
    output_name = f"{model_name}_{tracker_name.lower()}_sample.mp4"
    output_path = Path(output_dir) / output_name

    print(f"\n  Model: {model_name} | Tracker: {tracker_name}")
    print(f"  Loading model: {model_path}")

    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Input: {w}x{h} @ {fps:.1f}fps")

    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    start_time = time.time()
    frame_idx = 0
    detections_total = 0

    print(f"  Processing {max_frames} frames...")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run YOLO with tracker
        results = self.model.track(
            frame,
            imgsz=640,
            conf=CONF_THRESHOLD,
            tracker=tracker,
            persist=True,
            verbose=False
        )

        detections_total += len(results[0].boxes)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

        if frame_idx % 20 == 0:
            print(f"    Frame {frame_idx}/{max_frames} | Detections: {len(results[0].boxes)}")

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    fps_actual = frame_idx / elapsed if elapsed > 0 else 0

    size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0

    print(f"  ✓ Output: {output_name}")
    print(f"    Time: {elapsed:.1f}s | FPS: {fps_actual:.1f} | Size: {size_mb:.2f}MB | Total dets: {detections_total}")

    return str(output_path), elapsed


def main():
    print("=" * 80)
    print("QUICK TEST - First 100 frames only")
    print("=" * 80)

    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print(f"\nInput: {INPUT_VIDEO}")
    print(f"Testing: First {MAX_FRAMES} frames only")
    print(f"Output: {OUTPUT_DIR}/")

    for model_name, model_path in MODELS.items():
        for tracker in TRACKERS:
            try:
                tracker_name = TRACKER_NAMES[tracker]
                process_video_sample(model_name, model_path, tracker, OUTPUT_DIR, MAX_FRAMES)
            except Exception as e:
                print(f"  ✗ FAILED: {e}")

    print("\n" + "=" * 80)
    print("✓ Quick test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
