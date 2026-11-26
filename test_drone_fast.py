"""
Fast YOLO drone detection test on NDJSON dataset
"""
import json
import numpy as np
from pathlib import Path
import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

NDJSON_FILE = "datasets/drone-detection-yolo.ndjson"
OUTPUT_DIR = "drone_detection_outputs"
CONF_THRESHOLD = 0.5
TEST_SIZE = 150  # Create 150 frame video

MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov9n": "yolov9n.pt",
    "yolov10n": "yolov10n.pt",
}

TRACKERS = ["botsort.yaml", "bytetrack.yaml"]

TRACKER_NAMES = {
    "botsort.yaml": "BoTSORT",
    "bytetrack.yaml": "ByteTrack"
}


def load_dataset_annotations(ndjson_path, test_size=150):
    """Load sample annotations from NDJSON"""
    print("📂 Loading dataset annotations...")
    annotations = []

    with open(ndjson_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            data = json.loads(line)
            if data.get('type') == 'image':
                annotations.append(data)
                if len(annotations) >= test_size:
                    break

    print(f"  Loaded {len(annotations)} image annotations")

    # Stats
    splits = defaultdict(int)
    for ann in annotations:
        splits[ann.get('split', 'unknown')] += 1
    print(f"  Splits: {dict(splits)}")

    return annotations


def create_video_from_annotations(annotations, output_path, video_w=1280, video_h=720):
    """Create video with ground truth drone annotations"""
    print(f"🎥 Creating video with {len(annotations)} frames...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (video_w, video_h))

    for frame_idx, entry in enumerate(annotations):
        # Create frame with gradient background
        frame = np.zeros((video_h, video_w, 3), dtype=np.uint8)

        # Gradient background
        for y in range(video_h):
            intensity = int(100 + (y / video_h) * 50)
            frame[y, :] = [intensity // 2, intensity, intensity + 30]

        # Add checkerboard pattern
        square_size = 50
        for y in range(0, video_h, square_size * 2):
            for x in range(0, video_w, square_size * 2):
                frame[y:y+square_size, x:x+square_size] = [50, 50, 50]

        # Draw ground truth boxes
        boxes = entry.get('annotations', {}).get('boxes', [])
        for box_data in boxes:
            if len(box_data) >= 5:
                class_id, cx, cy, w, h = box_data[:5]
                # Denormalize
                x1 = int((cx - w / 2) * video_w)
                y1 = int((cy - h / 2) * video_h)
                x2 = int((cx + w / 2) * video_w)
                y2 = int((cy + h / 2) * video_h)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"GT", (x1, max(y1-5, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Add text
        cv2.putText(frame, f"Frame: {frame_idx+1}/{len(annotations)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"File: {entry.get('file', 'N/A')}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Drones: {len(boxes)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        writer.write(frame)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Created {frame_idx + 1}/{len(annotations)} frames...")

    writer.release()
    print(f"  ✓ Video created: {output_path}")
    return len(annotations)


def test_model_on_video(model_name, model_path, tracker, input_video, output_path):
    """Run YOLO model on video with tracker"""
    tracker_name = TRACKER_NAMES[tracker]
    print(f"\n  Testing {model_name} with {tracker_name}...")

    # Load model
    model = YOLO(model_path)

    # Open input video
    cap = cv2.VideoCapture(str(input_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_idx = 0
    total_dets = 0
    inference_times = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run inference with tracker
        t0 = time.time()
        results = model.track(
            frame,
            imgsz=640,
            conf=CONF_THRESHOLD,
            tracker=tracker,
            persist=True,
            verbose=False
        )
        inference_times.append(time.time() - t0)

        # Get detections
        boxes = results[0].boxes
        total_dets += len(boxes)

        # Annotate
        annotated = results[0].plot()
        writer.write(annotated)

        if frame_idx % 50 == 0:
            avg_inf = np.mean(inference_times) * 1000
            print(f"    Frame {frame_idx}/{total_frames} | Dets: {len(boxes)} | Avg Inf: {avg_inf:.1f}ms")

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    fps_actual = frame_idx / total_time if total_time > 0 else 0

    return {
        'model': model_name,
        'tracker': tracker_name,
        'frames': frame_idx,
        'total_detections': total_dets,
        'total_time': total_time,
        'fps_actual': fps_actual,
        'avg_inference_ms': np.mean(inference_times) * 1000,
        'output': str(output_path)
    }


def main():
    print("\n" + "="*80)
    print("🚁 YOLO Drone Detection Test - Fast Mode")
    print("="*80)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Load annotations
    annotations = load_dataset_annotations(NDJSON_FILE, TEST_SIZE)

    # Create test video
    test_video = Path(OUTPUT_DIR) / "test_video_drone.mp4"
    create_video_from_annotations(annotations, test_video)

    # Test all models
    print(f"\n{'='*80}")
    print("Testing Models")
    print(f"{'='*80}\n")

    results = []
    total = len(MODELS) * len(TRACKERS)
    current = 0

    print(f"{'Model':<12} {'Tracker':<12} {'Status':<8} {'FPS':<8} {'Dets':<8} {'Inf(ms)':<10}")
    print("-" * 80)

    for model_name, model_path in MODELS.items():
        for tracker in TRACKERS:
            current += 1
            tracker_name = TRACKER_NAMES[tracker]

            try:
                output_name = f"{model_name}_{tracker_name.lower()}_drone.mp4"
                output_path = Path(OUTPUT_DIR) / output_name

                print(f"[{current}/{total}] {model_name:<12} {tracker_name:<12}", end=" ", flush=True)

                result = test_model_on_video(model_name, model_path, tracker, test_video, output_path)

                print(f"✓ PASS    {result['fps_actual']:>6.2f}   {result['total_detections']:>6}   {result['avg_inference_ms']:>8.1f}")
                results.append(result)

            except Exception as e:
                print(f"✗ FAIL    Error: {str(e)[:40]}")

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Model':<12} {'Tracker':<12} {'FPS':<10} {'Dets':<10} {'Inf/F(ms)':<12} {'Output':<30}")
    print("-" * 80)

    for r in results:
        output_name = Path(r['output']).name
        print(f"{r['model']:<12} {r['tracker']:<12} {r['fps_actual']:>8.2f}   {r['total_detections']:>8}   {r['avg_inference_ms']:>10.1f}   {output_name:<30}")

    print("-" * 80)
    print(f"\nTotal Videos Generated: {len(results)}")
    print(f"Output Directory: {Path(OUTPUT_DIR).absolute()}\n")

    # List output files
    print("Generated Videos:")
    for video_file in sorted(Path(OUTPUT_DIR).glob("*_drone.mp4")):
        size_mb = video_file.stat().st_size / (1024*1024)
        print(f"  ✓ {video_file.name:<40} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
