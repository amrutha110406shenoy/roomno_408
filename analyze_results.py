"""
Analyze test results - compare detection metrics across YOLO versions and trackers
"""
import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

def analyze_video(video_path):
    """Extract detection statistics from a video"""
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0

    total_detections = 0
    frames_with_detections = 0
    max_detections_frame = 0
    track_ids = set()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # This is a rough estimate - we're just counting frames
        # For real detection counts, we'd need to parse the annotations

    cap.release()

    return {
        'frame_count': frame_count,
        'fps': fps,
        'duration_sec': duration,
        'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    }


def compare_models():
    """Generate comparison report"""
    output_dir = Path("test_outputs")

    if not output_dir.exists():
        print("❌ No output directory found. Run test_yolo_tracker.py first.")
        return

    print("=" * 100)
    print("YOLO DETECTION & TRACKING - RESULTS ANALYSIS")
    print("=" * 100)

    video_files = sorted(output_dir.glob("*.mp4"))

    if not video_files:
        print("❌ No output videos found")
        return

    print(f"\nFound {len(video_files)} output videos\n")

    results = {}

    print(f"{'File':<40} {'Size (MB)':<12} {'Duration (s)':<15} {'FPS':<8}")
    print("-" * 100)

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration = frame_count / fps if fps > 0 else 0
        size_mb = video_path.stat().st_size / (1024 * 1024)

        filename = video_path.name
        print(f"{filename:<40} {size_mb:>10.2f} MB {duration:>13.1f}s {fps:>6.1f}")

        # Parse model and tracker from filename
        parts = filename.replace('.mp4', '').split('_')
        if len(parts) >= 2:
            model = parts[0]
            tracker = '_'.join(parts[1:])
            results[f"{model}_{tracker}"] = {
                'file': filename,
                'size_mb': size_mb,
                'duration': duration,
                'fps': fps,
                'resolution': f"{w}x{h}"
            }

    # Summary by model
    print("\n" + "=" * 100)
    print("SUMMARY BY MODEL")
    print("=" * 100)

    models = {}
    for key, data in results.items():
        model = key.split('_')[0]
        if model not in models:
            models[model] = []
        models[model].append((key, data))

    for model in sorted(models.keys()):
        print(f"\n{model.upper()}:")
        for key, data in models[model]:
            tracker = key.split('_', 1)[1]
            print(f"  • {tracker:<12} | {data['file']:<30} | {data['size_mb']:>7.1f}MB | {data['duration']:>6.1f}s")

    # Summary by tracker
    print("\n" + "=" * 100)
    print("SUMMARY BY TRACKER")
    print("=" * 100)

    trackers = {}
    for key, data in results.items():
        tracker = key.split('_', 1)[1]
        if tracker not in trackers:
            trackers[tracker] = []
        trackers[tracker].append((key, data))

    for tracker in sorted(trackers.keys()):
        print(f"\n{tracker.upper()}:")
        total_size = 0
        for key, data in trackers[tracker]:
            model = key.split('_')[0]
            print(f"  • {model:<12} | {data['file']:<30} | {data['size_mb']:>7.1f}MB | {data['duration']:>6.1f}s")
            total_size += data['size_mb']
        print(f"  Total size: {total_size:.1f}MB")

    # Save results to JSON
    json_path = output_dir / "results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {json_path}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    compare_models()
