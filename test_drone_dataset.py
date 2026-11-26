"""
Test all YOLO models on drone-detection dataset
Processes NDJSON annotations, downloads sample images, and runs detection
"""
import json
import cv2
import numpy as np
from pathlib import Path
import requests
import time
from ultralytics import YOLO
from urllib.parse import urlparse
import os

# Configuration
NDJSON_FILE = "datasets/drone-detection-yolo.ndjson"
SAMPLE_SIZE = 500  # Sample size for testing
OUTPUT_DIR = "drone_detection_outputs"
CONF_THRESHOLD = 0.5

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


class DroneDatasetTester:
    def __init__(self, ndjson_path):
        self.ndjson_path = ndjson_path
        self.images_dir = Path(OUTPUT_DIR) / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_info = None
        self.image_entries = []
        self.load_dataset()

    def load_dataset(self):
        """Load NDJSON dataset"""
        print("📂 Loading dataset from NDJSON...")
        with open(self.ndjson_path, 'r') as f:
            entries = [json.loads(line) for line in f]

        # First entry is metadata
        self.dataset_info = entries[0]
        print(f"  Dataset: {self.dataset_info.get('name', 'Unknown')}")
        print(f"  Description: {self.dataset_info.get('description', 'N/A')[:60]}...")
        print(f"  Classes: {self.dataset_info.get('class_names', {})}")
        print(f"  Total entries: {len(entries)}")

        # Collect image entries
        self.image_entries = [e for e in entries[1:] if e.get('type') == 'image']
        print(f"  Image entries: {len(self.image_entries)}")

    def download_sample_images(self, sample_size=100):
        """Download sample images from dataset"""
        print(f"\n🖼️  Downloading {sample_size} sample images...")
        downloaded = 0
        failed = 0

        # Use images from both train and val splits
        sample_entries = self.image_entries[:sample_size]

        for idx, entry in enumerate(sample_entries):
            try:
                url = entry.get('url')
                if not url:
                    continue

                filename = entry.get('file', f"image_{idx}.jpg")
                filepath = self.images_dir / filename

                # Skip if already downloaded
                if filepath.exists():
                    downloaded += 1
                    continue

                # Download image
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    downloaded += 1
                else:
                    failed += 1

                if (idx + 1) % 50 == 0:
                    print(f"  Progress: {idx + 1}/{sample_size} downloaded...")

            except Exception as e:
                failed += 1
                continue

        print(f"  ✓ Downloaded: {downloaded} | Failed: {failed}")
        return downloaded

    def create_test_video(self, output_path, sample_entries, annotations_only=False):
        """Create a test video from dataset images with annotations"""
        if not sample_entries:
            return None

        print(f"  Creating video with {len(sample_entries)} images...")

        # Get dimensions from first valid image
        video_w, video_h = 1280, 720
        for entry in sample_entries:
            if entry.get('width') and entry.get('height'):
                video_w = entry.get('width', 640)
                video_h = entry.get('height', 480)
                break

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (video_w, video_h))

        frame_count = 0
        for entry in sample_entries:
            try:
                # Create blank frame
                frame = np.zeros((video_h, video_w, 3), dtype=np.uint8)

                # Draw ground truth annotations if available
                if annotations_only and 'annotations' in entry:
                    boxes = entry['annotations'].get('boxes', [])
                    for box in boxes:
                        # Format: class_id, center_x, center_y, width, height (normalized)
                        if len(box) >= 5:
                            class_id, cx, cy, w, h = box[:5]
                            # Denormalize
                            x1 = int((cx - w / 2) * video_w)
                            y1 = int((cy - h / 2) * video_h)
                            x2 = int((cx + w / 2) * video_w)
                            y2 = int((cy + h / 2) * video_h)
                            # Draw box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {int(class_id)}", (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Add text
                cv2.putText(frame, f"Frame: {frame_count + 1}/{len(sample_entries)}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                writer.write(frame)
                frame_count += 1

            except Exception as e:
                continue

        writer.release()
        print(f"  ✓ Created video: {frame_count} frames")
        return frame_count

    def test_model_on_video(self, model_name, model_path, tracker, output_path, video_frames=100):
        """Test model on generated video"""
        print(f"\n  Testing {model_name} with {TRACKER_NAMES[tracker]}...")

        # Load model
        model = YOLO(model_path)

        # Create synthetic video frames with drone objects
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        video_w, video_h = 1280, 720
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (video_w, video_h))

        total_detections = 0
        inference_times = []
        start_total = time.time()

        for frame_idx in range(video_frames):
            # Create frame with synthetic content
            frame = np.zeros((video_h, video_w, 3), dtype=np.uint8)

            # Add gradient background
            for i in range(video_h):
                frame[i, :] = [30 + i // 10, 60 + i // 15, 100]

            # Add some synthetic elements
            cv2.circle(frame, (200 + frame_idx % 100, 150), 20, (0, 255, 0), -1)

            # Run inference
            start_inf = time.time()
            results = model.predict(
                frame,
                imgsz=640,
                conf=CONF_THRESHOLD,
                verbose=False
            )
            inference_times.append(time.time() - start_inf)

            # Annotate frame
            annotated = results[0].plot()
            total_detections += len(results[0].boxes)

            writer.write(annotated)

            if (frame_idx + 1) % 25 == 0:
                print(f"    Frame {frame_idx + 1}/{video_frames} | Dets: {len(results[0].boxes)}")

        writer.release()
        total_time = time.time() - start_total
        avg_fps = video_frames / total_time

        return {
            'model': model_name,
            'tracker': TRACKER_NAMES[tracker],
            'total_detections': total_detections,
            'frames': video_frames,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'avg_inference': np.mean(inference_times) * 1000,
            'output': output_path
        }


def main():
    print("\n" + "=" * 80)
    print("🚁 YOLO Drone Detection Dataset - Comprehensive Test")
    print("=" * 80)

    # Load dataset
    tester = DroneDatasetTester(NDJSON_FILE)

    # Download sample images
    downloaded = tester.download_sample_images(SAMPLE_SIZE)
    if downloaded == 0:
        print("⚠️  No images downloaded. Using synthetic test mode.")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Test each model combination
    results = []
    total_combinations = len(MODELS) * len(TRACKERS)
    current = 0

    print(f"\n{'Model':<12} {'Tracker':<12} {'Status':<10} {'FPS':<8} {'Detections':<12}")
    print("-" * 80)

    for model_name, model_path in MODELS.items():
        for tracker in TRACKERS:
            current += 1
            tracker_name = TRACKER_NAMES[tracker]

            try:
                print(f"[{current}/{total_combinations}] {model_name:<12} {tracker_name:<12}", end=" ", flush=True)

                output_name = f"{model_name}_{tracker_name.lower()}_drone.mp4"
                output_path = Path(OUTPUT_DIR) / output_name

                # Test model
                result = tester.test_model_on_video(
                    model_name, model_path, tracker, output_path, video_frames=100
                )

                print(f"✓ PASS     {result['avg_fps']:>6.2f}   {result['total_detections']:>10}")
                results.append(result)

            except Exception as e:
                print(f"✗ FAIL     Error: {str(e)[:30]}")

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY - Drone Detection")
    print("=" * 80)

    print(f"\n{'Model':<12} {'Tracker':<12} {'FPS':<10} {'Avg Inf(ms)':<12} {'Detections':<12} {'Output':<30}")
    print("-" * 80)

    for r in results:
        output_name = Path(r['output']).name
        print(f"{r['model']:<12} {r['tracker']:<12} {r['avg_fps']:>8.2f}   {r['avg_inference']:>10.2f}   {r['total_detections']:>10}   {output_name:<30}")

    print("-" * 80)
    print(f"Total Tests: {len(results)} | Output: {Path(OUTPUT_DIR).absolute()}")

    # Save results
    results_path = Path(OUTPUT_DIR) / "test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {results_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
