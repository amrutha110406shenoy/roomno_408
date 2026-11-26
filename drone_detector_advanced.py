"""
Advanced Drone Detector - Custom filtering for precise drone detection
Filters YOLO detections to identify drones with high confidence
"""
import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO
from collections import defaultdict

class DroneDetector:
    """Advanced drone detection with filtering and post-processing"""

    # COCO classes that could be drones/UAVs
    DRONE_CLASSES = {
        'bird': 14,
        'airplane': 4,
        'frisbee': 32,
        'kite': 33,
    }

    # Classes to filter out (ground objects)
    FILTER_CLASSES = {
        'person': 0,
        'car': 2,
        'dog': 16,
        'cat': 17,
        'horse': 18,
        'cow': 19,
        'sheep': 20,
        'bear': 22,
        'zebra': 24,
        'giraffe': 25,
        'backpack': 26,
        'umbrella': 27,
        'handbag': 31,
        'skateboard': 36,
        'surfboard': 38,
        'tennis racket': 43,
    }

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.6):
        """Initialize drone detector"""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.track_history = {}

    def is_drone_like(self, box_data, frame_shape):
        """Check if detection is likely a drone"""
        cls_id = int(box_data[5])
        conf = float(box_data[4])

        # Filter out ground objects
        if cls_id in self.FILTER_CLASSES.values():
            return False

        # High confidence for drone-like objects
        if cls_id in self.DRONE_CLASSES.values():
            return conf >= self.conf_threshold

        # Additional heuristics for small flying objects
        x1, y1, x2, y2 = box_data[:4]
        h, w = frame_shape[:2]

        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        frame_area = h * w

        # Drone-like size (small relative to frame, aspect ratio)
        aspect_ratio = box_w / (box_h + 1e-6)
        area_ratio = box_area / frame_area

        # Drones are typically small to medium objects
        if 0.0001 <= area_ratio <= 0.3:  # 0.01% to 30% of frame
            # Good aspect ratio for small objects
            if 0.3 <= aspect_ratio <= 3.0:
                # High confidence required
                return conf >= 0.65

        return False

    def filter_detections(self, detections, frame_shape):
        """Filter and score detections for drone likelihood"""
        drone_detections = []

        for det in detections.boxes.data:
            if self.is_drone_like(det, frame_shape):
                drone_detections.append(det)

        return drone_detections

    def process_video(self, video_path, output_path, tracker="bytetrack.yaml"):
        """Process video and detect drones"""

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_idx = 0
        total_drone_dets = 0
        drone_frames = 0
        all_dets = 0
        inference_times = []
        track_ids_set = set()

        print(f"\n[*] Processing: {Path(video_path).name}")
        print(f"[*] Video: {w}x{h}, {fps:.1f}fps, {total_frames} frames")
        print(f"[*] Tracker: {tracker}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Detection with tracker
            t0 = time.time()
            results = self.model.track(
                frame,
                imgsz=640,
                conf=0.3,  # Lower initial threshold for filtering
                tracker=tracker,
                persist=True,
                verbose=False
            )
            inference_time = time.time() - t0
            inference_times.append(inference_time)

            # Count all detections
            all_boxes = results[0].boxes
            all_dets += len(all_boxes)

            # Filter for drones
            drone_boxes = self.filter_detections(results[0], frame.shape)

            if len(drone_boxes) > 0:
                total_drone_dets += len(drone_boxes)
                drone_frames += 1

                # Track drone IDs
                for drone_box in drone_boxes:
                    if hasattr(drone_box, 'id') and drone_box.id is not None:
                        track_ids_set.add(int(drone_box.id.item()))

            # Create annotated frame
            annotated_frame = frame.copy()

            # Draw drone detections
            for drone_box in drone_boxes:
                x1, y1, x2, y2 = map(int, drone_box[:4])
                conf = float(drone_box[4])
                track_id = int(drone_box[5]) if len(drone_box) > 5 else -1

                # Green box for drones
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"DRONE {track_id} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw filtered detections (non-drones) in red
            for box in all_boxes:
                if box not in drone_boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Add stats
            cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Drones: {len(drone_boxes)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Other Objects: {len(all_boxes) - len(drone_boxes)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            writer.write(annotated_frame)

            if frame_idx % max(1, total_frames // 4) == 0:
                avg_inf = np.mean(inference_times) * 1000
                print(f"  Frame {frame_idx}/{total_frames} | Drones: {len(drone_boxes)} | "
                      f"All: {len(all_boxes)} | Inf: {avg_inf:.1f}ms")

        cap.release()
        writer.release()

        total_time = time.time() - t0
        fps_actual = frame_idx / total_time if total_time > 0 else 0

        return {
            'frames': frame_idx,
            'total_drone_detections': total_drone_dets,
            'drone_frames': drone_frames,
            'total_all_detections': all_dets,
            'avg_inference_ms': np.mean(inference_times) * 1000,
            'unique_drone_tracks': len(track_ids_set),
            'output': str(output_path)
        }


def main():
    print("\n" + "="*80)
    print("ADVANCED DRONE DETECTOR - Filtered Detection with Tracking")
    print("="*80)

    output_dir = Path("drone_advanced_outputs")
    output_dir.mkdir(exist_ok=True)

    UAV_VIDEOS = [
        "datasets/istockphoto-1287277966-640_adpp_is.mp4",
        "datasets/istockphoto-2170041017-640_adpp_is.mp4",
        "datasets/istockphoto-2291118401-640_adpp_is.mp4",
        "datasets/istockphoto-2291460380-640_adpp_is.mp4",
    ]

    MODELS = ["yolov8n.pt", "yolov10n.pt"]
    TRACKERS = ["botsort.yaml", "bytetrack.yaml"]

    all_results = []

    print(f"\n[+] Testing {len(UAV_VIDEOS)} videos with advanced drone filtering")
    print(f"[+] Models: {MODELS}")
    print(f"[+] Trackers: {TRACKERS}")

    for model_path in MODELS:
        model_name = model_path.split('.')[0]
        detector = DroneDetector(model_path=model_path, conf_threshold=0.6)

        for tracker in TRACKERS:
            tracker_name = tracker.split('.')[0]

            for video_path in UAV_VIDEOS:
                video_name = Path(video_path).stem

                try:
                    output_name = f"{model_name}_{tracker_name}_DRONE_{video_name}.mp4"
                    output_path = output_dir / output_name

                    result = detector.process_video(video_path, output_path, tracker)
                    result['model'] = model_name
                    result['tracker'] = tracker_name
                    result['video'] = Path(video_path).name

                    all_results.append(result)

                except Exception as e:
                    print(f"  [!] Error: {str(e)[:50]}")

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY - Advanced Drone Detection")
    print("="*80)

    print(f"\n{'Model':<12} {'Tracker':<12} {'Video':<40} {'Drones':<10} {'Tracks':<10}")
    print("-"*100)

    for r in all_results:
        video_short = r['video'][:35] if len(r['video']) > 35 else r['video']
        print(f"{r['model']:<12} {r['tracker']:<12} {video_short:<40} "
              f"{r['total_drone_detections']:<10} {r['unique_drone_tracks']:<10}")

    # Detailed stats
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)

    for r in all_results:
        print(f"\n[+] {r['model']} + {r['tracker']} - {r['video']}")
        print(f"    Total Frames: {r['frames']}")
        print(f"    Drone Detections: {r['total_drone_detections']}")
        print(f"    Frames with Drones: {r['drone_frames']}")
        print(f"    Other Objects Filtered: {r['total_all_detections'] - r['total_drone_detections']}")
        print(f"    Unique Drone Tracks: {r['unique_drone_tracks']}")
        print(f"    Avg Inference: {r['avg_inference_ms']:.1f}ms/frame")
        print(f"    Output: {Path(r['output']).name}")

    # List outputs
    print(f"\n[+] Generated {len(list(output_dir.glob('*.mp4')))} output videos")
    print(f"[+] Location: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
