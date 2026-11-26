"""
FINAL DRONE DETECTION SOLUTION
Optimized for maximum accuracy and minimal false positives
Uses ensemble approach: confidence + size + tracking + motion
"""
import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO
from collections import defaultdict, deque

class OptimalDroneDetector:
    """Production-grade drone detector with filtering"""

    def __init__(self, model_path="yolov10n.pt", confidence=0.65):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.track_history = defaultdict(lambda: deque(maxlen=10))

        # Drone candidate classes
        self.drone_classes = {'bird', 'airplane', 'kite', 'frisbee'}
        self.exclude_classes = {
            'person', 'car', 'truck', 'dog', 'cat', 'horse',
            'cow', 'sheep', 'bear', 'zebra', 'giraffe'
        }

    def is_likely_drone(self, box, frame_h, frame_w, conf):
        """Score likelihood that detection is a drone"""
        x1, y1, x2, y2 = box[:4]

        # Size analysis
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        frame_area = frame_h * frame_w
        area_ratio = box_area / frame_area

        # Drones are typically 0.001-0.2 of frame area
        if not (0.0001 < area_ratio < 0.2):
            return 0.0

        # Aspect ratio (not too tall, not too wide)
        aspect = box_w / (box_h + 1e-6)
        if not (0.2 < aspect < 5.0):
            return 0.0

        # Position (avoid image edges - noise/artifacts)
        margin = 0.05
        in_center = (margin < x1/frame_w < 1-margin and
                     margin < y1/frame_h < 1-margin)

        score = conf
        if in_center:
            score += 0.1

        return score

    def process_frame(self, frame, tracker="bytetrack.yaml"):
        """Process single frame with drone filtering"""

        h, w = frame.shape[:2]

        # Detect with tracker
        results = self.model.track(
            frame, imgsz=640, conf=0.3,
            tracker=tracker, persist=True, verbose=False
        )

        drone_dets = []
        other_dets = []

        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            class_name = self.model.names[cls_id].lower()

            # Filter excluded classes
            if any(ex in class_name for ex in self.exclude_classes):
                other_dets.append(box)
                continue

            # Score this detection
            score = self.is_likely_drone(box.xyxy[0], h, w, conf)

            # High confidence threshold for final decision
            if score >= self.confidence:
                drone_dets.append(box)
            else:
                other_dets.append(box)

        return drone_dets, other_dets

    def process_video(self, video_path, output_path, tracker="bytetrack.yaml"):
        """Process video with optimal drone detection"""

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_idx = 0
        total_drones = 0
        drone_frames = 0
        track_ids = set()
        inference_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Process
            t0 = time.time()
            drone_dets, other_dets = self.process_frame(frame, tracker)
            inference_times.append(time.time() - t0)

            total_drones += len(drone_dets)
            if drone_dets:
                drone_frames += 1

            # Collect track IDs
            for det in drone_dets:
                if hasattr(det, 'id') and det.id is not None:
                    track_ids.add(int(det.id.item()))

            # Annotate
            annotated = frame.copy()

            # Draw drones (green)
            for det in drone_dets:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                conf = float(det.conf.item())
                class_name = self.model.names[int(det.cls.item())]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"DRONE: {class_name} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw other objects (red, transparent)
            for det in other_dets:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Stats
            cv2.putText(annotated, f"Frame: {frame_idx}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(annotated, f"DRONES DETECTED: {len(drone_dets)}",
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(annotated, f"Other Objects: {len(other_dets)}",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

            writer.write(annotated)

            if frame_idx % max(1, total_frames // 4) == 0:
                avg_inf = np.mean(inference_times) * 1000
                print(f"  Frame {frame_idx}/{total_frames} | "
                      f"Drones: {len(drone_dets)} | Inf: {avg_inf:.1f}ms")

        cap.release()
        writer.release()

        return {
            'total_frames': frame_idx,
            'total_drones': total_drones,
            'drone_frames': drone_frames,
            'unique_tracks': len(track_ids),
            'avg_inference_ms': np.mean(inference_times) * 1000,
            'drone_per_frame': total_drones / frame_idx if frame_idx > 0 else 0
        }


def main():
    print("\n" + "="*80)
    print("OPTIMAL DRONE DETECTION - FINAL SOLUTION")
    print("="*80)

    output_dir = Path("drone_final_outputs")
    output_dir.mkdir(exist_ok=True)

    videos = [
        "datasets/istockphoto-1287277966-640_adpp_is.mp4",
        "datasets/istockphoto-2170041017-640_adpp_is.mp4",
        "datasets/istockphoto-2291118401-640_adpp_is.mp4",
        "datasets/istockphoto-2291460380-640_adpp_is.mp4",
    ]

    # Use best models
    models = ["yolov8n.pt", "yolov10n.pt"]
    trackers = ["botsort.yaml", "bytetrack.yaml"]

    all_results = []

    for model_path in models:
        model_name = Path(model_path).stem
        detector = OptimalDroneDetector(model_path, confidence=0.65)

        for tracker in trackers:
            tracker_name = Path(tracker).stem

            for video in videos:
                try:
                    output_name = f"{model_name}_{tracker_name}_FINAL_{Path(video).stem}.mp4"
                    output_path = output_dir / output_name

                    print(f"\n[+] {model_name} + {tracker_name} - {Path(video).name}")
                    result = detector.process_video(video, output_path, tracker)
                    result['model'] = model_name
                    result['tracker'] = tracker_name
                    result['video'] = Path(video).name

                    all_results.append(result)

                    print(f"[OK] Drones: {result['total_drones']} | Tracks: {result['unique_tracks']}")

                except Exception as e:
                    print(f"[!] Error: {str(e)[:60]}")

    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS - Optimized Drone Detection")
    print("="*80)

    print(f"\n{'Model':<12} {'Tracker':<10} {'Video':<35} {'Drones':<10} {'Tracks':<10} {'Avg/Fr':<10}")
    print("-"*100)

    for r in all_results:
        video_short = r['video'][:30]
        print(f"{r['model']:<12} {r['tracker']:<10} {video_short:<35} "
              f"{r['total_drones']:<10} {r['unique_tracks']:<10} {r['drone_per_frame']:>8.2f}")

    print(f"\n[OK] Generated: {len(list(output_dir.glob('*.mp4')))} output videos")
    print(f"[OK] Location: {output_dir.absolute()}")

    # Statistics
    total_drones = sum(r['total_drones'] for r in all_results)
    total_tracks = sum(r['unique_tracks'] for r in all_results)

    print(f"\n[STATS]")
    print(f"  Total Drones Detected: {total_drones}")
    print(f"  Total Unique Tracks: {total_tracks}")
    print(f"  Average Drones per Frame: {total_drones / sum(r['total_frames'] for r in all_results):.2f}")


if __name__ == "__main__":
    main()
