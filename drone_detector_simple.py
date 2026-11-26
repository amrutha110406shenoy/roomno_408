"""
Simple yet effective drone detector - Remove false positives, keep drones
"""
import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO

def detect_drones_on_video(model_name, model_path, tracker, video_path, output_path):
    """Detect only drones in video"""

    # Drone-like classes from COCO
    DRONE_INDICATORS = {
        'bird': 0.7,
        'airplane': 0.75,
        'frisbee': 0.65,
        'kite': 0.70,
    }

    model = YOLO(model_path)
    class_names = model.names

    # Find drone class indices
    drone_class_ids = []
    for cls_name, threshold in DRONE_INDICATORS.items():
        for cls_id, model_cls_name in class_names.items():
            if cls_name.lower() in model_cls_name.lower():
                drone_class_ids.append((cls_id, threshold))
                break

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_idx = 0
    drone_count = 0
    track_set = set()
    inference_times = []

    print(f"\n[*] {model_name} + {Path(tracker).stem}")
    print(f"[*] {Path(video_path).name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Detect
        t0 = time.time()
        results = model.track(
            frame, imgsz=640, conf=0.4,
            tracker=tracker, persist=True, verbose=False
        )
        inference_times.append(time.time() - t0)

        # Filter for drones only
        annotated = frame.copy()
        drone_detections = 0

        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            # Check if this is a drone-like class
            is_drone = False
            min_conf = 0.5

            for drone_id, drone_conf in drone_class_ids:
                if cls_id == drone_id and conf >= drone_conf:
                    is_drone = True
                    break

            if is_drone:
                drone_detections += 1
                drone_count += 1

                # Draw drone detection
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_names[cls_id]} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Track ID
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id.item())
                    track_set.add(track_id)

        # Stats
        cv2.putText(annotated, f"Frame {frame_idx}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"DRONES: {drone_detections}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        writer.write(annotated)

        if frame_idx % max(1, total_frames // 4) == 0:
            print(f"  Frame {frame_idx}/{total_frames} | Drones: {drone_detections}")

    cap.release()
    writer.release()

    return {
        'model': model_name,
        'tracker': Path(tracker).stem,
        'video': Path(video_path).name,
        'frames': frame_idx,
        'total_drones': drone_count,
        'unique_tracks': len(track_set),
        'avg_inference_ms': np.mean(inference_times) * 1000,
        'output': str(output_path)
    }


def main():
    print("="*80)
    print("DRONE DETECTOR - Filtered for Precision")
    print("="*80)

    output_dir = Path("drone_outputs")
    output_dir.mkdir(exist_ok=True)

    videos = [
        "datasets/istockphoto-1287277966-640_adpp_is.mp4",
        "datasets/istockphoto-2170041017-640_adpp_is.mp4",
        "datasets/istockphoto-2291118401-640_adpp_is.mp4",
        "datasets/istockphoto-2291460380-640_adpp_is.mp4",
    ]

    models = ["yolov8n.pt", "yolov10n.pt"]
    trackers = ["botsort.yaml", "bytetrack.yaml"]

    all_results = []

    for model in models:
        model_name = Path(model).stem
        for tracker in trackers:
            for video in videos:
                try:
                    output_name = f"{model_name}_{Path(tracker).stem}_drone_{Path(video).stem}.mp4"
                    output_path = output_dir / output_name

                    result = detect_drones_on_video(model_name, model, tracker, video, output_path)
                    all_results.append(result)

                except Exception as e:
                    print(f"[!] Error: {str(e)[:60]}")

    # Summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print(f"\n{'Model':<12} {'Tracker':<10} {'Video':<35} {'Drones':<10} {'Tracks':<10} {'Inf(ms)':<10}")
    print("-"*100)

    for r in all_results:
        video_short = r['video'][:30]
        print(f"{r['model']:<12} {r['tracker']:<10} {video_short:<35} "
              f"{r['total_drones']:<10} {r['unique_tracks']:<10} {r['avg_inference_ms']:>8.1f}")

    print(f"\n[+] Total output videos: {len(list(output_dir.glob('*.mp4')))}")
    print(f"[+] Location: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
