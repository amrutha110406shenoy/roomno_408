"""
Compare all models on drone dataset vs original demo video
"""
import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO
from collections import defaultdict

OUTPUT_DIR = "drone_detection_outputs"
CONF_THRESHOLD = 0.5

MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolov10n": "yolov10n.pt",
}

TRACKERS = ["botsort.yaml", "bytetrack.yaml"]

TRACKER_NAMES = {
    "botsort.yaml": "BoTSORT",
    "bytetrack.yaml": "ByteTrack"
}


def test_on_video(model_name, model_path, tracker, video_path, is_drone=False):
    """Test model on video and return metrics"""
    tracker_name = TRACKER_NAMES[tracker]

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video
    output_name = f"{model_name}_{tracker_name.lower()}_{'drone' if is_drone else 'demo'}.mp4"
    output_path = Path(OUTPUT_DIR) / output_name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_idx = 0
    total_dets = 0
    class_counts = defaultdict(int)
    conf_scores = []
    track_ids = set()
    inference_times = []
    start_time = time.time()

    max_frames = 150 if is_drone else total_frames  # Limit drone to 150 frames

    print(f"\n  Processing {video_path}...")
    print(f"  Video: {w}x{h}, {fps:.1f}fps, {min(max_frames, total_frames)} frames")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Inference
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

        # Collect stats
        boxes = results[0].boxes
        total_dets += len(boxes)

        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id

            for cls, conf in zip(classes, confs):
                class_counts[int(cls)] += 1
                conf_scores.append(float(conf))

            if ids is not None:
                track_ids.update(ids.cpu().numpy())

        # Annotate
        annotated = results[0].plot()
        writer.write(annotated)

        if frame_idx % 50 == 0:
            avg_inf = np.mean(inference_times) * 1000
            print(f"    Frame {frame_idx}/{min(max_frames, total_frames)} | Dets: {len(boxes)} | Inf: {avg_inf:.1f}ms")

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    fps_actual = frame_idx / total_time if total_time > 0 else 0

    class_names = model.names

    return {
        'model': model_name,
        'tracker': tracker_name,
        'dataset': 'Drone' if is_drone else 'Demo',
        'frames': frame_idx,
        'total_detections': total_dets,
        'avg_det_per_frame': total_dets / frame_idx if frame_idx > 0 else 0,
        'total_time': total_time,
        'fps_actual': fps_actual,
        'avg_inference_ms': np.mean(inference_times) * 1000 if inference_times else 0,
        'min_inference_ms': np.min(inference_times) * 1000 if inference_times else 0,
        'max_inference_ms': np.max(inference_times) * 1000 if inference_times else 0,
        'avg_confidence': np.mean(conf_scores) if conf_scores else 0,
        'min_confidence': np.min(conf_scores) if conf_scores else 0,
        'max_confidence': np.max(conf_scores) if conf_scores else 0,
        'unique_tracks': len(track_ids),
        'class_distribution': dict(class_counts),
        'output': str(output_path),
        'class_names': class_names
    }


def print_report(results):
    """Print detailed report"""
    print("\n" + "="*100)
    print("COMPREHENSIVE TEST REPORT - Drone Detection Dataset vs Demo Video")
    print("="*100)

    # Group by dataset
    drone_results = [r for r in results if r['dataset'] == 'Drone']
    demo_results = [r for r in results if r['dataset'] == 'Demo']

    if drone_results:
        print("\n" + "─"*100)
        print("DRONE DETECTION DATASET RESULTS")
        print("─"*100)
        print(f"\n{'Model':<12} {'Tracker':<12} {'Frames':<10} {'Dets':<10} {'FPS':<10} {'Inf(ms)':<12} {'Tracks':<10}")
        print("─"*100)

        for r in drone_results:
            print(f"{r['model']:<12} {r['tracker']:<12} {r['frames']:<10} {r['total_detections']:<10} "
                  f"{r['fps_actual']:>8.2f}   {r['avg_inference_ms']:>10.1f}   {r['unique_tracks']:<10}")

    if demo_results:
        print("\n" + "─"*100)
        print("ORIGINAL DEMO VIDEO RESULTS")
        print("─"*100)
        print(f"\n{'Model':<12} {'Tracker':<12} {'Frames':<10} {'Dets':<10} {'FPS':<10} {'Inf(ms)':<12} {'Tracks':<10}")
        print("─"*100)

        for r in demo_results:
            print(f"{r['model']:<12} {r['tracker']:<12} {r['frames']:<10} {r['total_detections']:<10} "
                  f"{r['fps_actual']:>8.2f}   {r['avg_inference_ms']:>10.1f}   {r['unique_tracks']:<10}")

    # Detailed stats
    print("\n" + "="*100)
    print("DETAILED STATISTICS")
    print("="*100)

    for r in results:
        print(f"\n📊 {r['model'].upper()} + {r['tracker']} - {r['dataset']} Dataset")
        print("─"*100)
        print(f"  Frames Processed:        {r['frames']}")
        print(f"  Total Detections:        {r['total_detections']}")
        print(f"  Avg Detections/Frame:    {r['avg_det_per_frame']:.2f}")
        print(f"  Unique Tracks:           {r['unique_tracks']}")
        print(f"\n  Performance:")
        print(f"    Total Time:            {r['total_time']:.2f}s")
        print(f"    Actual FPS:            {r['fps_actual']:.2f}")
        print(f"    Avg Inference/Frame:   {r['avg_inference_ms']:.2f}ms")
        print(f"    Min Inference/Frame:   {r['min_inference_ms']:.2f}ms")
        print(f"    Max Inference/Frame:   {r['max_inference_ms']:.2f}ms")

        if r['total_detections'] > 0:
            print(f"\n  Detection Quality:")
            print(f"    Avg Confidence:        {r['avg_confidence']:.4f}")
            print(f"    Min Confidence:        {r['min_confidence']:.4f}")
            print(f"    Max Confidence:        {r['max_confidence']:.4f}")
            print(f"\n  Classes Detected:")
            for cls_id, count in sorted(r['class_distribution'].items()):
                cls_name = r['class_names'].get(int(cls_id), f"Class {cls_id}")
                pct = (count / r['total_detections'] * 100)
                print(f"    {cls_name:<20} {count:>5} ({pct:>5.1f}%)")
        else:
            print(f"\n  ⚠️  No detections found")

    print("\n" + "="*100)


def main():
    print("\n" + "="*100)
    print("🚁 COMPREHENSIVE YOLO MODEL COMPARISON")
    print("="*100)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    results = []

    # Test on drone dataset
    print("\n📂 Testing on Drone Detection Dataset...")
    for model_name, model_path in MODELS.items():
        for tracker in TRACKERS:
            try:
                result = test_on_video(model_name, model_path, tracker,
                                      "drone_detection_outputs/test_video_drone.mp4", is_drone=True)
                results.append(result)
            except Exception as e:
                print(f"  ✗ {model_name} + {TRACKER_NAMES[tracker]}: {str(e)[:40]}")

    # Test on demo video
    print("\n📹 Testing on Original Demo Video...")
    for model_name, model_path in MODELS.items():
        for tracker in TRACKERS:
            try:
                result = test_on_video(model_name, model_path, tracker,
                                      "video01_botsort_once.mp4", is_drone=False)
                results.append(result)
            except Exception as e:
                print(f"  ✗ {model_name} + {TRACKER_NAMES[tracker]}: {str(e)[:40]}")

    # Print report
    print_report(results)

    # Save report
    report_path = Path(OUTPUT_DIR) / "comprehensive_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("YOLO DRONE DETECTION COMPREHENSIVE TEST REPORT\n")
        f.write("="*100 + "\n\n")
        f.write(f"Models Tested: {', '.join(MODELS.keys())}\n")
        f.write(f"Trackers: {', '.join(TRACKER_NAMES.values())}\n")
        f.write(f"Datasets: Drone Detection (synthetic), Demo Video (airplanes)\n\n")

        for r in results:
            f.write(f"\n{r['model']} + {r['tracker']} - {r['dataset']}\n")
            f.write(f"  Frames: {r['frames']} | Dets: {r['total_detections']} | FPS: {r['fps_actual']:.2f}\n")
            f.write(f"  Inference: {r['avg_inference_ms']:.1f}ms/frame | Tracks: {r['unique_tracks']}\n")

    print(f"\n✅ Report saved to: {report_path}")
    print(f"\n📁 Generated Videos in: {Path(OUTPUT_DIR).absolute()}")

    # List all videos
    print("\nOutput Videos:")
    for video in sorted(Path(OUTPUT_DIR).glob("*.mp4")):
        if video.name != "test_video_drone.mp4":
            size_mb = video.stat().st_size / (1024*1024)
            print(f"  ✓ {video.name:<50} {size_mb:>6.1f}MB")


if __name__ == "__main__":
    main()
