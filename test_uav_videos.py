"""
Test yolo8n and yolo10n on 4 fast-moving UAV videos
"""
import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO
from collections import defaultdict

# Configuration
UAV_VIDEOS = [
    "datasets/istockphoto-1287277966-640_adpp_is.mp4",
    "datasets/istockphoto-2170041017-640_adpp_is.mp4",
    "datasets/istockphoto-2291118401-640_adpp_is.mp4",
    "datasets/istockphoto-2291460380-640_adpp_is.mp4",
]

OUTPUT_DIR = "uav_detection_outputs"
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


def test_model_on_video(model_name, model_path, tracker, input_video, output_path):
    """Test model on UAV video"""
    tracker_name = TRACKER_NAMES[tracker]

    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_idx = 0
    total_dets = 0
    class_counts = defaultdict(int)
    conf_scores = []
    track_ids = set()
    inference_times = []
    start_time = time.time()

    print(f"\n  Processing: {Path(input_video).name}")
    print(f"  Video: {w}x{h}, {fps:.1f}fps, {total_frames} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Inference with tracker
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

        # Annotate and write
        annotated = results[0].plot()
        writer.write(annotated)

        # Progress
        if frame_idx % max(1, total_frames // 4) == 0:
            avg_inf = np.mean(inference_times) * 1000
            print(f"    Frame {frame_idx}/{total_frames} | Dets: {len(boxes)} | Avg Inf: {avg_inf:.1f}ms")

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    fps_actual = frame_idx / total_time if total_time > 0 else 0

    class_names = model.names

    return {
        'model': model_name,
        'tracker': tracker_name,
        'video': Path(input_video).name,
        'frames': frame_idx,
        'total_detections': total_dets,
        'avg_det_per_frame': total_dets / frame_idx if frame_idx > 0 else 0,
        'total_time': total_time,
        'fps_actual': fps_actual,
        'avg_inference_ms': np.mean(inference_times) * 1000 if inference_times else 0,
        'unique_tracks': len(track_ids),
        'class_distribution': dict(class_counts),
        'output': str(output_path),
        'class_names': class_names,
        'avg_confidence': np.mean(conf_scores) if conf_scores else 0,
    }


def main():
    print("\n" + "="*100)
    print("🚁 UAV VIDEO DETECTION TEST - YOLOv8n & YOLOv10n")
    print("="*100)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    results = []
    total_combos = len(UAV_VIDEOS) * len(MODELS) * len(TRACKERS)
    current = 0

    print(f"\nTesting {len(UAV_VIDEOS)} UAV videos with {len(MODELS)} models and {len(TRACKERS)} trackers")
    print(f"Total combinations: {total_combos}")

    print(f"\n{'Video':<40} {'Model':<10} {'Tracker':<10} {'FPS':<8} {'Dets':<8} {'Tracks':<8}")
    print("─"*100)

    # Test each combination
    for video_path in UAV_VIDEOS:
        video_name = Path(video_path).name
        video_short = video_name[:25] + "..." if len(video_name) > 25 else video_name

        for model_name, model_path in MODELS.items():
            for tracker in TRACKERS:
                current += 1
                tracker_name = TRACKER_NAMES[tracker]

                try:
                    output_name = f"{model_name}_{tracker_name.lower()}_{Path(video_path).stem}.mp4"
                    output_path = Path(OUTPUT_DIR) / output_name

                    result = test_model_on_video(model_name, model_path, tracker, video_path, output_path)

                    print(f"{video_short:<40} {model_name:<10} {tracker_name:<10} "
                          f"{result['fps_actual']:>6.2f}   {result['total_detections']:>6}   {result['unique_tracks']:>6}")

                    results.append(result)

                except Exception as e:
                    print(f"{video_short:<40} {model_name:<10} {tracker_name:<10} "
                          f"✗ FAIL: {str(e)[:30]}")

    # Print summary
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)

    # Group by video
    for video_path in UAV_VIDEOS:
        video_name = Path(video_path).name
        video_results = [r for r in results if r['video'] == video_name]

        if video_results:
            print(f"\n📹 {video_name}")
            print("─"*100)
            print(f"{'Model':<12} {'Tracker':<12} {'Frames':<10} {'Detections':<12} {'FPS':<10} {'Inf(ms)':<12} {'Tracks':<10}")
            print("─"*100)

            for r in video_results:
                print(f"{r['model']:<12} {r['tracker']:<12} {r['frames']:<10} {r['total_detections']:<12} "
                      f"{r['fps_actual']:>8.2f}   {r['avg_inference_ms']:>10.1f}   {r['unique_tracks']:<10}")

    # Detailed metrics
    print("\n" + "="*100)
    print("DETAILED METRICS BY VIDEO")
    print("="*100)

    for video_path in UAV_VIDEOS:
        video_name = Path(video_path).name
        video_results = [r for r in results if r['video'] == video_name]

        if video_results:
            print(f"\n📊 {video_name}")
            print("─"*100)

            for r in video_results:
                print(f"\n  {r['model']} + {r['tracker']}:")
                print(f"    Total Detections:        {r['total_detections']}")
                print(f"    Avg Detections/Frame:    {r['avg_det_per_frame']:.2f}")
                print(f"    Unique Tracked Objects:  {r['unique_tracks']}")
                print(f"    Performance:")
                print(f"      FPS:                   {r['fps_actual']:.2f}")
                print(f"      Avg Inference:         {r['avg_inference_ms']:.2f}ms/frame")
                if r['total_detections'] > 0:
                    print(f"      Avg Confidence:        {r['avg_confidence']:.4f}")
                    if r['class_distribution']:
                        print(f"    Classes Detected:")
                        for cls_id, count in sorted(r['class_distribution'].items()):
                            cls_name = r['class_names'].get(int(cls_id), f"Class {cls_id}")
                            pct = (count / r['total_detections'] * 100)
                            print(f"      {cls_name:<20} {count:>5} ({pct:>5.1f}%)")

    # Output videos list
    print("\n" + "="*100)
    print("GENERATED OUTPUT VIDEOS")
    print("="*100)
    print(f"\n📁 Location: {Path(OUTPUT_DIR).absolute()}\n")

    output_videos = sorted([v for v in Path(OUTPUT_DIR).glob("*.mp4")])
    print(f"{'Filename':<55} {'Size':<12} {'Model':<10} {'Tracker':<12}")
    print("─"*100)

    for video_file in output_videos:
        size_mb = video_file.stat().st_size / (1024*1024)

        # Parse filename
        name_parts = video_file.stem.split('_')
        model = name_parts[0] if len(name_parts) > 0 else "?"
        tracker = name_parts[1] if len(name_parts) > 1 else "?"

        print(f"{video_file.name:<55} {size_mb:>10.1f}MB   {model:<10} {tracker:<12}")

    print("\n" + "="*100)
    print(f"✅ Testing Complete! Total Videos Generated: {len(output_videos)}")
    print("="*100)

    # Save detailed report
    report_path = Path(OUTPUT_DIR) / "uav_test_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("UAV VIDEO DETECTION TEST REPORT\n")
        f.write("="*100 + "\n")
        f.write(f"Models: {', '.join(MODELS.keys())}\n")
        f.write(f"Trackers: {', '.join(TRACKER_NAMES.values())}\n")
        f.write(f"Videos Tested: {len(UAV_VIDEOS)}\n\n")

        for video_path in UAV_VIDEOS:
            video_name = Path(video_path).name
            video_results = [r for r in results if r['video'] == video_name]

            f.write(f"\n{'='*100}\n{video_name}\n{'='*100}\n")

            for r in video_results:
                f.write(f"\n{r['model']} + {r['tracker']}:\n")
                f.write(f"  Frames: {r['frames']} | Detections: {r['total_detections']} | ")
                f.write(f"Avg/Frame: {r['avg_det_per_frame']:.2f} | FPS: {r['fps_actual']:.2f}\n")
                f.write(f"  Inference: {r['avg_inference_ms']:.1f}ms | Unique Tracks: {r['unique_tracks']}\n")
                f.write(f"  Output: {Path(r['output']).name}\n")

    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
