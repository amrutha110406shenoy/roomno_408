"""
Deploy Fine-tuned Drone Detector
Use the trained model to detect ONLY drones (no kites, birds, etc)
"""
import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO

def deploy_on_videos(model_path, output_dir="drone_production_outputs"):
    """Deploy fine-tuned model on UAV videos"""

    print("\n" + "="*80)
    print("DEPLOYING FINE-TUNED DRONE DETECTOR")
    print("="*80)

    if not Path(model_path).exists():
        print(f"[!] Model not found: {model_path}")
        return

    model = YOLO(str(model_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    videos = [
        "datasets/istockphoto-1287277966-640_adpp_is.mp4",
        "datasets/istockphoto-2170041017-640_adpp_is.mp4",
        "datasets/istockphoto-2291118401-640_adpp_is.mp4",
        "datasets/istockphoto-2291460380-640_adpp_is.mp4",
    ]

    print(f"\n[+] Model: {Path(model_path).name}")
    print(f"[+] Processing {len(videos)} UAV videos")
    print(f"[+] Output: {output_dir.absolute()}\n")

    all_results = []

    for video_idx, video_path in enumerate(videos, 1):
        try:
            video_name = Path(video_path).stem
            output_name = f"production_{video_name}.mp4"
            output_path = output_dir / output_name

            print(f"\n[{video_idx}/{len(videos)}] {Path(video_path).name}")

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            frame_idx = 0
            drone_count = 0
            drone_frames = 0
            inference_times = []
            class_counts = {}

            print(f"  Video: {w}x{h} | {total_frames} frames | {fps:.1f}fps")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Detect ONLY drones with fine-tuned model
                t0 = time.time()
                results = model.predict(
                    frame, imgsz=640, conf=0.5, verbose=False
                )
                inference_times.append(time.time() - t0)

                boxes = results[0].boxes
                drone_count += len(boxes)

                if len(boxes) > 0:
                    drone_frames += 1

                # Annotate
                annotated = results[0].plot()

                # Count classes
                for box in boxes:
                    cls_id = int(box.cls.item())
                    class_name = model.names.get(cls_id, f"Class {cls_id}")
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                # Add stats
                cv2.putText(annotated, f"Frame: {frame_idx}/{total_frames}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(annotated, f"DRONES DETECTED: {len(boxes)}",
                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(annotated, f"Model: Fine-tuned YOLOv8n (PRODUCTION)",
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

                writer.write(annotated)

                if frame_idx % max(1, total_frames // 4) == 0:
                    avg_inf = np.mean(inference_times) * 1000
                    print(f"  Frame {frame_idx}/{total_frames} | "
                          f"Drones: {len(boxes)} | Inf: {avg_inf:.1f}ms")

            cap.release()
            writer.release()

            avg_inference = np.mean(inference_times) * 1000
            fps_actual = frame_idx / sum(inference_times) if inference_times else 0

            result = {
                'video': Path(video_path).name,
                'frames': frame_idx,
                'total_drones': drone_count,
                'drone_frames': drone_frames,
                'avg_inference_ms': avg_inference,
                'fps': fps_actual,
                'class_distribution': class_counts,
                'output': str(output_path)
            }
            all_results.append(result)

            avg_inf = np.mean(inference_times) * 1000
            print(f"  [OK] COMPLETE - {drone_count} drones, {avg_inf:.1f}ms avg inference")

        except Exception as e:
            print(f"  [!] Error: {str(e)[:60]}")

    return all_results, output_dir

def print_summary(results, output_dir):
    """Print deployment summary"""

    print("\n" + "="*80)
    print("DEPLOYMENT RESULTS - PRODUCTION DRONE DETECTOR")
    print("="*80)

    print(f"\n{'Video':<40} {'Drones':<12} {'Frames':<10} {'Avg Inf(ms)':<12}")
    print("-"*100)

    total_drones = 0
    for r in results:
        video_short = r['video'][:35]
        total_drones += r['total_drones']
        print(f"{video_short:<40} {r['total_drones']:<12} {r['drone_frames']:<10} {r['avg_inference_ms']:>10.1f}")

    print(f"\n[SUMMARY]")
    print(f"  Total Drones Detected: {total_drones}")
    print(f"  Output Videos: {len(list(output_dir.glob('*.mp4')))}")
    print(f"  Location: {output_dir.absolute()}")

    print(f"\n[SUCCESS] ✓ Production deployment complete!")
    print(f"[SUCCESS] ✓ Videos ready for use")

def main():
    # Path to fine-tuned model
    model_path = "drone_finetuned_models/yolov8n_drone_fast/weights/best.pt"

    # Deploy
    results, output_dir = deploy_on_videos(model_path)

    # Print summary
    if results:
        print_summary(results, output_dir)

if __name__ == "__main__":
    main()
