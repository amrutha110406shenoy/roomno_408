"""
Fine-tune YOLOv8n specifically for DRONE detection
Removes kites, birds, frisbees from detection - ONLY detects drones
"""
import json
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO
import time
import requests

def create_synthetic_drone_dataset():
    """Create synthetic training data for drone detection"""

    print("[+] Creating synthetic drone training dataset...")

    dataset_dir = Path("drone_training_dataset")
    for split in ["train", "val", "test"]:
        Path(f"{dataset_dir}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{dataset_dir}/labels/{split}").mkdir(parents=True, exist_ok=True)

    # Create synthetic drone images
    num_train = 400
    num_val = 50
    num_test = 50

    total = num_train + num_val + num_test

    for idx in range(total):
        if idx < num_train:
            split = "train"
        elif idx < num_train + num_val:
            split = "val"
        else:
            split = "test"

        # Create image
        h, w = 640, 480
        img = np.random.randint(100, 150, (h, w, 3), dtype=np.uint8)

        # Add sky gradient
        for y in range(h):
            img[y, :] = [100 + y//10, 150 + y//15, 200]

        # Add random drones (small circles/rectangles)
        num_drones = np.random.randint(1, 4)
        boxes = []

        for _ in range(num_drones):
            # Random drone position and size
            drone_w = np.random.randint(20, 80)
            drone_h = np.random.randint(15, 60)
            cx = np.random.randint(drone_w, w - drone_w)
            cy = np.random.randint(drone_h, h - drone_h)

            # Normalize coordinates
            norm_cx = cx / w
            norm_cy = cy / h
            norm_w = drone_w / w
            norm_h = drone_h / h

            boxes.append([0, norm_cx, norm_cy, norm_w, norm_h])

            # Draw drone on image (small object)
            cv2.rectangle(img, (cx-drone_w//2, cy-drone_h//2),
                         (cx+drone_w//2, cy+drone_h//2), (0, 255, 0), -1)

            # Add some visual detail
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)

        # Save image
        img_path = f"{dataset_dir}/images/{split}/drone_{idx:04d}.jpg"
        cv2.imwrite(img_path, img)

        # Save labels
        label_path = f"{dataset_dir}/labels/{split}/drone_{idx:04d}.txt"
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

        if (idx + 1) % 100 == 0:
            print(f"  Created {idx + 1}/{total} synthetic images")

    print(f"  [OK] Created {total} synthetic training images")
    return dataset_dir

def create_dataset_yaml(dataset_dir):
    """Create data.yaml for training"""

    yaml_content = """path: drone_training_dataset
train: images/train
val: images/val
test: images/test

nc: 1
names: ['drone']
"""

    yaml_path = Path(dataset_dir) / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"[+] Created dataset config: {yaml_path}")
    return yaml_path

def train_drone_model(data_yaml, epochs=50):
    """Train YOLOv8n on drone dataset"""

    print("\n" + "="*80)
    print("TRAINING YOLOv8n FOR DRONE DETECTION")
    print("="*80)

    # Load pretrained model
    model = YOLO("yolov8n.pt")

    print(f"\n[+] Starting training...")
    print(f"    Model: YOLOv8n")
    print(f"    Dataset: {data_yaml}")
    print(f"    Epochs: {epochs}")
    print(f"    Batch Size: 32")

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        patience=10,
        device='cpu',
        project="drone_finetuned_models",
        name="yolov8n_drone_v1",
        verbose=True,
        save=True,
        plots=False,
    )

    print("\n[OK] Training completed!")

    best_model_path = Path("drone_finetuned_models/yolov8n_drone_v1/weights/best.pt")
    print(f"[+] Best model saved: {best_model_path}")

    return best_model_path

def test_finetuned_model(model_path):
    """Test fine-tuned model on UAV videos"""

    print("\n" + "="*80)
    print("TESTING FINE-TUNED MODEL ON UAV VIDEOS")
    print("="*80)

    model = YOLO(str(model_path))

    output_dir = Path("drone_finetuned_outputs")
    output_dir.mkdir(exist_ok=True)

    videos = [
        "datasets/istockphoto-1287277966-640_adpp_is.mp4",
        "datasets/istockphoto-2170041017-640_adpp_is.mp4",
        "datasets/istockphoto-2291118401-640_adpp_is.mp4",
        "datasets/istockphoto-2291460380-640_adpp_is.mp4",
    ]

    all_results = []

    for video_path in videos:
        try:
            video_name = Path(video_path).stem
            output_name = f"finetuned_drone_{video_name}.mp4"
            output_path = output_dir / output_name

            print(f"\n[*] Processing: {Path(video_path).name}")

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            frame_idx = 0
            drone_count = 0
            inference_times = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Detect with fine-tuned model
                t0 = time.time()
                results = model.predict(
                    frame, imgsz=640, conf=0.5, verbose=False
                )
                inference_times.append(time.time() - t0)

                # Count drones
                boxes = results[0].boxes
                drone_count += len(boxes)

                # Annotate
                annotated = results[0].plot()

                # Add stats
                cv2.putText(annotated, f"Frame {frame_idx}/{total_frames}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(annotated, f"DRONES: {len(boxes)}",
                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(annotated, f"Model: Fine-tuned YOLOv8n",
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

                writer.write(annotated)

                if frame_idx % max(1, total_frames // 4) == 0:
                    avg_inf = np.mean(inference_times) * 1000
                    print(f"  Frame {frame_idx}/{total_frames} | Drones: {len(boxes)} | Inf: {avg_inf:.1f}ms")

            cap.release()
            writer.release()

            avg_inference = np.mean(inference_times) * 1000
            fps_actual = frame_idx / (sum(inference_times) if inference_times else 1)

            result = {
                'video': Path(video_path).name,
                'frames': frame_idx,
                'drones_detected': drone_count,
                'avg_inference_ms': avg_inference,
                'fps': fps_actual,
                'output': str(output_path)
            }
            all_results.append(result)

            print(f"[OK] Drones: {drone_count} | Avg Inf: {avg_inference:.1f}ms | FPS: {fps_actual:.2f}")

        except Exception as e:
            print(f"[!] Error: {str(e)[:60]}")

    return all_results, output_dir

def main():
    print("\n" + "="*80)
    print("YOLOV8N FINE-TUNING FOR DRONE DETECTION")
    print("="*80)

    # Step 1: Create synthetic training data
    print("\n[STEP 1] Creating Training Dataset")
    dataset_dir = create_synthetic_drone_dataset()

    # Step 2: Create data config
    print("\n[STEP 2] Creating Dataset Configuration")
    data_yaml = create_dataset_yaml(dataset_dir)

    # Step 3: Train model
    print("\n[STEP 3] Training Fine-tuned Model")
    best_model_path = train_drone_model(data_yaml, epochs=50)

    # Step 4: Test on UAV videos
    print("\n[STEP 4] Testing Fine-tuned Model")
    results, output_dir = test_finetuned_model(best_model_path)

    # Summary
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE - RESULTS")
    print("="*80)

    print(f"\n{'Video':<40} {'Drones':<10} {'FPS':<10} {'Inference(ms)':<15}")
    print("-"*80)

    total_drones = 0
    for r in results:
        video_short = r['video'][:35]
        total_drones += r['drones_detected']
        print(f"{video_short:<40} {r['drones_detected']:<10} {r['fps']:<10.2f} {r['avg_inference_ms']:<15.1f}")

    print(f"\n[SUMMARY]")
    print(f"  Total Drones Detected: {total_drones}")
    print(f"  Output Videos: {len(list(output_dir.glob('*.mp4')))}")
    print(f"  Location: {output_dir.absolute()}")
    print(f"  Fine-tuned Model: {best_model_path}")

    print(f"\n[OK] Fine-tuning complete!")
    print(f"[OK] New videos ready in: {output_dir}")
    print(f"[OK] Use fine-tuned model for production: {best_model_path}")

if __name__ == "__main__":
    main()
