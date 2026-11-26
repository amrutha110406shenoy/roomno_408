"""
FAST Fine-tuning - Optimized for speed
Trains in 15-20 minutes on CPU
"""
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO
import time

def create_minimal_drone_dataset():
    """Create minimal but effective drone training data"""

    print("[+] Creating minimal drone training dataset (200 images)...")

    dataset_dir = Path("drone_minimal_dataset")
    for split in ["train", "val"]:
        Path(f"{dataset_dir}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{dataset_dir}/labels/{split}").mkdir(parents=True, exist_ok=True)

    num_train = 160
    num_val = 40

    for idx in range(num_train + num_val):
        split = "train" if idx < num_train else "val"

        # Create image
        h, w = 640, 480
        img = np.random.randint(100, 150, (h, w, 3), dtype=np.uint8)

        # Sky gradient
        for y in range(h):
            img[y, :] = [100 + y//10, 150 + y//15, 200]

        # Add drones
        num_drones = np.random.randint(1, 3)
        boxes = []

        for _ in range(num_drones):
            drone_w = np.random.randint(20, 80)
            drone_h = np.random.randint(15, 60)
            cx = np.random.randint(drone_w, w - drone_w)
            cy = np.random.randint(drone_h, h - drone_h)

            norm_cx = cx / w
            norm_cy = cy / h
            norm_w = drone_w / w
            norm_h = drone_h / h

            boxes.append([0, norm_cx, norm_cy, norm_w, norm_h])

            # Draw drone
            cv2.rectangle(img, (cx-drone_w//2, cy-drone_h//2),
                         (cx+drone_w//2, cy+drone_h//2), (0, 255, 0), -1)
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)

        # Save
        img_path = f"{dataset_dir}/images/{split}/drone_{idx:03d}.jpg"
        cv2.imwrite(img_path, img)

        label_path = f"{dataset_dir}/labels/{split}/drone_{idx:03d}.txt"
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

        if (idx + 1) % 50 == 0:
            print(f"  Created {idx + 1}/{num_train + num_val} images")

    print(f"  [OK] Dataset ready: {num_train} train, {num_val} val")
    return dataset_dir

def create_yaml(dataset_dir):
    """Create data.yaml"""

    yaml_content = """path: drone_minimal_dataset
train: images/train
val: images/val

nc: 1
names: ['drone']
"""

    yaml_path = Path(dataset_dir) / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    return yaml_path

def train_fast(data_yaml):
    """Quick training - 25 epochs"""

    print("\n" + "="*80)
    print("FAST FINE-TUNING (25 EPOCHS)")
    print("="*80)

    model = YOLO("yolov8n.pt")

    print("\n[+] Training YOLOv8n...")
    results = model.train(
        data=str(data_yaml),
        epochs=25,
        imgsz=640,
        batch=8,
        patience=5,
        device='cpu',
        project="drone_finetuned_models",
        name="yolov8n_drone_fast",
        verbose=False,
        save=True,
        plots=False,
    )

    best_model = Path("drone_finetuned_models/yolov8n_drone_fast/weights/best.pt")
    print(f"\n[OK] Training complete!")
    print(f"[OK] Model: {best_model}")

    return best_model

def test_model(model_path):
    """Test on UAV videos"""

    print("\n" + "="*80)
    print("TESTING FINE-TUNED MODEL")
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
            output_name = f"finetuned_{video_name}.mp4"
            output_path = output_dir / output_name

            print(f"\n[*] {Path(video_path).name}")

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

                # Detect
                t0 = time.time()
                results = model.predict(
                    frame, imgsz=640, conf=0.4, verbose=False
                )
                inference_times.append(time.time() - t0)

                boxes = results[0].boxes
                drone_count += len(boxes)

                # Annotate
                annotated = results[0].plot()

                cv2.putText(annotated, f"Frame {frame_idx}/{total_frames}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(annotated, f"DRONES: {len(boxes)}",
                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(annotated, "Fine-tuned YOLOv8n",
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                writer.write(annotated)

                if frame_idx % max(1, total_frames // 4) == 0:
                    print(f"  Frame {frame_idx}/{total_frames}")

            cap.release()
            writer.release()

            avg_inf = np.mean(inference_times) * 1000
            fps_actual = frame_idx / sum(inference_times) if inference_times else 0

            all_results.append({
                'video': Path(video_path).name,
                'drones': drone_count,
                'avg_inf_ms': avg_inf,
                'output': str(output_path)
            })

            print(f"  [OK] Drones: {drone_count} | Inference: {avg_inf:.1f}ms")

        except Exception as e:
            print(f"  [!] Error: {str(e)[:50]}")

    return all_results, output_dir

def main():
    print("\n" + "="*80)
    print("FAST FINE-TUNING FOR DRONE DETECTION")
    print("="*80)

    # Create dataset
    print("\n[STEP 1] Creating Dataset")
    dataset = create_minimal_drone_dataset()
    yaml_path = create_yaml(dataset)

    # Train
    print("\n[STEP 2] Training Model")
    model_path = train_fast(yaml_path)

    # Test
    print("\n[STEP 3] Testing Model")
    results, output_dir = test_model(model_path)

    # Summary
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE")
    print("="*80)

    total_drones = sum(r['drones'] for r in results)

    print(f"\n[RESULTS]")
    print(f"{'Video':<40} {'Drones':<10} {'Output':<40}")
    print("-"*90)

    for r in results:
        video_short = r['video'][:35]
        output_short = Path(r['output']).name[:35]
        print(f"{video_short:<40} {r['drones']:<10} {output_short:<40}")

    print(f"\n[SUMMARY]")
    print(f"  Total Drones Detected: {total_drones}")
    print(f"  Output Videos: {len(list(output_dir.glob('*.mp4')))}")
    print(f"  Location: {output_dir.absolute()}")
    print(f"  Fine-tuned Model: {model_path}")

    print(f"\n[OK] Fine-tuning complete!")
    print(f"[OK] Use fine-tuned model: {model_path}")

if __name__ == "__main__":
    main()
