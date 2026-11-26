"""
Fine-tune YOLOv8n for drone detection
"""
from ultralytics import YOLO
from pathlib import Path

def train_drone_model():
    """Train YOLOv8n on drone detection dataset"""

    print("\n" + "="*80)
    print("TRAINING YOLOV8N FOR DRONE DETECTION")
    print("="*80)

    # Load model
    model = YOLO("yolov8n.pt")

    # Train
    print("\n[+] Starting training...")
    results = model.train(
        data="drone_dataset_yolo/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        device=0,
        project="drone_models",
        name="yolov8n_drone",
        verbose=True,
        save=True,
        plots=True,
        conf=0.5,
    )

    print("\n[+] Training complete!")
    print(f"[+] Best model saved at: {results.save_dir}")

    return results

def validate_model(model_path):
    """Validate trained model"""
    print("\n" + "="*80)
    print("VALIDATING TRAINED DRONE MODEL")
    print("="*80)

    model = YOLO(model_path)

    metrics = model.val(data="drone_dataset_yolo/data.yaml")

    print("\n[+] Validation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    return metrics

if __name__ == "__main__":
    # Train
    results = train_drone_model()

    # Find best model
    best_model = Path("drone_models/yolov8n_drone/weights/best.pt")

    if best_model.exists():
        print(f"\n[+] Best model found at: {best_model}")
        metrics = validate_model(str(best_model))
    else:
        print("[!] Best model not found!")
