"""
Prepare drone detection dataset from NDJSON for YOLO training
"""
import json
import cv2
import numpy as np
from pathlib import Path
import requests
from urllib.parse import urlparse
import os
from concurrent.futures import ThreadPoolExecutor
import time

NDJSON_FILE = "datasets/drone-detection-yolo.ndjson"
DATASET_DIR = "drone_dataset_yolo"
IMAGE_DIR = "drone_dataset_yolo/images"
LABEL_DIR = "drone_dataset_yolo/labels"

def setup_dataset_dirs():
    """Create dataset directories"""
    Path(f"{IMAGE_DIR}/train").mkdir(parents=True, exist_ok=True)
    Path(f"{IMAGE_DIR}/val").mkdir(parents=True, exist_ok=True)
    Path(f"{IMAGE_DIR}/test").mkdir(parents=True, exist_ok=True)
    Path(f"{LABEL_DIR}/train").mkdir(parents=True, exist_ok=True)
    Path(f"{LABEL_DIR}/val").mkdir(parents=True, exist_ok=True)
    Path(f"{LABEL_DIR}/test").mkdir(parents=True, exist_ok=True)

def download_image(url, save_path, timeout=10):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        pass
    return False

def process_dataset(ndjson_path, max_images=3000):
    """Process NDJSON and prepare training data"""
    print("[+] Processing drone detection dataset...")

    setup_dataset_dirs()

    entries = []
    with open(ndjson_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            data = json.loads(line)
            if data.get('type') == 'image':
                entries.append(data)
                if len(entries) >= max_images:
                    break

    print(f"  Loaded {len(entries)} image entries")

    # Count by split
    splits = {}
    for entry in entries:
        split = entry.get('split', 'test')
        splits[split] = splits.get(split, 0) + 1
    print(f"  Distribution: {splits}")

    # Download images and create labels
    downloaded = 0
    failed = 0

    print(f"\n[*] Downloading images...")

    for idx, entry in enumerate(entries):
        try:
            url = entry.get('url')
            if not url:
                continue

            split = entry.get('split', 'test')
            filename = entry.get('file', f"image_{idx}")

            # Ensure filename has correct extension
            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                filename += '.jpg'

            image_path = Path(IMAGE_DIR) / split / filename
            label_path = Path(LABEL_DIR) / split / filename.rsplit('.', 1)[0] + '.txt'

            # Download image
            if not image_path.exists():
                if download_image(url, image_path):
                    downloaded += 1
                else:
                    failed += 1
                    continue

            # Create YOLO format label
            boxes = entry.get('annotations', {}).get('boxes', [])

            if boxes:
                with open(label_path, 'w') as f:
                    for box in boxes:
                        if len(box) >= 5:
                            # Format: class_id, center_x, center_y, width, height (normalized)
                            # In YOLO format: class_id x_center y_center width height
                            class_id = int(box[0])
                            cx, cy, w, h = box[1], box[2], box[3], box[4]
                            f.write(f"{class_id} {cx} {cy} {w} {h}\n")
            else:
                # Empty label for images without drones
                label_path.touch()

            if (idx + 1) % 100 == 0:
                print(f"  Progress: {idx + 1}/{len(entries)} | Downloaded: {downloaded}")

        except Exception as e:
            failed += 1
            continue

    print(f"\n  OK Downloaded: {downloaded} images")
    print(f"  FAILED: {failed} images")

    return downloaded

def create_dataset_yaml():
    """Create YOLO dataset.yaml configuration"""
    yaml_content = """path: drone_dataset_yolo
train: images/train
val: images/val
test: images/test

nc: 1
names: ['drone']
"""

    yaml_path = Path(DATASET_DIR) / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n[+] Created dataset config: {yaml_path}")
    return yaml_path

def main():
    print("\n" + "="*80)
    print("PREPARE DRONE DETECTION DATASET FOR YOLO TRAINING")
    print("="*80)

    # Process dataset
    downloaded = process_dataset(NDJSON_FILE, max_images=2000)

    # Create YAML config
    create_dataset_yaml()

    # Count files
    train_images = len(list(Path(f"{IMAGE_DIR}/train").glob("*")))
    val_images = len(list(Path(f"{IMAGE_DIR}/val").glob("*")))
    test_images = len(list(Path(f"{IMAGE_DIR}/test").glob("*")))

    print(f"\nDataset Summary:")
    print(f"  Training images: {train_images}")
    print(f"  Validation images: {val_images}")
    print(f"  Test images: {test_images}")
    print(f"  Total: {train_images + val_images + test_images}")

    print(f"\nOK Dataset prepared at: {Path(DATASET_DIR).absolute()}")

if __name__ == "__main__":
    main()
