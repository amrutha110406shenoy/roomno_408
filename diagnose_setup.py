"""
Diagnostic script to verify YOLO setup and video processing capability
"""
import cv2
import sys
import numpy as np
from pathlib import Path

# Fix encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("YOLO SETUP DIAGNOSTIC")
print("=" * 80)

# Check Python version
print("\n1. Python version: {}".format(sys.version.split()[0]))

# Check OpenCV
print("\n2. OpenCV:")
print("   Version: {}".format(cv2.__version__))

# Check if input video exists
input_video = Path("video01_botsort_once.mp4")
if input_video.exists():
    print("\n3. Input video:")
    print("   [OK] Found: {}".format(input_video))
    print("   Size: {:.2f}MB".format(input_video.stat().st_size / (1024**2)))
else:
    print("\n3. Input video:")
    print("   [ERR] NOT FOUND: {}".format(input_video))
    sys.exit(1)

# Test video reading
print("\n4. Video reading test:")
try:
    cap = cv2.VideoCapture(str(input_video))
    if cap.isOpened():
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("   [OK] Video opened successfully")
        print("   Frames: {}".format(frames))
        print("   FPS: {:.1f}".format(fps))
        print("   Resolution: {}x{}".format(w, h))

        # Try reading a frame
        ret, frame = cap.read()
        if ret:
            print("   [OK] Can read frames")
            print("   Frame shape: {}".format(frame.shape))
        else:
            print("   [ERR] Cannot read first frame")

        cap.release()
    else:
        print("   [ERR] Cannot open video")
        sys.exit(1)
except Exception as e:
    print("   [ERR] Error: {}".format(e))
    sys.exit(1)

# Check Ultralytics
print("\n5. Ultralytics YOLO:")
try:
    from ultralytics import YOLO
    print("   [OK] Ultralytics imported successfully")

    # Check if models exist or can be downloaded
    models_to_check = ["yolov8n.pt", "yolov9n.pt", "yolov10n.pt"]
    for model_name in models_to_check:
        path = Path(model_name)
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            print("   [OK] {} exists locally ({:.1f}MB)".format(model_name, size_mb))
        else:
            print("   [INFO] {} not found locally (will download on first use)".format(model_name))

except ImportError as e:
    print("   [ERR] Cannot import YOLO: {}".format(e))
    sys.exit(1)

# Check video writer codec
print("\n6. Video writer codec test:")
try:
    # Try to write a test frame
    output_file = "test_codec.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (640, 480))

    if out.isOpened():
        # Create a simple test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out.write(test_frame)
        out.release()

        # Check if file was created
        test_path = Path(output_file)
        if test_path.exists() and test_path.stat().st_size > 0:
            test_path.unlink()  # Delete test file
            print("   [OK] MP4V codec available")
        else:
            print("   [WARN] MP4V codec may not be working properly")
    else:
        print("   [WARN] Cannot open VideoWriter")

except Exception as e:
    print("   [WARN] Error testing codec: {}".format(e))

# Check tracker files
print("\n7. Tracker configuration:")
trackers = ["botsort.yaml", "bytetrack.yaml"]
for tracker in trackers:
    path = Path(tracker)
    if path.exists():
        print("   [OK] {} found".format(tracker))
    else:
        # These are usually downloaded by ultralytics
        print("   [INFO] {} will be auto-downloaded by ultralytics".format(tracker))

# Check output directory
print("\n8. Output directory:")
output_dir = Path("test_outputs")
try:
    output_dir.mkdir(exist_ok=True)
    print("   [OK] Output directory ready: {}".format(output_dir.absolute()))
except Exception as e:
    print("   [ERR] Cannot create output directory: {}".format(e))

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print("\nSetup appears to be ready for testing!")
print("Run: python test_yolo_optimal.py")
