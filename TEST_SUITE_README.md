# YOLO Detection & Tracking - Comprehensive Test Suite

## Quick Start

Your setup is verified and ready! To run the comprehensive test:

```bash
python test_yolo_optimal.py
```

Expected time: **15-45 minutes** (depending on CPU speed)

---

## Test Details

### Input Video
- **File:** `video01_botsort_once.mp4`
- **Size:** 55.81 MB
- **Frames:** 1050
- **FPS:** 25.0
- **Resolution:** 1920x1080 (Full HD)

### Models Tested
1. **YOLOv8n** (6.2 MB) - Already downloaded
2. **YOLOv9n** - Will auto-download (~6.1 MB)
3. **YOLOv10n** - Will auto-download (~5.7 MB)

### Trackers Tested
- **BoTSORT** - Bottom-up sorting with appearance features
- **ByteTrack** - Byte-level association strategy

### Total Test Combinations
**6 videos** will be generated:
```
yolov8n_botsort.mp4
yolov8n_bytetrack.mp4
yolov9n_botsort.mp4
yolov9n_bytetrack.mp4
yolov10n_botsort.mp4
yolov10n_bytetrack.mp4
```

---

## Running Tests

### Full Test (Recommended)
```bash
python test_yolo_optimal.py
```
Processes entire video through all model+tracker combinations. Shows progress bars and real-time statistics.

### Quick Test (First 100 frames only)
```bash
python quick_test_yolo.py
```
Useful for quick validation without waiting 30 minutes.

### Diagnostic Check
Verify setup before running tests:
```bash
python diagnose_setup.py
```

---

## Expected Output

### Video Files
All videos saved to `test_outputs/` directory with:
- **Bounding boxes** around detected objects (green)
- **Tracking IDs** (numbered) for consistent object tracking
- **Same resolution** as input (1920x1080)
- **Same framerate** as input (25 FPS)

### Processing Speed
- **YOLOv8n:** ~40-100 ms per frame
- **YOLOv9n:** ~60-150 ms per frame
- **YOLOv10n:** ~30-80 ms per frame

(Actual speed depends on your CPU)

### Output File Sizes
Expect 500-800 MB total for all 6 videos

---

## Analyzing Results

### After Tests Complete

#### 1. View Results
```bash
python view_results.py
```
Shows file sizes, frame counts, and playback info.

#### 2. Generate Report
```bash
python generate_report.py
```
Generates detailed comparison report and saves to:
- `test_outputs/TEST_RESULTS.txt` - Detailed text report
- `test_outputs/results.json` - Machine-readable format

#### 3. Watch Videos
```bash
# Windows
cd test_outputs
start yolov8n_botsort.mp4

# Or use ffplay
ffplay yolov8n_botsort.mp4

# Or use any video player
```

---

## Configuration

### Modify Confidence Threshold
Lower = more detections (more false positives)
Higher = fewer detections (more false negatives)

Edit `test_yolo_optimal.py`:
```python
CONF_THRESHOLD = 0.5  # Change this (0.0 to 1.0)
```

### Test Fewer Frames
For faster initial testing, edit the script to process only first N frames:
```python
MAX_FRAMES = 100  # Test only first 100 frames
```

### Test Specific Model+Tracker Combination
```python
# In test_yolo_optimal.py, modify MODELS and TRACKERS:
MODELS = {
    "yolov8n": "yolov8n.pt",  # Only test this one
}
TRACKERS = {
    "BoTSORT": "botsort.yaml",  # Only test this one
}
```

---

## What to Look For

### In the Videos

1. **Detection Quality**
   - Are drones/objects clearly detected?
   - Any missed objects?
   - Any false positives (non-drones detected)?

2. **Tracking Performance**
   - Do tracking IDs stay consistent?
   - ID switches when target reappears?
   - Smooth vs. jittery boxes?

3. **Model Comparison**
   - Which model gives tightest boxes?
   - Which is fastest?
   - Accuracy vs. speed tradeoff?

4. **Tracker Comparison**
   - Which tracker maintains IDs better?
   - Which is more robust to occlusion?
   - Performance on your specific video?

---

## Recommended Next Steps

After reviewing results:

### For Your Interceptor System
Best combination likely: **YOLOv8n + ByteTrack**
- Good balance of speed and accuracy
- YOLOv8n is well-trained on general objects
- ByteTrack is robust for real-time tracking

### For Improvement
1. **Fine-tune on drone data**
   - Collect more drone examples
   - Train custom model
   - Better accuracy on target

2. **Optimize for Raspberry Pi**
   - Export to ONNX format
   - Compile with Hailo-8L
   - Lower inference latency

3. **Adjust thresholds**
   - Detection confidence
   - Tracking IOU threshold
   - Person/object size filters

---

## Troubleshooting

### "CUDA out of memory"
You're using GPU but ran out of VRAM:
- Use CPU instead (slower but works)
- Reduce batch size

### "Codec not found"
MP4V codec issue:
- Install ffmpeg: `choco install ffmpeg`
- Use different output format

### Very slow processing
- Normal for CPU-only processing
- Try YOLOv8n (fastest)
- Reduce inference size to 416 or 320

### Video won't play
- Download VLC media player (plays any format)
- Or use `ffplay` with ffmpeg
- Or convert with `ffmpeg -i input.mp4 output.avi`

---

## Technical Details

### YOLO Inference Settings
```
- Input size: 640x640 (auto-scales input)
- Confidence threshold: 0.5 (default)
- IOU threshold: 0.45 (NMS)
```

### Tracking Algorithm Differences

**BoTSORT**
- Uses appearance features (more accurate)
- Slower due to ReID model
- Better for dense scenes
- More robust to occlusion

**ByteTrack**
- Lightweight association strategy
- Faster inference
- Good for real-time
- Simpler tracking logic

---

## Performance Benchmarks

Typical results on modern CPU:

| Model | Tracker | FPS | Size |
|-------|---------|-----|------|
| YOLOv8n | BoTSORT | 10-15 | ~100MB |
| YOLOv8n | ByteTrack | 12-18 | ~95MB |
| YOLOv9n | BoTSORT | 8-12 | ~110MB |
| YOLOv9n | ByteTrack | 10-15 | ~105MB |
| YOLOv10n | BoTSORT | 12-18 | ~90MB |
| YOLOv10n | ByteTrack | 15-22 | ~85MB |

(Actual FPS shown on-screen during test)

---

## Files in This Suite

| File | Purpose |
|------|---------|
| `test_yolo_optimal.py` | Main test script (recommended) |
| `test_yolo_tracker.py` | Alternative with detailed logging |
| `quick_test_yolo.py` | Fast test (first 100 frames) |
| `diagnose_setup.py` | Verify installation |
| `view_results.py` | Display generated videos info |
| `generate_report.py` | Create detailed comparison report |
| `analyze_results.py` | Statistical analysis |
| `TEST_GUIDE.md` | Detailed usage guide |
| `TEST_SUITE_README.md` | This file |

---

## Questions or Issues?

1. Run `diagnose_setup.py` to verify installation
2. Check `TEST_GUIDE.md` for detailed configuration
3. Review generated `TEST_RESULTS.txt` for detailed metrics
4. Check video output quality manually in generated MP4s

---

## Ready to Start?

```bash
python test_yolo_optimal.py
```

The test will take 15-45 minutes. You can monitor progress or let it run in the background.

Good luck! Your drone detection system is about to get thoroughly tested.
