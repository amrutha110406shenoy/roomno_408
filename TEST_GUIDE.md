# YOLO Detection & Tracking - Comprehensive Test Suite

## Overview
This test suite evaluates YOLOv8, v9, and v10 models with two different tracking algorithms (BoTSORT and ByteTrack) on your drone detection video.

## What Gets Tested

### Models
- **YOLOv8n** - Latest stable YOLO model
- **YOLOv9n** - Improved version with enhanced architecture
- **YOLOv10n** - Newest generation with optimizations

### Trackers
- **BoTSORT** - Bottom-up tracking with appearance features
- **ByteTrack** - Byte-level association for robust tracking

### Total Tests
- 6 video outputs (3 models × 2 trackers)
- Each output includes bounding boxes and tracking IDs

## Test Scripts

### 1. `test_yolo_optimal.py` ⭐ RECOMMENDED
**Best for**: Complete testing with real-time progress feedback

```bash
python test_yolo_optimal.py
```

**Features:**
- Real-time progress bars for each combination
- Comprehensive summary report
- Error handling and diagnostics
- Clear output organization

**Output:** `test_outputs/` directory with 6 MP4 files

### 2. `test_yolo_tracker.py`
**Alternative version** with detailed frame-by-frame logging

```bash
python test_yolo_tracker.py
```

### 3. `quick_test_yolo.py`
**Testing only first 100 frames** (for quick validation)

```bash
python quick_test_yolo.py
```

## Usage

### Run Full Test
```bash
python test_yolo_optimal.py
```

Expected time: ~15-30 minutes depending on your CPU

### View Results
After testing completes:

```bash
# Option 1: List all generated videos
python view_results.py

# Option 2: Analyze efficiency metrics
python analyze_results.py

# Option 3: Play videos (Windows)
cd test_outputs
start yolov8n_botsort.mp4
# or
ffplay yolov8n_botsort.mp4
```

## Output Files

Each combination generates an MP4 video file:
- `yolov8n_botsort.mp4` - YOLOv8n with BoTSORT
- `yolov8n_bytetrack.mp4` - YOLOv8n with ByteTrack
- `yolov9n_botsort.mp4` - YOLOv9n with BoTSORT
- `yolov9n_bytetrack.mp4` - YOLOv9n with ByteTrack
- `yolov10n_botsort.mp4` - YOLOv10n with BoTSORT
- `yolov10n_bytetrack.mp4` - YOLOv10n with ByteTrack

### Output Format
- **Green bounding boxes** = Detected objects
- **Tracking IDs** = Persistent object identifiers
- **Resolution** = Same as input (1920x1080 in demo)
- **Frame rate** = Same as input video

## Configuration

### Confidence Threshold
Default: `0.5` (50%)

To change, edit the CONF_THRESHOLD in the test script:
```python
CONF_THRESHOLD = 0.5  # 0.0 to 1.0
```

### Inference Size
Default: `640x640` YOLO inference

Models will auto-resize input frames to this size for detection.

## Performance Tips

### Speed Up Testing
1. **Use smaller resolution input**
   - Reduce input video resolution before testing
   
2. **Test fewer frames**
   - Edit the test script to process only first N frames
   - Useful for quick validation

3. **GPU Acceleration**
   - Ensure CUDA is configured for PyTorch
   - GPU processing is 10-50x faster than CPU

### Monitor Progress
During testing:
```bash
# In another terminal, check output files growing
dir test_outputs\*.mp4
```

## Expected Results

### Processing Speed (CPU, typical)
- **YOLOv8n**: ~50-100ms per frame
- **YOLOv9n**: ~60-120ms per frame  
- **YOLOv10n**: ~40-80ms per frame

### Accuracy (on drone detection)
- YOLOv8n: Good baseline, trained on COCO
- YOLOv9n: Better generalization
- YOLOv10n: Best performance-accuracy tradeoff

### Tracking Performance
- **BoTSORT**: Better for dense scenes, more computations
- **ByteTrack**: Faster, good for real-time, more robust

## Troubleshooting

### "Model not found" error
The script will auto-download models on first run. This requires:
- Internet connection
- ~500MB free disk space

### Video codec issues
Ensure `ffmpeg` is installed:
```bash
# Windows
choco install ffmpeg
# or download from ffmpeg.org
```

### Out of memory
If you run out of memory:
1. Close other applications
2. Reduce YOLO inference size to 416 or 320
3. Process video in sections

### Progress seems slow
- Check CPU usage in Task Manager
- If <50% CPU: Python may be waiting for disk I/O
- If >90% CPU: Processing is normal, be patient

## Advanced: Custom Testing

### Test with your own video
```python
# In test script, change:
INPUT_VIDEO = "your_video.mp4"
```

### Test with custom models
```python
MODELS = {
    "my-custom": "path/to/best.pt",
}
```

### Change tracker parameters
Edit the tracker YAML files or modify the track() call:
```python
results = model.track(
    frame,
    imgsz=640,
    conf=0.5,
    tracker="botsort.yaml",  # Change this
    tracker_conf=0.5,  # Tracker confidence
    tracker_iou=0.1,   # Tracker IoU threshold
    persist=True,      # Persist IDs between frames
    verbose=False
)
```

## Results Interpretation

### Bounding Box Quality
- **Tight boxes** = Good object localization
- **Missed detections** = False negatives (reduce conf threshold)
- **Extra boxes** = False positives (increase conf threshold)

### Tracking IDs
- **Stable IDs** = Smooth tracking
- **ID switches** = Tracking loss (try different tracker)
- **Multiple IDs for same object** = Fragmented tracking

### Comparison
1. **Model comparison**: Same tracker, different models
   - Better boxes = Better model
   - Faster = Preferred for real-time

2. **Tracker comparison**: Same model, different trackers
   - Fewer ID switches = Better tracker
   - More consistent IDs = More robust

## Next Steps

After reviewing results:

1. **For production**
   - Choose best model+tracker combination
   - Fine-tune on your specific data if needed
   - Deploy to Raspberry Pi with Hailo acceleration

2. **For optimization**
   - Test with quantized models (.onnx, .pt → .tflite)
   - Benchmark on actual target hardware
   - Profile performance bottlenecks

3. **For improvement**
   - Train custom model on more drone samples
   - Fine-tune tracker parameters
   - Add post-processing filters

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [BoTSORT Paper](https://arxiv.org/abs/2211.14734)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [YOLOv8 Guide](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv9 Guide](https://docs.ultralytics.com/models/yolov9/)
- [YOLOv10 Guide](https://docs.ultralytics.com/models/yolov10/)

## Questions?

Review the generated videos and check:
- Are objects being detected reliably?
- Are tracking IDs maintained across frames?
- Is performance acceptable for your use case?

You can also modify threshold and inference parameters to optimize for your specific scenario.
