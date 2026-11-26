"""
Analyze YOLO8n detection results with detailed metrics and readings
"""
import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import time

INPUT_VIDEO = "video01_botsort_once.mp4"
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.5

class YOLOAnalyzer:
    def __init__(self, model_path, tracker, tracker_name):
        self.model = YOLO(model_path)
        self.tracker = tracker
        self.tracker_name = tracker_name
        self.reset_stats()

    def reset_stats(self):
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts = defaultdict(int)
        self.confidence_scores = []
        self.track_ids = set()
        self.frame_detections = []
        self.inference_times = []

    def analyze_video(self, video_path):
        """Process video and collect detailed metrics"""
        print(f"\n{'='*80}")
        print(f"Analyzing with {self.tracker_name} Tracker")
        print(f"{'='*80}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_input = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n📹 Video Info:")
        print(f"  Resolution: {w}x{h}")
        print(f"  FPS: {fps_input:.2f}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps_input:.2f}s")

        self.reset_stats()
        start_total = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # Run inference with tracker
            start_inference = time.time()
            results = self.model.track(
                frame,
                imgsz=640,
                conf=CONF_THRESHOLD,
                tracker=self.tracker,
                persist=True,
                verbose=False
            )
            inference_time = time.time() - start_inference
            self.inference_times.append(inference_time)

            # Collect detection data
            boxes = results[0].boxes
            frame_det_count = len(boxes)
            self.total_detections += frame_det_count
            self.frame_detections.append(frame_det_count)

            # Collect class and confidence data
            if boxes is not None and len(boxes) > 0:
                classes = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                track_ids = boxes.id

                for cls, conf in zip(classes, confs):
                    cls_idx = int(cls)
                    self.class_counts[cls_idx] += 1
                    self.confidence_scores.append(float(conf))

                if track_ids is not None:
                    self.track_ids.update(track_ids.cpu().numpy())

            if self.frame_count % 150 == 0:
                print(f"  Processed {self.frame_count}/{total_frames} frames...")

        cap.release()
        total_time = time.time() - start_total

        return self.generate_report(total_time, total_frames)

    def generate_report(self, total_time, total_frames):
        """Generate detailed metrics report"""
        report = []
        report.append(f"\n{'='*80}")
        report.append(f"📊 YOLO8n {self.tracker_name} Detection Report")
        report.append(f"{'='*80}")

        # Frame statistics
        report.append(f"\n📈 Frame Statistics:")
        report.append(f"  Total Frames Processed: {self.frame_count}")
        report.append(f"  Total Detections: {self.total_detections}")
        report.append(f"  Avg Detections/Frame: {self.total_detections/self.frame_count:.2f}")
        report.append(f"  Max Detections/Frame: {max(self.frame_detections) if self.frame_detections else 0}")
        report.append(f"  Min Detections/Frame: {min(self.frame_detections) if self.frame_detections else 0}")
        report.append(f"  Frames with Detections: {sum(1 for x in self.frame_detections if x > 0)}")

        # Inference performance
        avg_inference = np.mean(self.inference_times) if self.inference_times else 0
        min_inference = np.min(self.inference_times) if self.inference_times else 0
        max_inference = np.max(self.inference_times) if self.inference_times else 0
        fps_actual = self.frame_count / total_time

        report.append(f"\n⚡ Performance Metrics:")
        report.append(f"  Total Processing Time: {total_time:.2f}s")
        report.append(f"  Actual FPS: {fps_actual:.2f}")
        report.append(f"  Avg Inference/Frame: {avg_inference*1000:.2f}ms")
        report.append(f"  Min Inference/Frame: {min_inference*1000:.2f}ms")
        report.append(f"  Max Inference/Frame: {max_inference*1000:.2f}ms")

        # Confidence statistics
        if self.confidence_scores:
            report.append(f"\n🎯 Confidence Statistics:")
            report.append(f"  Avg Confidence: {np.mean(self.confidence_scores):.4f}")
            report.append(f"  Min Confidence: {np.min(self.confidence_scores):.4f}")
            report.append(f"  Max Confidence: {np.max(self.confidence_scores):.4f}")
            report.append(f"  Std Dev: {np.std(self.confidence_scores):.4f}")

        # Class distribution
        if self.class_counts:
            report.append(f"\n🏷️  Class Distribution:")
            class_names = self.model.names
            for cls_idx in sorted(self.class_counts.keys()):
                count = self.class_counts[cls_idx]
                class_name = class_names.get(cls_idx, f"Class {cls_idx}")
                percentage = (count / self.total_detections * 100) if self.total_detections > 0 else 0
                report.append(f"  {class_name:<25} {count:>5} detections ({percentage:>5.1f}%)")

        # Tracking statistics
        report.append(f"\n🔗 Tracking Statistics ({self.tracker_name}):")
        report.append(f"  Unique Track IDs: {len(self.track_ids)}")

        return "\n".join(report)


def main():
    print("\n" + "="*80)
    print("🚀 YOLO8n Detailed Analysis - Both Trackers")
    print("="*80)

    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    trackers = [
        ("botsort.yaml", "BoTSORT"),
        ("bytetrack.yaml", "ByteTrack")
    ]

    all_reports = []

    for tracker_path, tracker_name in trackers:
        analyzer = YOLOAnalyzer(MODEL_PATH, tracker_path, tracker_name)
        report = analyzer.analyze_video(INPUT_VIDEO)
        all_reports.append(report)
        print(report)

    # Save combined report
    combined_report = "\n\n".join(all_reports)
    report_path = "yolo8n_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(combined_report)

    print(f"\n✅ Report saved to: {report_path}")


if __name__ == "__main__":
    main()
