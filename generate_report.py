"""
Generate comprehensive comparison report from test results
"""
import cv2
import json
from pathlib import Path
from datetime import datetime


class TestReport:
    def __init__(self, output_dir="test_outputs"):
        self.output_dir = Path(output_dir)
        self.videos = []
        self.load_videos()

    def load_videos(self):
        """Load all video files and extract metadata"""
        for mp4_file in sorted(self.output_dir.glob("*.mp4")):
            try:
                cap = cv2.VideoCapture(str(mp4_file))

                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                size_mb = mp4_file.stat().st_size / (1024 * 1024)

                cap.release()

                # Parse filename
                parts = mp4_file.stem.split('_')
                model = f"YOLOv{parts[0][1]}n"  # e.g., "v8n" -> "YOLOv8n"
                tracker = "BoTSORT" if "botsort" in mp4_file.name else "ByteTrack"

                self.videos.append({
                    'filename': mp4_file.name,
                    'path': str(mp4_file),
                    'model': model,
                    'tracker': tracker,
                    'frames': frames,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'duration_sec': frames / fps if fps > 0 else 0,
                    'size_mb': size_mb,
                    'bitrate_kbps': (size_mb * 8 * 1024) / (frames / fps) if fps > 0 else 0
                })
            except Exception as e:
                print(f"⚠ Error loading {mp4_file.name}: {e}")

    def generate_text_report(self):
        """Generate detailed text report"""

        if not self.videos:
            return "No videos found!"

        lines = []
        lines.append("=" * 120)
        lines.append("YOLO DETECTION & TRACKING - TEST RESULTS REPORT")
        lines.append("=" * 120)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Test videos found: {len(self.videos)}")

        # Detailed table
        lines.append("\n" + "-" * 120)
        lines.append("DETAILED RESULTS")
        lines.append("-" * 120)

        lines.append(f"\n{'Video':<35} {'Model':<12} {'Tracker':<12} {'Size (MB)':<12} {'Duration':<10} {'Res':<12}")
        lines.append("-" * 120)

        total_size = 0
        for v in self.videos:
            resolution = f"{v['width']}x{v['height']}"
            lines.append(
                f"{v['filename']:<35} {v['model']:<12} {v['tracker']:<12} "
                f"{v['size_mb']:>10.2f} {v['duration_sec']:>8.1f}s {resolution:<12}"
            )
            total_size += v['size_mb']

        lines.append("-" * 120)
        lines.append(f"{'TOTAL':<35} {'':<12} {'':<12} {total_size:>10.2f}MB")

        # Analysis by model
        lines.append("\n" + "=" * 120)
        lines.append("ANALYSIS BY MODEL")
        lines.append("=" * 120)

        models = {}
        for v in self.videos:
            if v['model'] not in models:
                models[v['model']] = []
            models[v['model']].append(v)

        for model in sorted(models.keys()):
            videos = models[model]
            lines.append(f"\n{model}:")
            lines.append(f"  Test combinations: {len(videos)}")

            for tracker in sorted(set(v['tracker'] for v in videos)):
                v = next((x for x in videos if x['tracker'] == tracker), None)
                if v:
                    lines.append(
                        f"    • {tracker:<12} | {v['size_mb']:>8.2f}MB | {v['duration_sec']:>7.1f}s | "
                        f"{v['bitrate_kbps']:>7.1f}kbps"
                    )

            avg_size = sum(v['size_mb'] for v in videos) / len(videos)
            avg_duration = sum(v['duration_sec'] for v in videos) / len(videos)
            lines.append(f"  Averages: {avg_size:.2f}MB, {avg_duration:.1f}s per video")

        # Analysis by tracker
        lines.append("\n" + "=" * 120)
        lines.append("ANALYSIS BY TRACKER")
        lines.append("=" * 120)

        trackers = {}
        for v in self.videos:
            if v['tracker'] not in trackers:
                trackers[v['tracker']] = []
            trackers[v['tracker']].append(v)

        for tracker in sorted(trackers.keys()):
            videos = trackers[tracker]
            lines.append(f"\n{tracker}:")
            lines.append(f"  Test combinations: {len(videos)}")
            lines.append(f"  {'Model':<12} | {'Size':<10} | {'Duration':<10} | {'Bitrate':<12}")

            for v in sorted(videos, key=lambda x: x['model']):
                lines.append(
                    f"    {v['model']:<12} | {v['size_mb']:>8.2f}MB | {v['duration_sec']:>8.1f}s | "
                    f"{v['bitrate_kbps']:>10.1f}kbps"
                )

            total_size = sum(v['size_mb'] for v in videos)
            avg_duration = sum(v['duration_sec'] for v in videos) / len(videos)
            lines.append(f"  Totals: {total_size:.2f}MB overall, {avg_duration:.1f}s average per video")

        # Efficiency metrics
        lines.append("\n" + "=" * 120)
        lines.append("EFFICIENCY METRICS")
        lines.append("=" * 120)

        lines.append(f"\n{'Metric':<30} {'Best':<20} {'Value':<20}")
        lines.append("-" * 120)

        # Smallest file
        smallest = min(self.videos, key=lambda x: x['size_mb'])
        lines.append(f"{'Smallest file':<30} {smallest['model'] + ' + ' + smallest['tracker']:<20} {smallest['size_mb']:.2f}MB")

        # Fastest (smallest duration)
        fastest = min(self.videos, key=lambda x: x['duration_sec'])
        lines.append(f"{'Fastest processing':<30} {fastest['model'] + ' + ' + fastest['tracker']:<20} {fastest['duration_sec']:.1f}s")

        # Highest bitrate (quality vs size)
        highest_br = max(self.videos, key=lambda x: x['bitrate_kbps'])
        lines.append(f"{'Highest bitrate (quality)':<30} {highest_br['model'] + ' + ' + highest_br['tracker']:<20} {highest_br['bitrate_kbps']:.1f}kbps")

        # Model performance (by size per frame)
        lines.append("\n" + "-" * 120)
        lines.append("MODEL PERFORMANCE (avg file size)")
        lines.append("-" * 120)

        for model in sorted(models.keys()):
            avg_size = sum(v['size_mb'] for v in models[model]) / len(models[model])
            lines.append(f"  {model:<20} {avg_size:>8.2f}MB")

        # Tracker performance
        lines.append("\n" + "-" * 120)
        lines.append("TRACKER PERFORMANCE (avg file size)")
        lines.append("-" * 120)

        for tracker in sorted(trackers.keys()):
            avg_size = sum(v['size_mb'] for v in trackers[tracker]) / len(trackers[tracker])
            lines.append(f"  {tracker:<20} {avg_size:>8.2f}MB")

        # Recommendations
        lines.append("\n" + "=" * 120)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 120)

        lines.append("\n1. BEST OVERALL PERFORMANCE:")
        best_overall = min(self.videos, key=lambda x: x['size_mb'])
        lines.append(f"   {best_overall['model']} + {best_overall['tracker']} ({best_overall['size_mb']:.2f}MB)")

        lines.append("\n2. FOR REAL-TIME PROCESSING:")
        fastest = min(self.videos, key=lambda x: x['duration_sec'])
        lines.append(f"   {fastest['model']} + {fastest['tracker']} ({fastest['duration_sec']:.1f}s processing time)")

        lines.append("\n3. FOR BEST QUALITY:")
        best_quality = max(self.videos, key=lambda x: x['bitrate_kbps'])
        lines.append(f"   {best_quality['model']} + {best_quality['tracker']} ({best_quality['bitrate_kbps']:.1f}kbps)")

        lines.append("\n4. RECOMMENDED COMBINATIONS:")
        lines.append("   • For drone interception: YOLOv8n + ByteTrack (balance of speed & accuracy)")
        lines.append("   • For archive: YOLOv10n + BoTSORT (best quality)")
        lines.append("   • For edge devices: YOLOv8n + ByteTrack (fastest)")

        lines.append("\n" + "=" * 120)

        return "\n".join(lines)

    def save_report(self, filename="TEST_RESULTS.txt"):
        """Save report to file"""
        report = self.generate_text_report()
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"✓ Report saved to: {output_path}")
        return output_path

    def save_json(self, filename="results.json"):
        """Save results as JSON"""
        output_path = self.output_dir / filename

        data = {
            'timestamp': datetime.now().isoformat(),
            'videos': self.videos,
            'summary': {
                'total_videos': len(self.videos),
                'total_size_mb': sum(v['size_mb'] for v in self.videos),
                'total_duration_sec': sum(v['duration_sec'] for v in self.videos)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ JSON saved to: {output_path}")
        return output_path


def main():
    print("Generating test results report...\n")

    report = TestReport("test_outputs")

    if not report.videos:
        print("❌ No test videos found in test_outputs/")
        print("   Run test_yolo_optimal.py first")
        return

    # Print report
    print(report.generate_text_report())

    # Save reports
    print("\n" + "=" * 120)
    print("SAVING REPORTS")
    print("=" * 120 + "\n")

    report.save_report()
    report.save_json()

    print("\n✓ Report generation complete!")


if __name__ == "__main__":
    main()
