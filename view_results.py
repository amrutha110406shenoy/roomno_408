"""
View and compare generated output videos
"""
import cv2
import os
from pathlib import Path
from dataclasses import dataclass
import subprocess

@dataclass
class VideoInfo:
    path: str
    name: str
    size_mb: float
    frames: int
    fps: float
    duration: float
    width: int
    height: int


def get_video_info(video_path):
    """Extract video information"""
    cap = cv2.VideoCapture(str(video_path))

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration = frames / fps if fps > 0 else 0
    size_mb = Path(video_path).stat().st_size / (1024 * 1024)

    return VideoInfo(
        path=str(video_path),
        name=Path(video_path).name,
        size_mb=size_mb,
        frames=frames,
        fps=fps,
        duration=duration,
        width=width,
        height=height
    )


def print_comparison_table(videos):
    """Print comparison table"""
    if not videos:
        print("❌ No output videos found")
        return

    # Group by model and tracker
    by_model = {}
    for v in videos:
        parts = v.name.replace('.mp4', '').split('_')
        model = parts[0]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(v)

    print("\n" + "=" * 120)
    print("OUTPUT VIDEOS - DETAILED COMPARISON")
    print("=" * 120)

    print(f"\n{'Video File':<35} {'Size (MB)':<12} {'Frames':<10} {'FPS':<8} {'Duration':<10} {'Resolution':<15}")
    print("-" * 120)

    for v in sorted(videos, key=lambda x: x.name):
        resolution = f"{v.width}x{v.height}"
        print(f"{v.name:<35} {v.size_mb:>10.2f} {v.frames:>8} {v.fps:>6.1f} {v.duration:>8.1f}s {resolution:<15}")

    # Statistics by model
    print("\n" + "=" * 120)
    print("STATISTICS BY MODEL")
    print("=" * 120)

    for model in sorted(by_model.keys()):
        videos_for_model = by_model[model]
        print(f"\n{model.upper()} - {len(videos_for_model)} videos:")

        total_size = sum(v.size_mb for v in videos_for_model)
        avg_duration = sum(v.duration for v in videos_for_model) / len(videos_for_model)

        for v in sorted(videos_for_model, key=lambda x: x.name):
            tracker = v.name.replace(f"{model}_", "").replace(".mp4", "")
            print(f"  • {tracker:<20} {v.size_mb:>8.2f}MB | {v.duration:>7.1f}s | {v.frames:>5} frames @ {v.fps:>5.1f}fps")

        print(f"  Total: {total_size:.2f}MB | Avg duration: {avg_duration:.1f}s")


def print_playback_commands(videos):
    """Print commands to play each video"""
    print("\n" + "=" * 120)
    print("PLAYBACK COMMANDS (Windows)")
    print("=" * 120)

    for v in sorted(videos, key=lambda x: x.name):
        print(f"\nTo play {v.name}:")
        print(f"  cd test_outputs && ffplay {v.name}")
        # Alternative Windows media player
        print(f"  OR: start {v.name}")


def analyze_efficiency():
    """Compare efficiency metrics"""
    print("\n" + "=" * 120)
    print("EFFICIENCY COMPARISON")
    print("=" * 120)

    output_dir = Path("test_outputs")
    videos = []

    for video_path in output_dir.glob("*.mp4"):
        try:
            info = get_video_info(video_path)
            videos.append(info)
        except:
            pass

    if not videos:
        print("❌ No videos to analyze")
        return

    # Calculate metrics per tracker
    trackers = {}
    for v in videos:
        tracker = v.name.split('_')[1].replace('.mp4', '')
        if tracker not in trackers:
            trackers[tracker] = []
        trackers[tracker].append(v)

    print(f"\n{'Tracker':<20} {'Avg Size (MB)':<15} {'Avg Duration (s)':<18} {'Avg Frames':<15}")
    print("-" * 120)

    for tracker in sorted(trackers.keys()):
        videos_for_tracker = trackers[tracker]
        avg_size = sum(v.size_mb for v in videos_for_tracker) / len(videos_for_tracker)
        avg_duration = sum(v.duration for v in videos_for_tracker) / len(videos_for_tracker)
        avg_frames = sum(v.frames for v in videos_for_tracker) / len(videos_for_tracker)

        print(f"{tracker:<20} {avg_size:>13.2f} {avg_duration:>16.1f} {avg_frames:>13.0f}")


def main():
    output_dir = Path("test_outputs")

    if not output_dir.exists():
        print("❌ No test_outputs directory found. Run test_yolo_tracker.py first.")
        return

    print("=" * 120)
    print("YOLO DETECTION & TRACKING - TEST RESULTS VIEWER")
    print("=" * 120)

    # Find all MP4 files
    video_files = sorted(output_dir.glob("*.mp4"))

    if not video_files:
        print("❌ No output videos found in test_outputs/")
        return

    print(f"\n✓ Found {len(video_files)} output video(s)\n")

    # Get info for all videos
    videos = []
    for video_path in video_files:
        try:
            info = get_video_info(video_path)
            videos.append(info)
            print(f"✓ Loaded: {info.name} ({info.size_mb:.2f}MB)")
        except Exception as e:
            print(f"✗ Failed to load {video_path.name}: {e}")

    if videos:
        print_comparison_table(videos)
        print_playback_commands(videos)
        analyze_efficiency()

    print("\n" + "=" * 120)
    print("✓ All test outputs are ready in: test_outputs/")
    print("=" * 120)


if __name__ == "__main__":
    main()
