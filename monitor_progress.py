"""
Monitor test progress in real-time
"""
import os
import time
from pathlib import Path
from datetime import datetime

def monitor():
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    expected_videos = [
        "yolov8n_botsort.mp4",
        "yolov8n_bytetrack.mp4",
        "yolov9n_botsort.mp4",
        "yolov9n_bytetrack.mp4",
        "yolov10n_botsort.mp4",
        "yolov10n_bytetrack.mp4",
    ]

    print("=" * 100)
    print("MONITORING TEST PROGRESS")
    print("=" * 100)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Expected {len(expected_videos)} videos to generate\n")

    completed = set()

    while True:
        # Check which videos exist
        completed.clear()
        for video in expected_videos:
            path = output_dir / video
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                completed.add(video)
                status = "✓ DONE" if size_mb > 1 else "⟳ WRITING"
                print(f"  [{len(completed)}/{len(expected_videos)}] {video:<30} {size_mb:>8.2f}MB {status}")

        # Clear console
        print("\n" + "=" * 100)

        if len(completed) == len(expected_videos):
            print("✓ ALL TESTS COMPLETED!")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            break

        remaining = set(expected_videos) - completed
        print(f"Waiting for {len(remaining)} videos...")
        for video in remaining:
            print(f"  ⏳ {video}")

        print("=" * 100)
        time.sleep(10)


if __name__ == "__main__":
    monitor()
