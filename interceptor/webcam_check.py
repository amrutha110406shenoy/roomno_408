"""Detection-only check: python webcam_check.py [weights] [camera_index]. Press q to quit."""
import sys

import cv2
from ultralytics import YOLO

model = YOLO(sys.argv[1] if len(sys.argv) > 1 else "best.pt")
cap = cv2.VideoCapture(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
while True:
    ok, frame = cap.read()
    if not ok:
        sys.exit("camera read failed")
    r = model(frame, imgsz=640, conf=0.4, verbose=False)[0]
    cv2.imshow("drone check (q = quit)", r.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
