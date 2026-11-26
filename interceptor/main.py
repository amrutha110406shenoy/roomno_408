"""Onboard interceptor (Raspberry Pi 5 + AI HAT+ 13 TOPS + Pixhawk/ArduPilot).
Camera -> Hailo YOLOv8n (640x640) -> tracker -> GUIDED body-velocity commands.
Streams 1080p30 H.264 with boxes to the operator and records it to the SD card.

States: IDLE -LAUNCH-> TAKEOFF -> SEARCH -lock-> TRACK -CONFIRM-> STRIKE
        ABORT / operator link lost / search timeout -> RTL
"""
import argparse
import glob
import itertools
import os
import socket
import time

import cv2
from pymavlink import mavutil

from logic import CommandAuth, Tracker, V_SEARCH, clamp, guidance, wrap180

ap = argparse.ArgumentParser()
ap.add_argument("--gcs-ip", required=True, help="operator laptop IP (127.0.0.1 for laptop test)")
ap.add_argument("--hef", default="drone_yolov8n.hef", help="Pi: Hailo model")
ap.add_argument("--source", help="LAPTOP TEST: webcam index, video file or image folder (instead of Pi camera)")
ap.add_argument("--model", default="best.pt", help="laptop test: Ultralytics weights")
ap.add_argument("--mav", help="default /dev/ttyAMA0 on Pi, tcp:127.0.0.1:5760 (SITL) with --source")
ap.add_argument("--baud", type=int, default=921600)
ap.add_argument("--key", default="secret.key")
ap.add_argument("--alt", type=float, default=15.0, help="takeoff altitude m")
ap.add_argument("--conf", type=float, default=0.5)
args = ap.parse_args()
args.mav = args.mav or ("tcp:127.0.0.1:5760" if args.source else "/dev/ttyAMA0")

VIDEO_PORT, STATUS_PORT, CMD_PORT = 5600, 5601, 5602
LINK_TIMEOUT = 3.0      # s without operator heartbeat -> RTL
SEARCH_TIMEOUT = 60.0   # s without a lock -> RTL
VIDEO_SIZE = (1920, 1080)

overlay = {"dets": [], "box": None, "text": "IDLE"}


def draw(img, w, h):
    px = lambda b: ((int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)))
    for d in overlay["dets"]:
        cv2.rectangle(img, *px(d), (0, 255, 255), 2)
    if overlay["box"]:
        cv2.rectangle(img, *px(overlay["box"]), (0, 0, 255), 4)
    cv2.putText(img, overlay["text"], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)


def pi_detector():
    """Pi camera + Hailo. Streams the overlaid 1080p30 video to the operator and records it to SD."""
    from picamera2 import MappedArray, Picamera2
    from picamera2.devices import Hailo
    from picamera2.encoders import LibavH264Encoder
    from picamera2.outputs import FileOutput, PyavOutput

    hailo = Hailo(args.hef)
    mh, mw, _ = hailo.get_input_shape()
    cam = Picamera2()
    # lores BGR888 = R,G,B byte order = what YOLO was trained on.
    # shortcut: 16:9 sensor squashed to 640x640 (training used letterbox); crop/letterbox if accuracy suffers.
    cam.configure(cam.create_video_configuration(
        main={"size": VIDEO_SIZE, "format": "XRGB8888"},
        lores={"size": (mw, mh), "format": "BGR888"},
        controls={"FrameRate": 30}))

    def pre(request):
        with MappedArray(request, "main") as m:  # XRGB8888 = B,G,R,X in memory
            draw(m.array, *VIDEO_SIZE)

    cam.pre_callback = pre
    enc = LibavH264Encoder(bitrate=6_000_000)  # Pi 5 has no HW H.264 encoder: software libav
    cam.start_recording(enc, [PyavOutput(f"udp://{args.gcs_ip}:{VIDEO_PORT}", format="mpegts"),
                              FileOutput(time.strftime("flight_%Y%m%d_%H%M%S.h264"))])

    def detect():
        frame = cam.capture_array("lores")
        # NMS HEF output: one array per class of [y0, x0, y1, x1, score] (normalized)
        return [(x0, y0, x1, y1, s) for y0, x0, y1, x1, s in hailo.run(frame)[0] if s >= args.conf]

    return detect


def laptop_detector():
    """Laptop test: webcam / video / image folder + Ultralytics on CPU, shown in a window."""
    from ultralytics import YOLO

    model = YOLO(args.model)
    src = args.source
    if os.path.isdir(src):  # hold each image 30 frames so the tracker can lock like on video
        files = sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")))
        assert files, f"no images in {src}"
        frames = itertools.cycle([f for f in files for _ in range(30)])
        read = lambda: cv2.imread(next(frames))
    else:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)

        def read():
            ok, f = cap.read()
            if not ok:  # loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = cap.read()
            assert ok, f"cannot read {src}"
            return f

    def detect():
        f = read()
        h, w = f.shape[:2]
        b = model(f, imgsz=640, conf=args.conf, verbose=False)[0].boxes
        dets = [(x0 / w, y0 / h, x1 / w, y1 / h, s) for (x0, y0, x1, y1), s in zip(b.xyxy.tolist(), b.conf.tolist())]
        draw(f, w, h)  # overlay of the previous frame: one frame late, fine for a test view
        cv2.imshow("interceptor (laptop test)", f)
        cv2.waitKey(1)
        return dets

    return detect


def send_vel(m, vx, vz, yaw_rate):
    # type_mask: use velocity + yaw_rate only. ArduPilot stops by itself if these stop (GUID_TIMEOUT).
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000010111000111, 0, 0, 0, vx, 0, vz, 0, 0, 0, 0, yaw_rate)


def launch(m, alt):
    m.set_mode("GUIDED")
    m.arducopter_arm()
    t = time.time()
    while time.time() - t < 5:
        m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if m.motors_armed():
            m.mav.command_long_send(m.target_system, m.target_component,
                                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)
            return True
    return False  # pre-arm checks failed: see Mission Planner messages


def main():
    with open(args.key, "rb") as f:
        auth = CommandAuth(f.read().strip())
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", CMD_PORT))
    sock.setblocking(False)

    m = mavutil.mavlink_connection(args.mav, baud=args.baud)
    m.wait_heartbeat()
    m.mav.request_data_stream_send(m.target_system, m.target_component,
                                   mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    print("FC connected")

    detect = laptop_detector() if args.source else pi_detector()
    tracker = Tracker()
    state, heading = "IDLE", 0.0
    t_search = last_hb = last_status = last_fc_hb = time.time()
    while True:
        dets = detect()
        now = time.time()
        locked = tracker.update(dets, now)

        while m.recv_match(blocking=False):
            pass  # mavutil caches the latest of each message in m.messages
        hud, gp = m.messages.get("VFR_HUD"), m.messages.get("GLOBAL_POSITION_INT")

        cmd = None
        while True:
            try:
                data, _ = sock.recvfrom(256)
            except BlockingIOError:
                break
            except ConnectionResetError:  # Windows: ICMP "port unreachable" from an earlier status send
                continue
            c = auth.parse(data)
            if c is not None:
                last_hb = now
                if c != "HB":
                    cmd = c

        if state != "RTL" and (cmd == "ABORT" or (state != "IDLE" and now - last_hb > LINK_TIMEOUT)):
            m.set_mode("RTL")
            state = "RTL"
        elif state == "IDLE" and cmd and cmd.startswith("LAUNCH "):
            try:
                heading = float(cmd.split()[1]) % 360
            except ValueError:
                continue
            if launch(m, args.alt):
                state = "TAKEOFF"
        elif state == "TAKEOFF":
            if gp and gp.relative_alt / 1000 >= 0.9 * args.alt:
                state, t_search = "SEARCH", now
        elif state == "SEARCH":
            if locked:
                state = "TRACK"
            elif now - t_search > SEARCH_TIMEOUT:
                m.set_mode("RTL")
                state = "RTL"
            elif hud:  # fly along the operator's heading, yawing onto it first
                err = wrap180(heading - hud.heading)
                send_vel(m, V_SEARCH if abs(err) < 20 else 0.0, 0.0, clamp(0.03 * err, -1.0, 1.0))
        elif state in ("TRACK", "STRIKE"):
            if not locked:  # lost: search on the current heading; STRIKE needs a fresh CONFIRM
                state, t_search = "SEARCH", now
                heading = hud.heading if hud else heading
            else:
                if state == "TRACK" and cmd == "CONFIRM":
                    state = "STRIKE"
                send_vel(m, *guidance(tracker.box, state == "STRIKE"))

        if now - last_fc_hb > 1:  # lets ArduPilot's GCS failsafe catch a dead Pi
            m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                 mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            last_fc_hb = now
        alt = gp.relative_alt / 1000 if gp else 0.0
        overlay.update(dets=dets, box=tracker.box if locked else None, text=f"{state} alt {alt:.0f}m")
        if now - last_status > 0.5:
            sock.sendto(f"{state} lock={locked} dets={len(dets)} alt={alt:.1f}".encode(),
                        (args.gcs_ip, STATUS_PORT))
            last_status = now


if __name__ == "__main__":
    main()
