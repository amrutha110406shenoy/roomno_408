# Avarodhak – interceptor drone

Operator sees the target → `l <heading>` → drone takes off, flies that heading, YOLOv8n finds the drone,
locks → operator checks the live video, sends `c` (confirm) → drone flies straight at the target.
`a` aborts (RTL) at any time. Losing the operator link for 3 s also triggers RTL.

```
train/        voc2yolo.py (DUT Anti-UAV -> YOLO) · colab_train.ipynb (real GPU training)
hailo/        compile_hef.sh (ONNX -> Hailo-8L .hef, in WSL Ubuntu 22.04)
interceptor/  logic.py (tracker, guidance, auth) · main.py (Pi or laptop test) · ground.py (operator) · test_logic.py
tools/sitl/   ArduCopter simulator for Windows (from firmware.ardupilot.org)
datasets/     DUT Anti-UAV raw + converted (dut_yolo/data.yaml)
```

## 1. Laptop test (no Pi, no drone) – do this first
Three terminals in `interceptor/` (need `pip install ultralytics pymavlink`, and a `secret.key`:
`python -c "import secrets;print(secrets.token_hex(32))" > secret.key`):
```bash
# 1 simulator (waits for main.py)
cd ../tools/sitl && ./ArduCopter.exe -M + -O -35.363261,149.165230,584,353 -s 1 -w --defaults copter.parm
# 2 drone brain: webcam (0), a video file, or a folder of images
python main.py --gcs-ip 127.0.0.1 --source 0 --model best.pt
# 3 operator
python ground.py --drone-ip 127.0.0.1
```
Wait ~30 s for the simulator's GPS, then `l 90`. If it doesn't arm, send `l 90` again.
Point the webcam at a drone video on your phone → it locks (red box) → `c` → STRIKE → `a` → RTL.
Optional: Mission Planner → connect TCP 127.0.0.1 port 5762 to watch the simulated drone on a map.

## 2. Train for real (Google Colab, T4 GPU)
Upload `train/colab_train.ipynb` to colab.research.google.com → Runtime → T4 GPU → Run all (~1–2 h).
It downloads DUT Anti-UAV, trains YOLOv8n 640 for 100 epochs, and downloads `best.pt`, `best.onnx`, `calib.zip`.
Put `best.pt` in `interceptor/` (laptop test), and `best.onnx` + `calib.zip` in `hailo/`.
Add 200+ images from your own camera (sky, birds, your target-drone type) for real-world accuracy.

## 3. Compile for the AI HAT+ (Hailo-8L)
One-time: PowerShell as admin → `wsl --install -d Ubuntu-22.04` (reboot, create a user).
Download from hailo.ai Developer Zone (free login) the **Dataflow Compiler** and **Model Zoo** wheels
(Linux, Python 3.10) matching the Pi's HailoRT version (`hailortcli --version` on the Pi), put them in `hailo/`, then in Ubuntu:
```bash
cd /mnt/c/Users/amrut/Avarodhak/hailo && bash compile_hef.sh
```
Copy `hailo/drone_yolov8n.hef` to the Pi's `interceptor/`.

## 4. Raspberry Pi 5 setup (Pi OS Bookworm 64-bit on the 128 GB card)
```bash
sudo apt update && sudo apt install -y hailo-all python3-picamera2 python3-opencv python3-av
python3 -m venv --system-site-packages ~/venv && ~/venv/bin/pip install pymavlink
echo "dtparam=uart0=on" | sudo tee -a /boot/firmware/config.txt && sudo reboot
```
Copy the `interceptor/` folder (with the **same** `secret.key` as the laptop) to the Pi.
Wiring: Pixhawk TELEM2 TX→Pi GPIO15 (RX), RX→GPIO14 (TX), GND→GND. Power the Pi from its own 5 V / 5 A BEC, **not** from TELEM.
Pi camera on the CAM/DISP 0 connector; check with `rpicam-hello`.

ArduPilot params: `SERIAL2_PROTOCOL=2`, `SERIAL2_BAUD=921`, `FENCE_ENABLE=1` (+ `FENCE_RADIUS`, `FENCE_ALT_MAX`),
`FS_GCS_ENABLE=1` (the Pi heartbeats as GCS → RTL if the Pi dies), `ANGLE_MAX` (limits strike speed).

## 5. Run on the drone
Pi: `cd interceptor && ~/venv/bin/python main.py --gcs-ip <laptop-ip>`
Laptop: `python ground.py --drone-ip <pi-ip>` and `ffplay -fflags nobuffer -flags low_delay -framedrop udp://0.0.0.0:5600`
Each flight is recorded to `interceptor/flight_*.h264` on the SD card.

Test order after the laptop test: props off with real FC (watch Mission Planner) → tethered / open field with a
slow target and low `V_STRIKE` in `logic.py` → raise gains.

Notes: video is 1080p30 (Pi 5 has no H.264 hardware encoder, Pi cameras can't do 4K30).
Use a camera with a wide field of view (Camera Module 3 Wide) so the target stays in frame.
