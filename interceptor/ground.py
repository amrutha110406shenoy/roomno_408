"""Operator console (laptop). Sends signed commands + heartbeat, prints drone status.
Video: ffplay -fflags nobuffer -flags low_delay -framedrop udp://0.0.0.0:5600

  l <heading_deg>  launch toward the target (compass heading you see it at)
  c                confirm lock -> strike
  a                abort -> RTL
  q                quit (drone RTLs after 3 s without heartbeat)
"""
import argparse
import socket
import threading
import time

from logic import make_packet

ap = argparse.ArgumentParser()
ap.add_argument("--drone-ip", required=True)
ap.add_argument("--key", default="secret.key")
args = ap.parse_args()

with open(args.key, "rb") as f:
    key = f.read().strip()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("", 5601))
lock = threading.Lock()


def send(cmd):
    with lock:  # time_ns keeps the sequence increasing across restarts
        sock.sendto(make_packet(key, time.time_ns(), cmd), (args.drone_ip, 5602))


def heartbeat():
    while True:
        send("HB")
        time.sleep(0.5)


def status():
    while True:
        try:
            print("\r[drone]", sock.recvfrom(256)[0].decode(), end="   ", flush=True)
        except ConnectionResetError:  # Windows: drone not listening yet
            pass


threading.Thread(target=heartbeat, daemon=True).start()
threading.Thread(target=status, daemon=True).start()
print(__doc__)
while True:
    parts = input().strip().split()
    if not parts:
        continue
    if parts[0] == "l" and len(parts) == 2:
        try:
            send(f"LAUNCH {float(parts[1]) % 360}")
        except ValueError:
            print("heading must be a number")
    elif parts[0] == "c":
        send("CONFIRM")
    elif parts[0] == "a":
        send("ABORT")
    elif parts[0] == "q":
        break
