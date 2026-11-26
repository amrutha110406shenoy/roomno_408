"""Pure logic (no hardware): target tracker, pursuit guidance, signed operator commands.
Shared by main.py (drone) and ground.py (operator laptop). Tested by test_logic.py."""
import hashlib
import hmac

# Guidance tuning. Boxes are normalized 0..1 image coords. Tune in SITL / tethered tests first.
K_YAW = 2.0        # rad/s per unit horizontal error (0.5 = target at image edge)
K_Z = 4.0          # m/s per unit vertical error
K_X = 40.0         # m/s per unit of box-width error while tracking
MAX_YAW = 1.5      # rad/s
MAX_VZ = 3.0       # m/s
V_SEARCH = 3.0     # m/s forward while searching
V_TRACK = 6.0      # m/s max forward while tracking (before operator CONFIRM)
V_STRIKE = 12.0    # m/s forward after CONFIRM; also bounded by ArduPilot ANGLE_MAX
TRACK_W = 0.15     # box width to hold while tracking (stand-off distance)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def wrap180(deg):
    return (deg + 180) % 360 - 180


class Tracker:
    """Single-target lock: first pick the most confident drone, then follow the detection
    nearest the last position. Locked after `lock_frames` consecutive-ish hits."""

    def __init__(self, lock_frames=5, max_jump=0.15, lost_s=0.5, alpha=0.6):
        self.lock_frames, self.max_jump, self.lost_s, self.alpha = lock_frames, max_jump, lost_s, alpha
        self.box, self.hits, self.last_seen = None, 0, 0.0

    @property
    def locked(self):
        return self.box is not None and self.hits >= self.lock_frames

    def update(self, dets, now):
        """dets: list of (x0, y0, x1, y1, score). Returns locked state."""
        cand = None
        if dets:
            if self.box is None:
                cand = max(dets, key=lambda d: d[4])
            else:
                cx, cy = center(self.box)
                dist = lambda d: ((center(d)[0] - cx) ** 2 + (center(d)[1] - cy) ** 2) ** 0.5
                cand = min(dets, key=dist)
                if dist(cand) > self.max_jump:
                    cand = None  # different object: don't jump the lock to it
        if cand:
            a = self.alpha
            self.box = tuple(cand[:4]) if self.box is None else tuple(a * n + (1 - a) * o for n, o in zip(cand[:4], self.box))
            self.hits += 1
            self.last_seen = now
        elif self.box is not None and now - self.last_seen > self.lost_s:
            self.box, self.hits = None, 0
        return self.locked


def guidance(box, strike):
    """Pursuit from the box -> (vx forward m/s, vz down m/s, yaw_rate rad/s) in body frame.
    shortcut: pure pursuit (yaw at target, fly forward); upgrade to proportional navigation
    with gimbal/attitude compensation if it can't catch crossing targets."""
    cx, cy = center(box)
    ex, ey = cx - 0.5, cy - 0.5            # +ex target right, +ey target low
    yaw_rate = clamp(K_YAW * ex, -MAX_YAW, MAX_YAW)
    vz = clamp(K_Z * ey, -MAX_VZ, MAX_VZ)  # NED: positive = descend
    vx = V_STRIKE if strike else clamp(K_X * (TRACK_W - (box[2] - box[0])), -1.0, V_TRACK)
    vx *= max(0.2, 1 - 2 * abs(ex))        # turn onto the target before charging
    return vx, vz, yaw_rate


def sign(key, seq, cmd):
    return hmac.new(key, f"{seq}|{cmd}".encode(), hashlib.sha256).hexdigest()


def make_packet(key, seq, cmd):
    return f"{seq}|{cmd}|{sign(key, seq, cmd)}".encode()


class CommandAuth:
    """Accepts only HMAC-signed commands with a strictly increasing sequence (blocks spoof + replay)."""

    def __init__(self, key):
        self.key, self.last = key, 0

    def parse(self, data):
        try:
            seq_s, cmd, mac = data.decode().split("|")
            seq = int(seq_s)
        except (ValueError, UnicodeDecodeError):
            return None
        if seq <= self.last or not hmac.compare_digest(mac, sign(self.key, seq, cmd)):
            return None
        self.last = seq
        return cmd
