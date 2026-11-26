"""python test_logic.py  -- runs on any PC, no hardware needed."""
from logic import CommandAuth, Tracker, guidance, make_packet, wrap180

# tracker: locks after 5 hits, ignores a far-away second object, drops after 0.5 s unseen
t = Tracker()
d = (0.45, 0.45, 0.55, 0.55, 0.9)
for i in range(4):
    assert not t.update([d], i * 0.03)
assert t.update([d, (0.0, 0.0, 0.05, 0.05, 0.99)], 0.12)
assert abs(t.box[0] - 0.45) < 1e-6, "lock jumped to the other object"
assert t.update([], 0.3)          # brief dropout keeps lock
assert not t.update([], 0.7)      # long dropout loses it

# guidance signs: target right -> yaw right, target low -> descend, strike -> full speed
vx, vz, yr = guidance((0.7, 0.6, 0.8, 0.7), strike=False)
assert yr > 0 and vz > 0 and vx > 0
vx, vz, yr = guidance((0.2, 0.2, 0.3, 0.3), strike=False)
assert yr < 0 and vz < 0
assert guidance((0.45, 0.45, 0.55, 0.55), strike=True)[0] == 12.0
assert guidance((0.3, 0.45, 0.7, 0.55), strike=False)[0] < 0   # too close -> back off

assert wrap180(350 - 10) == -20 and wrap180(10 - 350) == 20

# auth: valid accepted, replay / tamper / wrong key / garbage rejected
a = CommandAuth(b"k")
p = make_packet(b"k", 5, "CONFIRM")
assert a.parse(p) == "CONFIRM"
assert a.parse(p) is None
assert a.parse(make_packet(b"k", 6, "ABORT").replace(b"ABORT", b"STRIK")) is None
assert a.parse(make_packet(b"x", 7, "ABORT")) is None
assert a.parse(b"\xff|junk") is None
assert a.parse(make_packet(b"k", 8, "LAUNCH 90.0")) == "LAUNCH 90.0"
print("ok")
