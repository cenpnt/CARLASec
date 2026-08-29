"""
Windows equivalent of Linux `cansend`, for CARLASec on Windows.

The README's manual CAN attacks use SocketCAN's `cansend`, e.g.:
    while true; do cansend vcan0 000001A0#5508010100000012; done
which does not exist on Windows. This sends the same CAN frame over the
Windows UDP CAN shim (win_can_shim, auto-loaded via .pth).

Usage (mirrors cansend's ID#DATA syntax):
    python win_cansend.py vcan0 000001A0#5508010100000012          # send once
    python win_cansend.py --loop vcan0 000001A0#5508010100000012  # repeat (README while-loop)

--interval sets the gap between repeats (default 0.3s). A gap is required on
Windows so the vehicle's ~30 Hz game loop drains each frame before the next
arrives; sending faster makes identical frames pile up in CARLASec's
priority queue, which cannot tie-break equal-priority frames and crashes.
"""
import sys
import time
import argparse
import can  # shim auto-loaded via .pth


def parse_frame(spec):
    # "000001A0#5508010100000012" -> (0x1A0, b'\x55\x08...')
    id_str, data_str = spec.split("#", 1)
    return int(id_str, 16), bytes.fromhex(data_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channel")                     # e.g. vcan0
    ap.add_argument("frame")                       # e.g. 000001A0#5508010100000012
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=float, default=0.3)
    args = ap.parse_args()

    arb_id, data = parse_frame(args.frame)
    bus = can.interface.Bus(channel=args.channel, bustype="socketcan")
    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)

    try:
        if args.loop:
            print(f"Looping {args.frame} on {args.channel} every {args.interval}s (Ctrl+C to stop)", flush=True)
            while True:
                bus.send(msg)
                time.sleep(args.interval)
        else:
            bus.send(msg)
            print(f"Sent {args.frame} on {args.channel}", flush=True)
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()
