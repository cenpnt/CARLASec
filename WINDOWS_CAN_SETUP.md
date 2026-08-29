# Running CARLASec's CAN layer on Windows

CARLASec's in-vehicle network is written against **Linux SocketCAN**
(`can.interface.Bus(bustype='socketcan', channel='vcan0'|'kcan4')`). SocketCAN
is a Linux *kernel* feature (`AF_CAN` sockets + the `vcan` module) and does not
exist on Windows. python-can's own `udp_multicast` backend is nominally
cross-platform but relies on Linux-only socket features (`IPPROTO_IPV6`,
`SO_TIMESTAMPNS`) that fail on this Windows / Python 3.7 build.

This branch (`windows-udp-can`) adds a small compatibility layer so the **raw
CAN** parts of CARLASec run on Windows unchanged.

## What was added

- **`win_can_shim.py`** — a self-contained ~60-line module. It defines a tiny
  `python-can` bus (`UdpBroadcastBus`) that carries CAN frames over plain IPv4
  UDP broadcast on loopback, and monkeypatches `can.interface.Bus` so every
  SocketCAN request is served by it instead. Each logical channel (`vcan0`,
  `kcan4`) maps to its own UDP port, so the buses stay isolated, and multiple
  processes bind the same port (`SO_REUSEADDR`) so the server / sniffer / IDS /
  attacker terminals all see each other's frames — exactly like SocketCAN.
  The patch only activates on Windows (`os.name == 'nt'`); on Linux it is inert
  and the real SocketCAN path is used.

- **`win_can_demo.py`** — a one-command proof: the repo's own
  `attacker.can_attack.CANAttack` injects a throttle-manipulation replay + a
  CAN DoS burst, and a sniffer process captures them. Run:
  ```
  python win_can_demo.py
  ```

## One-time environment setup (not in git)

The shim is auto-activated for every process in the `carla-venv` virtualenv via
a `.pth` file dropped in site-packages:

`C:\Users\s4990998\carla-venv\lib\site-packages\carlasec_win_can.pth`
```
import sys, os; _r = r'\\puffball.labs.eait.uq.edu.au\s4990998\Documents\REIT4842\Code\CARLASec'; (_r not in sys.path) and sys.path.insert(0, _r); __import__('win_can_shim') if os.name == 'nt' else None
```
This adds the repo to `sys.path` and imports the shim at interpreter startup, so
the ~30 existing `can.interface.Bus(...)` call sites need no edits. To disable
the whole adaptation, delete that one `.pth` file.

Extra dependency installed: `msgpack` (pulled in while evaluating
`udp_multicast`; harmless to keep). Also `python-can==4.2.2` (newest that
supports Python 3.7; the repo pins 4.4.2, which needs Python >= 3.8).

## Scope and caveats

- **Covered:** raw CAN frame injection and sniffing — README sections 6-7's CAN
  attacks (throttle/steering replay, DoS, fuzzing) and the CAN-bus IDS.
- **NOT covered:** UDS / ISO-TP attacks (door control, UDS routine control).
  These go through python's `isotp` transport, whose `tpsock` module *explicitly
  refuses to run on Windows* (`raise NotImplementedError("...cannot be used on
  Windows")`). Those demos need real Linux SocketCAN.
- The UDP transport carries CAN *frames* faithfully but is not bit-timing or
  hardware-accurate; use it for functional/attack-logic testing, not timing
  studies.
- Behaviour may differ subtly from the Linux original the authors validated.
  For publication-grade CAN results, run on Linux.

## Landmine unrelated to CAN

`client_run.py`, `client_run_response.py`, and `generate_traffic.py` all default
`--host` to `192.168.160.1` (a leftover), **not** `127.0.0.1`. Always pass
`--host 127.0.0.1` explicitly or they silently hang trying to reach the wrong
address.
