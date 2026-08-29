"""
Windows CAN compatibility shim for CARLASec.

CARLASec's in-vehicle network is written against Linux SocketCAN
(can.interface.Bus(bustype='socketcan', channel='vcan0'|'kcan4')), which does
not exist on Windows. python-can's own 'udp_multicast' backend is nominally
cross-platform but relies on several Linux-only socket features
(IPPROTO_IPV6, SO_TIMESTAMPNS) that fail on Windows.

This module instead provides a tiny, self-contained UDP-broadcast CAN bus that
works on any OS, and monkeypatches can.interface.Bus so every SocketCAN request
is transparently served by it. Import this module BEFORE any bus is created and
none of the ~30 existing call sites need to change.

Each logical SocketCAN channel maps to its own UDP port, so separate buses
(vcan0, kcan4) stay isolated, matching the Linux setup. Multiple processes bind
the same port via SO_REUSEADDR and all receive the broadcast frames, preserving
CARLASec's multi-terminal design (server / sniffer / IDS / attacker).

Scope: raw CAN frames only. UDS/ISO-TP attacks are not covered (python's
isotp/tpsock refuses to run on Windows by design).
"""
import os
import socket
import struct
import can
import can.interface

# One UDP port per logical channel keeps the buses isolated.
_CHANNEL_PORTS = {
    "vcan0": 47411,
    "kcan4": 47412,
}
_DEFAULT_PORT = 47411

# Wire format: arb_id (uint32 LE) | dlc (uint8) | flags (uint8) | data (8 bytes)
# flags bit0 = extended id.  Fixed 14-byte frame, trivially parseable.
_FMT = "<IBB8s"
_FRAME_LEN = struct.calcsize(_FMT)


class UdpBroadcastBus(can.BusABC):
    def __init__(self, channel="vcan0", port=None, receive_own_messages=False, **kwargs):
        self.channel_info = f"udp-broadcast {channel}"
        self._port = port if port is not None else _CHANNEL_PORTS.get(channel, _DEFAULT_PORT)
        self._recv_own = receive_own_messages

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.bind(("", self._port))
        # Record our own source port so we can drop our own echoes if asked.
        self._local = self._sock.getsockname()
        super().__init__(channel=channel, **kwargs)

    def _recv_internal(self, timeout):
        self._sock.settimeout(timeout)
        try:
            data, sender = self._sock.recvfrom(1024)
        except socket.timeout:
            return None, False
        except OSError:
            return None, False
        if len(data) < _FRAME_LEN:
            return None, False
        if not self._recv_own and sender == self._local:
            return None, False
        arb_id, dlc, flags, payload = struct.unpack(_FMT, data[:_FRAME_LEN])
        msg = can.Message(
            arbitration_id=arb_id,
            dlc=dlc,
            is_extended_id=bool(flags & 0x1),
            data=payload[:dlc],
        )
        return msg, False

    def send(self, msg, timeout=None):
        payload = bytes(msg.data[:8]).ljust(8, b"\x00")
        flags = 0x1 if msg.is_extended_id else 0x0
        frame = struct.pack(_FMT, msg.arbitration_id, msg.dlc, flags, payload)
        self._sock.sendto(frame, ("255.255.255.255", self._port))

    def shutdown(self):
        try:
            super().shutdown()
        finally:
            try:
                self._sock.close()
            except OSError:
                pass


_ACTIVE = os.name == "nt"
_orig_Bus = can.interface.Bus


def _patched_Bus(*args, **kwargs):
    bustype = kwargs.pop("bustype", None) or kwargs.pop("interface", None)
    channel = kwargs.pop("channel", None)
    if channel is None and args:
        channel = args[0]
        args = args[1:]

    if _ACTIVE and bustype in ("socketcan", None):
        return UdpBroadcastBus(
            channel=channel or "vcan0",
            receive_own_messages=kwargs.pop("receive_own_messages", False),
        )

    if bustype is not None:
        kwargs["interface"] = bustype
    if channel is not None:
        kwargs["channel"] = channel
    return _orig_Bus(*args, **kwargs)


can.interface.Bus = _patched_Bus
can.Bus = _patched_Bus
