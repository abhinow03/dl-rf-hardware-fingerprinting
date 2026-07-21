"""Fixed synthetic RF geometry for MOCKED rssi / aoa / tdoa (ported from the earlier demo).

Pure functions of (receiver position, emitter scene position) -> consistent across messages
so downstream multilateration is coherent. Keyed by SCENE INDEX (a physical position), never
by device/protocol label. The discovery path ignores these fields; they ride along only for
the AR / multilateration layer.

DEMO-SIDE — NOT PAPER RESULTS.
"""
import math

C_MPS = 299_792_458.0
_TX_P0_DBM = -30.0
_PATHLOSS_N = 2.2
_D0 = 1.0

RECEIVER_POS = [
    (-120.0, -120.0), (120.0, -120.0), (120.0, 120.0), (-120.0, 120.0),
    (0.0, -170.0), (170.0, 0.0), (0.0, 170.0), (-170.0, 0.0),
]
EMITTER_POS = [
    (10.0, 15.0), (-40.0, 30.0), (35.0, -25.0), (-20.0, -45.0),
    (55.0, 50.0), (-60.0, -10.0), (5.0, 60.0), (-5.0, -65.0),
]


def receiver_pos(i):
    return RECEIVER_POS[i % len(RECEIVER_POS)]


def emitter_pos(i):
    return EMITTER_POS[i % len(EMITTER_POS)]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def rssi_dbm(r, s):
    d = max(_dist(receiver_pos(r), emitter_pos(s)), _D0)
    return round(_TX_P0_DBM - 10.0 * _PATHLOSS_N * math.log10(d / _D0), 2)


def aoa_deg(r, s):
    rx, em = receiver_pos(r), emitter_pos(s)
    return round(math.degrees(math.atan2(em[1] - rx[1], em[0] - rx[0])), 2)


def tdoa_ns(r, s, ref=0):
    em = emitter_pos(s)
    return round((_dist(receiver_pos(r), em) - _dist(receiver_pos(ref), em)) / C_MPS * 1e9, 2)
