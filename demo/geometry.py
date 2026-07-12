"""Fixed synthetic RF geometry for MOCKED rssi / aoa / tdoa.

A scene places R receiver-drones and E emitters at fixed 2-D positions. The mock
measurements are pure functions of (receiver position, emitter position), so they are
CONSISTENT across every message from the same receiver-emitter pair -> downstream
multilateration is coherent.

IMPORTANT: geometry is keyed by the emitter's SCENE INDEX (a physical position), never
by its airframe label. The discovery path (base-station clustering) ignores these fields
entirely; they are carried only for the downstream AR / multilateration layer. This is a
deliberate honesty point: geometry could trivially separate co-located units, but the
fingerprint discovery is geometry-free.

DEMO-SIDE — NOT PAPER RESULTS.
"""
import math

C_MPS = 299_792_458.0        # speed of light
_TX_P0_DBM = -30.0           # reference emitted power proxy
_PATHLOSS_N = 2.2            # path-loss exponent
_D0 = 1.0                    # reference distance (m)

# Fixed receiver-drone layout (up to 8 receivers), meters.
RECEIVER_POS = [
    (-120.0, -120.0), (120.0, -120.0), (120.0, 120.0), (-120.0, 120.0),
    (0.0, -170.0), (170.0, 0.0), (0.0, 170.0), (-170.0, 0.0),
]

# Fixed emitter scene positions (up to 8 distinct units), meters. Distinct so a
# geometry-based system COULD separate them; the fingerprint system does not use them.
EMITTER_POS = [
    (10.0, 15.0), (-40.0, 30.0), (35.0, -25.0), (-20.0, -45.0),
    (55.0, 50.0), (-60.0, -10.0), (5.0, 60.0), (-5.0, -65.0),
]


def receiver_pos(receiver_idx):
    return RECEIVER_POS[receiver_idx % len(RECEIVER_POS)]


def emitter_pos(scene_idx):
    return EMITTER_POS[scene_idx % len(EMITTER_POS)]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def rssi_dbm(receiver_idx, scene_idx):
    d = max(_dist(receiver_pos(receiver_idx), emitter_pos(scene_idx)), _D0)
    return round(_TX_P0_DBM - 10.0 * _PATHLOSS_N * math.log10(d / _D0), 2)


def aoa_deg(receiver_idx, scene_idx):
    rx, em = receiver_pos(receiver_idx), emitter_pos(scene_idx)
    return round(math.degrees(math.atan2(em[1] - rx[1], em[0] - rx[0])), 2)


def tdoa_ns(receiver_idx, scene_idx, ref_receiver_idx=0):
    em = emitter_pos(scene_idx)
    d = _dist(receiver_pos(receiver_idx), em)
    d_ref = _dist(receiver_pos(ref_receiver_idx), em)
    return round((d - d_ref) / C_MPS * 1e9, 2)
