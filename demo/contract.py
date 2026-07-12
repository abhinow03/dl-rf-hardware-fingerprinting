"""FROZEN JSON message contract (Design-2 base-station association).

A receiver-drone emits one message per track. The message is the ONLY thing that
crosses the receiver -> base-station boundary. The base station may add exactly one
field (`fingerprint_id`) after association; it adds nothing else and removes nothing.

Ground-truth emitter identity is NEVER a field of this message — see scoring.py.

DEMO-SIDE — NOT PAPER RESULTS.
"""

EMB_DIM = 512

# Fields the receiver MUST set (pre-association contract).
RECEIVER_FIELDS = {
    "receiver_id": str,     # opaque receiver-drone id, e.g. "rx0"
    "track_id": str,        # opaque track id (unique per receiver-emitter observation)
    "timestamp": float,     # synthetic capture time (seconds)
    "embedding": list,      # 512 floats, L2-normalized track fingerprint (head-free)
    "rssi": float,          # dBm, MOCKED from fixed synthetic geometry
    "aoa_deg": float,       # angle-of-arrival degrees, MOCKED
    "tdoa_ns": float,       # time-diff-of-arrival ns vs reference receiver, MOCKED
    "n_windows": int,       # windows actually accumulated into this track (<= N*)
    "class": str,           # always "unknown" at the receiver (open-world)
}

# The ONLY field the base station is allowed to add.
BASE_STATION_FIELD = "fingerprint_id"   # str, e.g. "d1" .. "dK"; global discovered id


def validate_receiver_message(m):
    """Return list of contract violations for a pre-association (receiver) message."""
    errs = []
    for k, t in RECEIVER_FIELDS.items():
        if k not in m:
            errs.append(f"missing field '{k}'"); continue
        if t is float and isinstance(m[k], (int, float)):
            pass
        elif not isinstance(m[k], t):
            errs.append(f"field '{k}' type {type(m[k]).__name__} != {t.__name__}")
    if isinstance(m.get("embedding"), list) and len(m["embedding"]) != EMB_DIM:
        errs.append(f"embedding len {len(m['embedding'])} != {EMB_DIM}")
    if m.get("class") != "unknown":
        errs.append(f"class '{m.get('class')}' != 'unknown'")
    if BASE_STATION_FIELD in m:
        errs.append(f"receiver message must NOT carry '{BASE_STATION_FIELD}' "
                    "(assigned only at base station)")
    return errs


def validate_associated_message(m):
    """Return list of contract violations for a post-association (base-station) message."""
    errs = [e for e in _validate_receiver_core(m)]
    if BASE_STATION_FIELD not in m:
        errs.append(f"associated message missing '{BASE_STATION_FIELD}'")
    elif not isinstance(m[BASE_STATION_FIELD], str):
        errs.append(f"'{BASE_STATION_FIELD}' must be str")
    return errs


def _validate_receiver_core(m):
    errs = []
    for k, t in RECEIVER_FIELDS.items():
        if k not in m:
            errs.append(f"missing field '{k}'"); continue
        if t is float and isinstance(m[k], (int, float)):
            continue
        if not isinstance(m[k], t):
            errs.append(f"field '{k}' type {type(m[k]).__name__} != {t.__name__}")
    if isinstance(m.get("embedding"), list) and len(m["embedding"]) != EMB_DIM:
        errs.append(f"embedding len {len(m['embedding'])} != {EMB_DIM}")
    if m.get("class") != "unknown":
        errs.append(f"class '{m.get('class')}' != 'unknown'")
    return errs
