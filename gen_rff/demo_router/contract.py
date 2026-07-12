"""FROZEN JSON message contract — Design-2 + Phase-4 `protocol` amendment.

One message per track. Crosses receiver -> router -> base-station. The receiver sets
everything except `fingerprint_id`; the ROUTER sets `protocol`; the BASE STATION sets
`fingerprint_id` (namespaced d<k>.<protocol>, e.g. d1.drone / d2.wifi). Ground-truth device
identity is NEVER a field here (see scoring.py).

DEMO-SIDE — NOT PAPER RESULTS.
"""
EMB_DIM = 512
PROTOCOLS = {"wifi", "drone_ocusync", "unknown"}

RECEIVER_FIELDS = {
    "receiver_id": str, "track_id": str, "timestamp": float,
    "embedding": list,          # 512 floats, L2-normalized track fingerprint (deep tier)
    "rssi": float, "aoa_deg": float, "tdoa_ns": float,
    "n_windows": int, "partial": bool, "class": str,   # class == "unknown" (open-world)
}
ROUTER_FIELD = "protocol"        # str in PROTOCOLS, + "protocol_conf" float
BASE_STATION_FIELD = "fingerprint_id"   # "d<k>.<protocol>" assigned per protocol group


def validate_receiver_message(m):
    errs = []
    for k, t in RECEIVER_FIELDS.items():
        if k not in m:
            errs.append(f"missing '{k}'"); continue
        if t is float and isinstance(m[k], (int, float)):
            continue
        if not isinstance(m[k], t):
            errs.append(f"'{k}' type {type(m[k]).__name__}!={t.__name__}")
    if isinstance(m.get("embedding"), list) and len(m["embedding"]) != EMB_DIM:
        errs.append(f"embedding len {len(m['embedding'])}!={EMB_DIM}")
    if m.get("class") != "unknown":
        errs.append("class!='unknown'")
    if BASE_STATION_FIELD in m:
        errs.append("receiver msg must NOT carry fingerprint_id")
    return errs


def validate_routed_message(m):
    errs = validate_receiver_message(m) if BASE_STATION_FIELD not in m else []
    if m.get(ROUTER_FIELD) not in PROTOCOLS:
        errs.append(f"protocol '{m.get(ROUTER_FIELD)}' not in {PROTOCOLS}")
    return errs


def validate_associated_message(m):
    errs = []
    for k, t in RECEIVER_FIELDS.items():
        if k not in m:
            errs.append(f"missing '{k}'")
    if m.get(ROUTER_FIELD) not in PROTOCOLS:
        errs.append("missing/invalid protocol")
    fid = m.get(BASE_STATION_FIELD)
    if not isinstance(fid, str) or "." not in fid:
        errs.append("fingerprint_id must be namespaced 'd<k>.<protocol>'")
    if isinstance(m.get("embedding"), list) and len(m["embedding"]) != EMB_DIM:
        errs.append(f"embedding len!={EMB_DIM}")
    return errs
