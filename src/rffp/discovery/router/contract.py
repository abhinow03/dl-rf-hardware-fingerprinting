"""FROZEN JSON message contract — Design-2 + Phase-4 `protocol` amendment.

One message per track. Crosses receiver -> router -> base-station. The receiver sets
everything except `fingerprint_id`; the ROUTER sets `protocol`; the BASE STATION sets
`fingerprint_id` (namespaced d<k>.<protocol>, e.g. d1.drone / d2.wifi / d3.ble). Ground-truth
device identity is NEVER a field here (see scoring.py).

Phase-5 amendment: the embedding dimension is PER PROTOCOL. Deep tiers (T-A/T-B/T-C) emit
512-D; the BLE native tier (T-D) emits 128-D. EMB_DIM stays the legacy default (512) and
EMB_DIM_BY_PROTO carries the per-protocol contract; validators check length against the
message's own protocol when present.

DEMO-SIDE — NOT PAPER RESULTS.
"""
EMB_DIM = 512                                   # legacy default (deep tiers)
EMB_DIM_BY_PROTO = {"wifi": 512, "drone_ocusync": 512, "unknown": 512, "ble": 128}
_ALLOWED_DIMS = set(EMB_DIM_BY_PROTO.values())  # accepted when protocol not yet known
PROTOCOLS = {"wifi", "drone_ocusync", "unknown", "ble"}

RECEIVER_FIELDS = {
    "receiver_id": str, "track_id": str, "timestamp": float,
    "embedding": list,          # 512 floats, L2-normalized track fingerprint (deep tier)
    "rssi": float, "aoa_deg": float, "tdoa_ns": float,
    "n_windows": int, "partial": bool, "class": str,   # class == "unknown" (open-world)
}
ROUTER_FIELD = "protocol"        # str in PROTOCOLS, + "protocol_conf" float
BASE_STATION_FIELD = "fingerprint_id"   # "d<k>.<protocol>" assigned per protocol group


def _emb_len_error(m):
    """Per-protocol embedding-length check. Uses the message's protocol when present,
    else accepts any contracted dim (receiver stage doesn't know the protocol yet)."""
    emb = m.get("embedding")
    if not isinstance(emb, list):
        return None
    proto = m.get(ROUTER_FIELD)
    if proto in EMB_DIM_BY_PROTO:
        want = EMB_DIM_BY_PROTO[proto]
        if len(emb) != want:
            return f"embedding len {len(emb)}!={want} (protocol '{proto}')"
    elif len(emb) not in _ALLOWED_DIMS:
        return f"embedding len {len(emb)} not in contracted dims {sorted(_ALLOWED_DIMS)}"
    return None


def validate_receiver_message(m):
    errs = []
    for k, t in RECEIVER_FIELDS.items():
        if k not in m:
            errs.append(f"missing '{k}'"); continue
        if t is float and isinstance(m[k], (int, float)):
            continue
        if not isinstance(m[k], t):
            errs.append(f"'{k}' type {type(m[k]).__name__}!={t.__name__}")
    e = _emb_len_error(m)
    if e:
        errs.append(e)
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
    e = _emb_len_error(m)
    if e:
        errs.append(e)
    return errs
