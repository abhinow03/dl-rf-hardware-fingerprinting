# FROZEN JSON MESSAGE CONTRACT — Design-2 base-station association

*DEMO-SIDE — NOT PAPER RESULTS.*

One message per **track** (one receiver-drone's observation of one emitter). The message is
the only thing that crosses the **receiver → base-station** boundary. The authoritative
schema + validators live in [`contract.py`](contract.py); this document is the human-readable
freeze.

## Receiver message (pre-association)

Emitted by a receiver-drone. Every field is REQUIRED.

| field | type | meaning |
|-------|------|---------|
| `receiver_id` | str | opaque receiver-drone id, e.g. `"rx0"` |
| `track_id` | str | opaque unique id for this receiver-emitter track |
| `timestamp` | float | synthetic capture time (seconds) |
| `embedding` | list[float] | **512** floats, L2-normalized track fingerprint (head-free encoder, mean-pooled over the track's windows) |
| `rssi` | float | dBm, **MOCKED** from fixed synthetic geometry |
| `aoa_deg` | float | angle-of-arrival degrees, **MOCKED** |
| `tdoa_ns` | float | time-difference-of-arrival ns vs the reference receiver, **MOCKED** |
| `n_windows` | int | windows actually accumulated into the track (≤ N\*=120) |
| `class` | str | always `"unknown"` (open-world) |

A receiver message MUST NOT contain `fingerprint_id`.

## Associated message (post-association)

The base station returns each message **unchanged except for exactly one added field**:

| field | type | meaning |
|-------|------|---------|
| `fingerprint_id` | str | global discovered id, `"d1"` … `"dK"` — the emitter cluster this track was grouped into |

`rssi` / `aoa_deg` / `tdoa_ns` are carried through for the downstream AR / multilateration
layer. **They are NOT used for grouping** — the base station clusters on `embedding` only
(see `base_station.associate`, which reads no other field and takes no label argument).

## Example

```json
{
  "receiver_id": "rx0",
  "track_id": "mini3pro_2__rx0__s0__rep0",
  "timestamp": 0.0,
  "embedding": [-0.0742, 0.0145, "... 512 floats ..."],
  "rssi": -80.0,
  "aoa_deg": 46.08,
  "tdoa_ns": 0.0,
  "n_windows": 120,
  "class": "unknown",
  "fingerprint_id": "d1"
}
```

Streams are written to [`out/associated_stream.jsonl`](out/) (post-association, one JSON
object per line, with a `_run` tag) and `out/receiver_stream.jsonl` (pre-association).
