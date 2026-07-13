# GEN-RFF EXT-PROTOCOLS — PHASE 0: Dataset Acquisition + Audit

*R&D sandbox, branch `experimental/generalized-rff`. DEMO/RESEARCH-SIDE — not part of the
locked paper; nothing here trains, downloads, or touches frozen assets. One-session dataset-risk
resolution (GO / CONDITIONAL / NO-GO per protocol) **before** any Zigbee/BLE modelling investment.*

Retrieval/triage date: **2026-07-13.** Session is **CPU-only, download-blocked** (see STEP 0).

> **Note on the plan doc.** `EXT_PROTOCOL_PLAN.md` referenced by the task spec does not exist in
> the tree (`find ~/CAPSTONE -iname '*EXT_PROTOCOL*'` → empty). The task spec is self-contained
> (it embeds the C1–C5 checklist, the D1–D5 candidate list, and the checkpoint requirements
> verbatim), so this audit uses the **spec itself** as authority. The plan doc, if it is authored
> later, should be reconciled against this file.

---

## STEP 0 — DISK BUDGET (hard gate)

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme1n1p5  488G  438G   26G  95% /home        ← data volume
```

**Free space = 26 GB.** Hard rule: *no download that leaves < 100 GB free.* We are **already
below the floor** before downloading anything.

**VERDICT: NO DOWNLOADS THIS SESSION.** This is a global budget failure, not a per-dataset one.
Consequences, stated plainly:

- **STEP 2 (download) does not run** for any candidate — including the two that *would* have
  survived triage on their own merits.
- **STEP 3 (physical audit of downloaded data) cannot run** — there is nothing on disk to audit.
  Every C1–C5 line below is therefore a **metadata-triage** judgement (from papers / dataset
  landing pages), **not** a from-the-bytes confirmation. C2/C3-from-data confirmation is
  **explicitly deferred** to a future download session.
- Nothing was written to `data_raw/` or `audit_out/`; both are pre-emptively gitignored so raw
  data can never enter git.

The session still achieves its purpose: **triage alone resolves the go/no-go risk** (see verdict
section). To actually download later, free the volume to **> 100 GB + dataset size** first
(D2 XIAO ≈ a few GB; D1 CC2530 size unknown-request; D4 is 72 GB and stays out of scope regardless).

---

## STEP 1 — AVAILABILITY TRIAGE (metadata only, no downloads)

| # | Dataset | Protocol | Host | Size | Format | License | Same-model units | Verdict |
|---|---|---|---|---|---|---|---|---|
| **D1** | CC2530×54 (Xie et al., arXiv 2108.04436 "NS-RFF") | Zigbee 802.15.4 | code: [github.com/xrj-com/NS-RFF](https://github.com/xrj-com/NS-RFF); **data: not released** | unknown | raw IQ preambles (1280 samples) | none stated for data | **54** ✓✓ | **CONDITIONAL** (author request) |
| **D2** | XIAO-ESP32C3×31 (Albousayri et al., arXiv 2510.09940) | BLE (GFSK) | OSU NetSTAR / Hamdaoui lab server (`research.engr.oregonstate.edu/hamdaoui`), released Aug 2025 | few GB (est.) | raw IQ | not explicit; lab release-note convention = research use | **31** ✓✓ | **GO** (public, downloadable) |
| **D3** | SDR4IoT (Zenodo 4639390) | BLE + Zigbee | [zenodo.org/records/4639390](https://zenodo.org/records/4639390) | 28.2 GB (zip index 78.7 MB) | raw IQ + demod | CC-BY-4.0 | **3 BLE (nRF52) / 3 Zigbee** ✗ | **NO-GO** (fails C2) |
| **D4** | NEU-BT-WIFI (Elmaghbub/Hamdaoui, arXiv 2303.13538) | BLE + WiFi | [IEEE DataPort](https://ieee-dataport.org/documents/real-world-commercial-wifi-and-bluetooth-dataset-rf-fingerprinting) | 72 GB | raw IQ | IEEE DataPort (open) | 10 **different** COTS combo chips ✗ | **NO-GO for same-model** / retain SECONDARY |
| **D5a** | Uzundurukan BT-Classic DB (MDPI *Data* 2020, [10.3390/data5020055](https://doi.org/10.3390/data5020055)) | Bluetooth Classic | MDPI / repo | — | high-rate captures | CC-BY | assorted phones; twin-pairs only (2/model) ✗ | fallback log only |
| **D5b** | Wearable BT/BLE PHY dataset (2024, RG 379550538) | BLE | ResearchGate/repo | — | PHY-layer | — | mixed wearable models ✗ | fallback log only |

---

## STEP 1 detail — per-candidate C1–C5 evidence

### D1 — ZIGBEE CC2530×54 (arXiv 2108.04436, "NS-RFF")
Spec description **confirmed accurate** from the paper (ar5iv HTML): 54 same-model TI **CC2530**
Zigbee devices, receiver **USRP N210**, **10 MS/s**, 2.4 GHz, **1280-sample** raw preambles;
temporal blocks 1–5 (same day) + blocks 6–9 (**after 18 months** — aging axis).

- **C1 raw IQ** ✓ — 1280-sample raw preamble signals (not symbols/CSI).
- **C2 ≥8 same-model** ✓✓ — 54 units, far above floor; per-unit labels implicit in the classification task.
- **C3 ≥2 sessions/days** ✓✓ — 9 blocks, incl. an 18-month gap; session-disjoint (aging) splits constructible.
- **C4 documented chain** ✓ — USRP N210 @ 10 MS/s, 2.4 GHz stated.
- **C5 license + downloadable now** ✗ — **only the source code is public** (`github.com/xrj-com/NS-RFF`,
  which contains **no data link and no contact/request instructions**). The paper gives **no data-availability
  statement** and **no download URL**. Dataset is **not publicly downloadable**.
- **Verdict: CONDITIONAL** — scientifically the ideal Zigbee set; blocked solely on availability.
  **Contact path for Abhinav (manual, no time pressure):** email the corresponding author of
  arXiv 2108.04436 (Renjie Xie, GitHub `xrj-com`) / open a data-request issue on `xrj-com/NS-RFF`.
  Only the failing item is C5.

### D2 — BLE XIAO-ESP32C3×31 (arXiv 2510.09940)
31 identical **Seeed XIAO ESP32-C3** boards; **raw IQ** via GNURadio; **2× USRP B210** (Rx1/Rx2);
**6 MS/s**; **4 BLE channels** (Ch1/2/14/32); environments = wired lab + **4 outdoor** locations (1–3 m);
6-min warm-up + 2-min capture per device.

- **C1 raw IQ** ✓ — explicitly raw IQ.
- **C2 ≥8 same-model** ✓✓ — 31 identical XIAO units, per-unit labels.
- **C3 ≥2 sessions/days** ✓ (by domain) — 2 receivers × 5 environments × 4 channels give
  **session-disjoint splits by receiver/environment**. The explicit **calendar-day** axis is
  **unconfirmed from metadata** — flag to confirm from the data in the download session; the
  receiver/environment axes already satisfy the spirit of C3 (disjoint-condition splits).
- **C4 documented chain** ✓ — USRP B210 @ 6 MS/s, channels + geometry stated.
- **C5 license + downloadable now** ✓ (public) — hosted on the OSU NetSTAR / Hamdaoui research
  server, "publicly available," Aug-2025 release. **License not printed on the landing page**;
  the sibling WiFi/BT datasets from the same lab ship release notes permitting research use →
  **treat as research-use, confirm the exact license text at download.**
- **Verdict: GO** — the qualifying BLE dataset. Open items (both minor, resolve at download):
  explicit license text, and the calendar-day component of C3.

### D3 — SDR4IoT (Zenodo 4639390)
`experiment_doc.md` (fetched) enumerates emitters by scenario: **Scenario 5 = 3 nRF52 dev kits**
(BLE advertisers: `apuN2`, `apuP22`, `apuQ2`); **Scenario 6 = a Zigbee network of 3 nodes**
(no model names). It is a **testbed traffic / localization** capture (motes + smartphones on
robots at varied positions), CC-BY-4.0, raw IQ + demod present.

- **C1 raw IQ** ✓ — raw IQ present.
- **C2 ≥8 same-model** ✗ **(fatal)** — only **3** BLE units and **3** Zigbee nodes; below the
  6-unit floor. This is exactly the "testbed ≠ same-model fleet" trap the checklist warns about.
- **C3** n/a-ish — positions/scenarios vary but the unit count already disqualifies.
- **C5** ✓ CC-BY, public — but moot.
- **Verdict: NO-GO** — fails C2 (unit count). Not usable for a same-model discovery study.

### D4 — NEU-BT-WIFI (arXiv 2303.13538) — SECONDARY only
**10 COTS combo chipsets** (2 laptops + 8 chips), **72 GB**, multi-day, raw IQ, IEEE DataPort (open).

- **C1 raw IQ** ✓; **C4** ✓; **C5** ✓ (IEEE DataPort open).
- **C2 same-model** ✗ — 10 **distinct** commercial devices, not a same-model fleet → cannot pose
  the same-model RFF question.
- **Verdict: NO-GO for the same-model discovery question.** **Retained as SECONDARY** (router /
  protocol-classifier training material and cross-standard BLE↔WiFi analysis), consistent with the
  spec. **Metadata only — not downloaded** (72 GB, and budget-blocked regardless).

### D5 — logged fallbacks (no action, download nothing)
- **D5a Uzundurukan BT-Classic DB** (MDPI *Data* 2020, CC-BY, publicly available): **Bluetooth
  Classic** (not BLE/Zigbee) smartphone captures; assorted phone models with twin variants
  (≈2 same-model per model) → **fails C2 same-model** and is off-protocol. Fallback log only.
- **D5b Wearable BT/BLE PHY dataset** (2024): publicly described; mixed wearable device models →
  unlikely to meet the ≥8 same-model floor. Fallback log only.

---

## STEP 2 — DOWNLOAD LOG

**None.** Budget gate (STEP 0) failed globally (26 GB free < 100 GB floor). No archives fetched,
no sha256 to record. `data_raw/` empty.

## STEP 3 — PHYSICAL AUDIT

**Not run** — no data on disk to audit. Inventory (3a), IQ sanity (3b), burst structure (3c), SNR
(3d), and session-axis (3e) confirmations are **deferred to the download session**. All C2/C3
judgements above are therefore metadata-level, awaiting from-the-bytes confirmation.

---

## STEP 4 — VERDICT (per protocol)

| Protocol | Best qualifying dataset | Metadata verdict | Pool same-model units | Blocking item |
|---|---|---|---|---|
| **BLE** | **D2 XIAO-ESP32C3×31** (arXiv 2510.09940) | **GO** — public, raw IQ, 31 units | **31** (≥10 ✓) | download-session: confirm license text + calendar-day axis, then physical audit |
| **Zigbee** | **D1 CC2530×54** (arXiv 2108.04436) | **CONDITIONAL** — ideal but data not released | 54 *if obtained* | **author data request** (C5); no public Zigbee same-model set exists as fallback (D3 = 3 units) |

**Data-sufficiency (plan B3 note, ≥10 same-model pool units?):**
- BLE: **31 units — comfortably sufficient**, largest same-model BLE RFF pool found.
- Zigbee: **54 units — comfortably sufficient IF the author request succeeds**; otherwise Zigbee has
  **no qualifying public dataset** (SDR4IoT's 3 units and NEU's different-model chips both fail C2).

**Go/no-go recommendation:**
- **BLE → proceed** (once disk is freed): D2 is a clean GO and the natural first ext-protocol.
- **Zigbee → gated on the D1 author request.** Send the request now (no time pressure); if it is
  not granted, **Zigbee drops to router-enrollment-only and exits the comparative study** — do
  **not** substitute a different-model classification dataset to keep it alive (that changes the
  question, per spec).

**Open CONDITIONALs / contact paths:**
- **D1 CC2530:** email corresponding author of arXiv 2108.04436 (Renjie Xie / GitHub `xrj-com`)
  or open a data-request issue on `github.com/xrj-com/NS-RFF`. Failing item: **C5 only**.
- **D2 XIAO:** confirm exact license text on the OSU Hamdaoui landing page at download; confirm
  the calendar-day component of C3 from the captured metadata.

**Bottom line:** dataset risk is resolved — **BLE is a GO, Zigbee is availability-gated (CONDITIONAL)**.
No modelling investment should precede (a) freeing disk to > 100 GB and (b) resolving the D1 request.
