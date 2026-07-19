# Paper — Open-World RF Emitter Discovery (DRAFT v0.1)

`main.tex` — IEEE conference format (IEEEtran), 4–6 pp target per `summer_work/PAPER_PLAN.md` §14.
Drafted 2026-07-19 from the frozen findings ledgers; **not yet compiled locally** (no TeX on
mars-4090 — build on Overleaf or any TeX Live).

## Number provenance (do not edit numbers without re-checking the source)

| section | source of truth |
|---|---|
| Protocol, WiSig splits, locked noise rule, P0–P3, board-18 TEST | `summer_work/EVAL_PROTOCOL.md` |
| N-dependent wall, generalist ablations F1–F8, DANN NULL | `gen_rff/FINDINGS.md` |
| BLE three-way study EF1–EF8, M1/M2, W1–W5 | `gen_rff/ext_protocols/EXT_FINDINGS.md` |
| Router demo, S1–S6 gates, rate-aware routing | `gen_rff/demo_router/README.md` |
| Demo operating point (N*=120, eigengap) | `summer_work/results/demo_play1b/DEMO_OPERATING_POINT.md` |

## Open TODOs (marked `\todo{}` in red in the tex)

- Affiliation / advisor block.
- Dataset citations (WiSig, ORACLE, DRFF, BLE D2) + related-work cites from the literature
  survey (LITERATURE_SURVEY.md referenced in NEIGHBOR_ANALYSIS but not found in this repo —
  may live on the other machine).
- Figures: architecture diagram, N-curves (`w4_t3_ncurves.png`), UMAP (`w4_umap.png`),
  regime map — currently text-only; audit_out PNGs are gitignored, re-export for the paper.
- Public-release / repo-URL decision.
- Venue pick (PAPER_PLAN §15: MILCOM / GLOBECOM / ICASSP / DySPAN / workshop).

## Scope guards (from PAPER_PLAN + findings docs)

- Localization / multilateration: out of scope, future-work only.
- Demo geometry (rssi/aoa/tdoa) is mocked — never claim multi-sensor capture.
- Board-18 TEST is spent (one-touch): its 0.219 is reported as-is, never re-run.
- Assisted-K vs deployable-K must stay separately labeled everywhere.
