# EVAL_PROTOCOL.md — Step-5 correction (added § 4.1)

Added a new subsection **"4.1 DRFF-R2 field semantics — CORRECTION (Step-5)"** after the
one-touch rule (§4). Nothing deleted; Step-3b files retained.

Substance of the correction:
- `TD` = airframe (unchanged, correct).
- `U` = USRP RECEIVER NUMBER (two clock-synced USRP-2943s: u1, u2) — NOT a within-airframe
  channel condition. Step-3b's confound probe wrongly lumped U into {D,C,U,H,St} nuisance
  group; that treated the receiver-diversity axis as channel noise. Flagged superseded.
- Coverage consequence: only 6/23 airframes have both receivers; mavicAir2 has 1, mavicAir2s
  has 1. R=2 cross-receiver same-model test infeasible (needs >=4) -> reported + skipped.
- Step-3b session-disjoint holdout re-verified: session=(airframe,U,D,C) held U IN, so the
  0.60 (mavicAir2) / 0.833 (mavicAir2s) OPT-B numbers are valid cross-capture generalization,
  no re-run needed. Receiver-disjoint probe infeasible (only mavicAir2_1 spans both rx).
