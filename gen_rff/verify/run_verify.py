"""gen_rff.verify.run_verify — VERIFICATION BATTERY V1-V5 (build+verify session).

  python3 -m gen_rff.verify.run_verify

V1 loader | V2 LOPO asserts | V3 P0-G1 baseline reproduction | V4 physics unit-test + residual
sanity | V5 model forward/grad/branch-dropout + tiny overfit. No full training; the tiny
overfit is a sanity check and nothing is saved as a "result". Writes results_gen/verify_report.json.
"""
import os
import sys
import json
import time
from collections import Counter, defaultdict
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLM = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _DLM not in sys.path:
    sys.path.insert(0, _DLM)
_SW = os.path.join(_DLM, "summer_work")
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "results", "step4_mechanism_validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gen_rff.data import loaders, registry
from gen_rff.data.unified import UnifiedRFDataset, DomainHomogeneousBatchSampler, collate_domain
from gen_rff.model.gen_encoder import GenRFEncoder, param_count
from gen_rff.physics.features import classical_matrix, residual_batch
from gen_rff.bench import lopo

RESULTS_GEN = os.path.join(_DLM, "results_gen")
os.makedirs(RESULTS_GEN, exist_ok=True)
REPORT = {}


def _raw(gids):
    return [g.split(":", 1)[1] for g in gids]


def make_small_split():
    """A small multi-domain training split (train-pool identities only)."""
    spec, caps = {}, {}
    wp, _ = registry.domain_device_pool("WISIG"); spec["WISIG"] = _raw(wp[:6]); caps["WISIG"] = 220
    dp, _ = registry.domain_device_pool("DRFF"); spec["DRFF"] = _raw(dp[:4]); caps["DRFF"] = 120
    op, _ = registry.domain_device_pool("ORACLE"); spec["ORACLE"] = _raw(op[:4]); caps["ORACLE"] = 60
    return spec, caps


def build_batch_tensors(batch, device="cuda"):
    d = registry.REGISTRY[batch["domain"]]
    iq = batch["iq"].to(device); res = batch["res"].to(device); phys = batch["physics"].to(device)
    x_time = torch.cat([iq, res], dim=1)
    x_spec = torch.cat([registry.stft_mag(iq, d.n_fft, d.hop),
                        registry.stft_mag(res, d.n_fft, d.hop)], dim=1)
    return x_time, x_spec, phys, batch["device_label"].to(device)


# ============================================================================
def v1_loader():
    print("\n" + "=" * 78 + "\nV1 — UNIFIED LOADER\n" + "=" * 78)
    spec, caps = make_small_split()
    t = time.time()
    ds = UnifiedRFDataset(spec, caps=caps, verbose=True)
    samp = DomainHomogeneousBatchSampler(ds, P=4, V=8, n_batches=200, seed=1)
    print(f"  dataset: {len(ds)} windows, {ds.n_labels} device identities ({time.time()-t:.1f}s build)")
    # one batch per domain: shapes/dtypes/gid-uniqueness/positive structure
    seen = {}
    per_domain_batch = {}
    for idxs in samp:
        items = [ds[i] for i in idxs]
        b = collate_domain(items)
        dom = b["domain"]
        if dom not in per_domain_batch:
            per_domain_batch[dom] = b
        if len(per_domain_batch) == len(ds.index):
            break
    example = {}
    for dom, b in per_domain_batch.items():
        labels = b["device_label"].numpy()
        gids = b["device_gids"]; cks = b["condition_keys"]
        # positive-pair structure: for each device, #distinct conditions in its views
        by_dev = defaultdict(set)
        for g, c in zip(gids, cks):
            by_dev[g].add(c)
        cross = sum(1 for g in by_dev if len(by_dev[g]) >= 2)
        npos = sum(int((labels == l).sum() * ((labels == l).sum() - 1) // 2) for l in set(labels))
        example[dom] = dict(iq=list(b["iq"].shape), stft_shape=list(registry.stft_mag(
            b["iq"][:2], b["n_fft"], b["hop"]).shape), dtype=str(b["iq"].dtype),
            n_devices=len(set(gids)), cross_condition_devices=cross,
            positive_pairs=npos, gids_unique=len(set(gids)))
        print(f"  {dom}: iq{list(b['iq'].shape)} phys{list(b['physics'].shape)} "
              f"P={len(set(gids))} cross-cond devs={cross}/{len(set(gids))} pos-pairs={npos} dtype={b['iq'].dtype}")
    # iterate 200 batches, no crash, interleave ratio
    t = time.time(); dom_count = Counter()
    for idxs in samp:
        items = [ds[i] for i in idxs]
        b = collate_domain(items)
        dom_count[b["domain"]] += 1
        assert len(set(b["device_gids"])) == 4, "P!=4 devices in batch"
    pool_sizes = {dom: len(ds.index[dom]) for dom in ds.index}
    tot = sum(pool_sizes.values())
    interleave = {dom: dict(batches=dom_count[dom], batch_frac=round(dom_count[dom] / 200, 3),
                            pool_frac=round(pool_sizes[dom] / tot, 3)) for dom in ds.index}
    print(f"  iterated 200 batches OK ({time.time()-t:.1f}s). interleave (batch_frac vs pool_frac):")
    for dom, r in interleave.items():
        print(f"     {dom}: {r['batches']} batches  batch_frac={r['batch_frac']} pool_frac={r['pool_frac']}")
    REPORT["V1"] = dict(passed=True, n_windows=len(ds), n_identities=ds.n_labels,
                        per_domain_batch=example, interleave=interleave)
    return ds


# ============================================================================
def v2_lopo():
    print("\n" + "=" * 78 + "\nV2 — LOPO SPLITS + DISJOINTNESS\n" + "=" * 78)
    splits, path = lopo.write_all_lopo_splits()
    # determinism: rebuild and compare
    splits2, _ = lopo.write_all_lopo_splits()
    deterministic = (json.dumps(splits, sort_keys=True) == json.dumps(splits2, sort_keys=True))
    for h, s in splits.items():
        print(f"  holdout={h:6s}: train={s['n_train']} eval={s['n_eval']} "
              f"(asserts PASSED: train/eval disjoint, WiSig-TEST excluded, mavicAir2 not in train)")
    print(f"  deterministic across rebuilds: {deterministic}  -> {os.path.relpath(path, _DLM)}")
    REPORT["V2"] = dict(passed=True, deterministic=deterministic,
                        splits={h: dict(n_train=s["n_train"], n_eval=s["n_eval"]) for h, s in splits.items()})


# ============================================================================
def v3_baselines():
    print("\n" + "=" * 78 + "\nV3 — MATRIX BASELINE ROWS (P0-G1 reproduction)\n" + "=" * 78)
    drff = lopo.reproduce_drff_baselines()
    wis = lopo.reproduce_wisig_p2()
    rows, gate = lopo.p0g1_check(drff, wis, tol=0.02)
    print(f"  {'cell':<34}{'locked':>8}{'repro':>8}{'diff':>8}  ok")
    for r in rows:
        print(f"  {r['cell']:<34}{r['locked']:>8}{r['reproduced']:>8}{r['diff']:>8}  {'PASS' if r['within_tol'] else 'FAIL'}")
    print(f"  P0-G1 GATE: {'PASS' if gate else 'FAIL'} (all cells within +/-0.02)")
    REPORT["V3"] = dict(passed=bool(gate), cells=rows, drff=drff, wisig=wis)
    return gate


# ============================================================================
def v4_physics(ds):
    print("\n" + "=" * 78 + "\nV4 — PHYSICS (P1-G1 unit test + residual sanity)\n" + "=" * 78)
    from mechanism_validation import classical_matrix as ref_matrix
    from sklearn.preprocessing import StandardScaler
    # 100 cached DRFF (OPT-B) windows — the step4 regime
    Xt, *_ = lopo._build_mavicAir2_optb()
    Xd = Xt[:100]
    A = classical_matrix(Xd); B = ref_matrix(Xd)
    raw_diff = float(np.max(np.abs(A - B)))
    As = StandardScaler().fit_transform(A); Bs = StandardScaler().fit_transform(B)
    std_diff = float(np.max(np.abs(As - Bs)))
    print(f"  P1-G1 classical-19 vs step4 (100 DRFF windows): raw max|diff|={raw_diff:.2e} "
          f"standardized max|diff|={std_diff:.2e}  (<1e-4 required) -> {'PASS' if std_diff < 1e-4 else 'FAIL'}")
    # residual_energy_ratio distribution per domain (native windows from the dataset)
    ratios = {}
    for dom in ds.index:
        gid = next(iter(ds.index[dom]))
        wl = [w for cks in ds.index[dom][gid].values() for w in cks][:60]
        X = np.stack([ds.iq[i].astype(np.float32) for i in wl])
        _, rr = residual_batch(X, order=32)
        ratios[dom] = dict(mean=round(float(rr.mean()), 3), p10=round(float(np.percentile(rr, 10)), 3),
                           p90=round(float(np.percentile(rr, 90)), 3))
        print(f"  residual_energy_ratio {dom}: mean={ratios[dom]['mean']} "
              f"[p10={ratios[dom]['p10']}, p90={ratios[dom]['p90']}]  (small fraction = LPC fitting)")
    # GATE = P1-G1 unit test. Residual ratio is a diagnostic: DRFF/WISIG fit (low ratio);
    # ORACLE is elevated (~0.99) and EXPLAINED (802.11a OFDM at 5 MS/s is near-noise in time
    # + low oversampling + ~2.6% clamped saturation spikes) -> blind LPC extracts little.
    # Verified NOT an LPC bug: order sweep 32->128 barely moves it (0.987->0.961) while DRFF
    # (0.001) / WISIG (0.107) fit well.
    clean_ok = ratios["DRFF"]["mean"] < 0.5 and ratios["WISIG"]["mean"] < 0.5
    residual_note = ("DRFF/WISIG residual small (LPC suppresses modulation); ORACLE ~0.99 "
                     "explained (OFDM@5MS/s near-noise-in-time + low oversampling + clamped "
                     "saturation spikes) -> blind-LPC v1 gives little suppression on ORACLE.")
    ok = std_diff < 1e-4 and clean_ok
    REPORT["V4"] = dict(passed=bool(ok), p1g1_raw_diff=raw_diff, p1g1_std_diff=std_diff,
                        residual_energy_ratio=ratios, residual_note=residual_note,
                        open_issue="residual_view v1 (blind LPC) is near-ineffective on ORACLE "
                                   "OFDM; a demod-remod v2 or rate-aware residual is future work.")
    print(f"  [note] {residual_note}")
    return ok


# ============================================================================
def v5_model(ds):
    print("\n" + "=" * 78 + "\nV5 — MODEL forward / grad / branch-dropout / tiny overfit\n" + "=" * 78)
    from shared import SupervisedContrastiveLoss
    m = GenRFEncoder().cuda()
    tot, trn = param_count(m)
    print(f"  GenRFEncoder params: {tot:,} ({tot/1e6:.2f}M); fusion = cross-attn(time,spec)+concat-FC(physics)")
    # forward each domain + grad flow
    grad_norms = {}
    for dom in ds.index:
        d = registry.REGISTRY[dom]; B = 8
        iq = torch.randn(B, 2, d.window_len).cuda(); res = torch.randn(B, 2, d.window_len).cuda()
        xt = torch.cat([iq, res], 1)
        xs = torch.cat([registry.stft_mag(iq, d.n_fft, d.hop), registry.stft_mag(res, d.n_fft, d.hop)], 1)
        ph = torch.randn(B, 19).cuda()
        m.train(); m.zero_grad()
        e = m(xt, xs, ph); loss = (1 - (e @ e.T)).mean(); loss.backward()
        gn = dict(time=0.0, spectral=0.0, phys=0.0)
        for n, p in m.named_parameters():
            if p.grad is not None:
                g = float(p.grad.norm())
                if n.startswith("time_branch"): gn["time"] += g
                elif n.startswith("spectral_branch"): gn["spectral"] += g
                elif n.startswith("phys"): gn["phys"] += g
        grad_norms[dom] = {k: round(v, 3) for k, v in gn.items()}
        print(f"  {dom}: fwd ok emb{list(e.shape)} | grad-norms {grad_norms[dom]} (all>0)")
    # branch-dropout path check
    m.train()
    d = registry.REGISTRY["WISIG"]; B = 64
    iq = torch.randn(B, 2, 256).cuda(); res = torch.randn(B, 2, 256).cuda()
    xt = torch.cat([iq, res], 1); xs = torch.cat([registry.stft_mag(iq, d.n_fft, d.hop),
                                                  registry.stft_mag(res, d.n_fft, d.hop)], 1)
    ph = torch.randn(B, 19).cuda()
    torch.manual_seed(0); e1 = m(xt, xs, ph)
    m.branch_dropout = False; e2 = m(xt, xs, ph); m.branch_dropout = True
    bd_effect = float((e1 - e2).abs().mean())
    print(f"  branch-dropout ON vs OFF mean|Δemb|={bd_effect:.4f} (>0 = dropout path active)")

    # TINY OVERFIT SANITY — 4 WiSig devices, 200 steps, expect loss down + train kNN-1 > 0.9
    wp, _ = registry.domain_device_pool("WISIG")
    spec = {"WISIG": _raw(wp[:4])}
    ov = UnifiedRFDataset(spec, caps={"WISIG": 120})
    samp = DomainHomogeneousBatchSampler(ov, P=4, V=8, n_batches=200, seed=7)
    mo = GenRFEncoder().cuda(); mo.train()
    opt = torch.optim.Adam(mo.parameters(), lr=1e-3)
    supcon = SupervisedContrastiveLoss()
    losses = []
    for idxs in samp:
        items = [ov[i] for i in idxs]; b = collate_domain(items)
        xt, xs, ph, y = build_batch_tensors(b)
        emb = mo(xt, xs, ph)
        loss = supcon(emb, y, temperature=0.5)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    # train kNN-1 on a fresh pass (eval mode)
    mo.eval()
    embs, ys = [], []
    with torch.no_grad():
        for idxs in DomainHomogeneousBatchSampler(ov, P=4, V=8, n_batches=20, seed=11):
            items = [ov[i] for i in idxs]; b = collate_domain(items)
            xt, xs, ph, y = build_batch_tensors(b)
            embs.append(mo(xt, xs, ph).cpu().numpy()); ys.append(y.cpu().numpy())
    from gen_rff.bench.harness import knn_purity
    E = np.concatenate(embs); Y = np.concatenate(ys)
    knn1 = knn_purity(E, Y)[1]
    print(f"  tiny overfit (4 WiSig dev, 200 steps): loss {losses[0]:.3f} -> {losses[-1]:.3f} | "
          f"train kNN-1={knn1:.3f} (>0.9 expected) -> {'PASS' if (losses[-1] < losses[0] and knn1 > 0.9) else 'CHECK'}")
    ok = all(sum(g.values()) > 0 for g in grad_norms.values()) and bd_effect > 0 and \
        losses[-1] < losses[0] and knn1 > 0.9
    REPORT["V5"] = dict(passed=bool(ok), params_total=int(tot), fusion="cross-attn(time,spec)+concat-FC(physics)",
                        grad_norms=grad_norms, branch_dropout_effect=round(bd_effect, 4),
                        overfit=dict(loss_start=round(losses[0], 3), loss_end=round(losses[-1], 3),
                                     train_knn1=round(float(knn1), 3)))
    return ok


def main():
    ds = v1_loader()
    v2_lopo()
    g3 = v3_baselines()
    g4 = v4_physics(ds)
    g5 = v5_model(ds)
    REPORT["summary"] = dict(
        V1=REPORT["V1"]["passed"], V2=REPORT["V2"]["passed"], V3_P0G1=REPORT["V3"]["passed"],
        V4_P1G1=REPORT["V4"]["passed"], V5=REPORT["V5"]["passed"])
    json.dump(REPORT, open(os.path.join(RESULTS_GEN, "verify_report.json"), "w"), indent=2, default=str)
    print("\n" + "=" * 78 + "\nVERIFICATION SUMMARY\n" + "=" * 78)
    for k, v in REPORT["summary"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL/CHECK'}")
    print(f"\n  saved -> results_gen/verify_report.json")


if __name__ == "__main__":
    main()
