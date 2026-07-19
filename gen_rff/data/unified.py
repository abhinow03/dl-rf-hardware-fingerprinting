"""gen_rff.data.unified — UnifiedRFDataset + DomainHomogeneousBatchSampler.

Yields per-window records with domain-native shapes plus the physics 19-D and the LPC
residual channel. Batches are DOMAIN-HOMOGENEOUS (shapes differ across domains, no padding
hacks); within a batch, P device identities x V views with same-device CROSS-CONDITION
positive pairs guaranteed wherever a device has >= 2 conditions.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .registry import REGISTRY
from ..physics.features import classical_matrix, residual_batch


class UnifiedRFDataset(Dataset):
    def __init__(self, split_spec, caps=None, residual_order=32, verbose=False):
        """split_spec: {domain_name: [raw_device_ids]} (ids WITHOUT the 'DOMAIN:' prefix).
        caps: {domain_name: cap_windows_per_device}. Quarantined domains are refused."""
        caps = caps or {}
        self.records = []          # per-window dicts of small metadata
        self.iq = []               # list of [2,L] float16
        self.res = []              # list of [2,L] float16
        self.phys = []             # list of [19] float32
        self.device_gids = []
        # sampler index: domain -> device_gid -> condition_key -> [global window idx]
        self.index = {}
        gid_to_label = {}
        for dom, ids in split_spec.items():
            d = REGISTRY[dom]
            if d.quarantined:
                raise ValueError(f"{dom} is quarantined; not allowed in a training pool")
            units = d.loader_fn(ids, caps.get(dom, 400))
            self.index.setdefault(dom, {})
            for u in units:
                X = u["X"].astype(np.float32)               # [n,2,L]
                phys = classical_matrix(X, fs=d.native_rate)
                res, _ = residual_batch(X, order=residual_order)
                gid = u["device_gid"]; ck = u["condition_key"]
                if gid not in gid_to_label:
                    gid_to_label[gid] = len(gid_to_label)
                for j in range(X.shape[0]):
                    gi = len(self.iq)
                    self.iq.append(X[j].astype(np.float16))
                    self.res.append(res[j].astype(np.float16))
                    self.phys.append(phys[j])
                    self.device_gids.append(gid)
                    self.records.append(dict(domain=dom, domain_id=list(REGISTRY).index(dom),
                                             protocol_id=d.protocol_id, device_gid=gid,
                                             device_label=gid_to_label[gid], condition_key=ck))
                    self.index[dom].setdefault(gid, {}).setdefault(ck, []).append(gi)
        self.n_labels = len(gid_to_label)
        self.gid_to_label = gid_to_label
        if verbose:
            for dom in self.index:
                nd = len(self.index[dom]); nw = sum(len(v) for g in self.index[dom].values() for v in g.values())
                print(f"  [dataset] {dom}: {nd} devices, {nw} windows")

    def __len__(self):
        return len(self.iq)

    def __getitem__(self, i):
        r = self.records[i]
        return dict(iq=torch.from_numpy(self.iq[i].astype(np.float32)),
                    res=torch.from_numpy(self.res[i].astype(np.float32)),
                    physics=torch.from_numpy(self.phys[i]),
                    device_label=r["device_label"], domain=r["domain"],
                    domain_id=r["domain_id"], protocol_id=r["protocol_id"],
                    device_gid=r["device_gid"], condition_key=r["condition_key"])


def collate_domain(items):
    """All items share one domain -> stack + attach domain STFT params."""
    dom = items[0]["domain"]
    d = REGISTRY[dom]
    return dict(
        iq=torch.stack([it["iq"] for it in items]),
        res=torch.stack([it["res"] for it in items]),
        physics=torch.stack([it["physics"] for it in items]),
        device_label=torch.tensor([it["device_label"] for it in items]),
        domain=dom, n_fft=d.n_fft, hop=d.hop,
        device_gids=[it["device_gid"] for it in items],
        condition_keys=[it["condition_key"] for it in items])


class DomainHomogeneousBatchSampler:
    """Each batch is one domain: P identities x V views, cross-condition positives guaranteed.
    Domains are interleaved round-robin weighted by pool size (#devices)."""

    def __init__(self, dataset, P=4, V=8, n_batches=None, seed=0):
        self.ds = dataset
        self.P, self.V = P, V
        self.rng = np.random.default_rng(seed)
        self.domains = [dom for dom in dataset.index if len(dataset.index[dom]) >= P]
        sizes = {dom: len(dataset.index[dom]) for dom in self.domains}
        tot = sum(sizes.values())
        # round-robin order weighted by pool size
        if n_batches is None:
            n_batches = max(10, tot)
        self.n_batches = n_batches
        # build a weighted domain schedule
        weights = np.array([sizes[dom] for dom in self.domains], float)
        weights = weights / weights.sum()
        self.schedule = list(self.rng.choice(self.domains, size=n_batches, p=weights))

    def _draw_device(self, dom, gid):
        """V views for one device, spanning >=2 conditions when available."""
        conds = self.ds.index[dom][gid]
        ckeys = list(conds.keys())
        picks = []
        if len(ckeys) >= 2:
            # guarantee cross-condition: seed with one window from two distinct conditions
            self.rng.shuffle(ckeys)
            for ck in ckeys[:2]:
                pool = conds[ck]
                picks.append(int(pool[self.rng.integers(len(pool))]))
        # fill the rest from any condition of this device
        allw = [w for ck in ckeys for w in conds[ck]]
        while len(picks) < self.V:
            picks.append(int(allw[self.rng.integers(len(allw))]))
        return picks[:self.V]

    def __iter__(self):
        for dom in self.schedule:
            gids = list(self.ds.index[dom].keys())
            chosen = self.rng.choice(len(gids), size=self.P, replace=False)
            batch = []
            for gi in chosen:
                batch.extend(self._draw_device(dom, gids[gi]))
            yield batch

    def __len__(self):
        return self.n_batches
