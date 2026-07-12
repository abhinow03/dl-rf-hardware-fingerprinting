"""gen_rff.model.gen_encoder — GenRFEncoder: physics-residual front end + physics injection.

Reuses the proven dual-branch family (1D ResNet time + 2D ResNet spectral + cross-attention,
GroupNorm, GAP). Extensions vs the parent RFEncoder:

  * TIME branch input = 4 channels [I, Q, I_res, Q_res]   (residual = LPC-suppressed protocol)
  * SPECTRAL branch input = 4 channels [|STFT(iq)| 2ch, |STFT(res)| 2ch]
  * PHYSICS INJECTION: 19-D classical vector -> LayerNorm -> MLP(19->64->128) = physics token.

FUSION VARIANT (implemented ONE, stated here): **cross-attention for the two DEEP branches
(time<->spectral, exactly as the parent), then CONCAT+FC to inject the physics token.**
Extending nn.MultiheadAttention to a 3rd token of different width (128 vs 256) was not clean,
so physics uses concat+FC. i.e. deep = norm(cat[attn_out+t, s]) (512); out = out_norm(FC(
cat[deep(512), physics(128)] -> 512)). The 512-D is the eval layer (get_encoder_output); a
128-D L2 head sits on top.

FORCED-USAGE branch-dropout (training only, `branch_dropout=True`): per batch element, with
p each, zero the DEEP token (physics-only) OR zero the PHYSICS token (deep-only), mutually
exclusive, so neither pathway can be ignored. `p` and the flag are exposed for ablation.

BRANCH-SAFE: imports (never edits) the parent building blocks from ../../model.py.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLM = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _DLM not in sys.path:
    sys.path.insert(0, _DLM)
import model as _parent          # parent model.py — GN, ResBlock1D/2D (import-safe, class defs only)


class TimeBranch4(nn.Module):
    """Parent TimeBranch with a 4-channel stem [I, Q, I_res, Q_res]."""
    def __init__(self):
        super().__init__()
        GN = _parent.GN
        self.stem = nn.Sequential(
            nn.Conv1d(4, 64, 63, padding=31, bias=False), GN(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1, bias=False), GN(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1, bias=False), GN(64), nn.ReLU())
        self.b1 = _parent.ResBlock1D(64, 64)
        self.b2 = _parent.ResBlock1D(64, 64)
        self.b3 = _parent.ResBlock1D(64, 128)
        self.b4 = _parent.ResBlock1D(128, 128)
        self.proj = nn.Linear(128, 256)

    def forward(self, x):                      # [B, 4, L]
        x = self.stem(x)
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.proj(x.mean(2))            # GAP over time -> [B, 256]


class SpectralBranch4(nn.Module):
    """Parent SpectralBranch with a 4-channel input [|STFT(iq)| 2ch, |STFT(res)| 2ch]."""
    def __init__(self):
        super().__init__()
        GN = _parent.GN
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1, bias=False), GN(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=(2, 1), padding=1, bias=False), GN(64), nn.ReLU())
        self.b1 = _parent.ResBlock2D(64, 64)
        self.b2 = _parent.ResBlock2D(64, 64)
        self.b3 = _parent.ResBlock2D(64, 128)
        self.b4 = _parent.ResBlock2D(128, 128)
        self.proj = nn.Linear(128, 256)

    def forward(self, x):                      # [B, 4, F, T]
        x = self.stem(x)
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.proj(x.mean([2, 3]))       # GAP over freq/time -> [B, 256]


class GenRFEncoder(nn.Module):
    def __init__(self, n_physics=19, branch_dropout=True, p=0.15):
        super().__init__()
        self.time_branch = TimeBranch4()
        self.spectral_branch = SpectralBranch4()
        self.cross_attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.deep_norm = nn.LayerNorm(512)
        # physics token
        self.phys_norm = nn.LayerNorm(n_physics)
        self.phys_mlp = nn.Sequential(nn.Linear(n_physics, 64), nn.ReLU(), nn.Linear(64, 128))
        # concat+FC fusion (deep 512 + physics 128 -> 512 eval layer)
        self.fuse = nn.Linear(512 + 128, 512)
        self.out_norm = nn.LayerNorm(512)
        # 128-D projection head
        self.projection_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
        self.branch_dropout = branch_dropout
        self.p = p
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=0.01)
        nn.init.constant_(self.cross_attn.out_proj.bias, 0.0)
        nn.init.normal_(self.projection_head[-1].weight, std=0.01)
        nn.init.constant_(self.projection_head[-1].bias, 0.0)

    def _deep(self, x_time, x_spec):
        t = self.time_branch(x_time)                      # [B,256]
        s = self.spectral_branch(x_spec)                  # [B,256]
        attn_out, _ = self.cross_attn(t.unsqueeze(1), s.unsqueeze(1), s.unsqueeze(1))
        attn_out = attn_out.squeeze(1) + t                # residual (prevents init collapse)
        return self.deep_norm(torch.cat([attn_out, s], dim=1))   # [B,512]

    def _fuse(self, deep, physics):
        pt = self.phys_mlp(self.phys_norm(physics))       # [B,128]
        if self.training and self.branch_dropout:
            B = deep.shape[0]
            u = torch.rand(B, device=deep.device)
            drop_deep = (u < self.p).float().unsqueeze(1)             # physics-only
            drop_phys = ((u >= self.p) & (u < 2 * self.p)).float().unsqueeze(1)  # deep-only
            deep = deep * (1.0 - drop_deep)
            pt = pt * (1.0 - drop_phys)
        return self.out_norm(self.fuse(torch.cat([deep, pt], dim=1)))    # [B,512]

    def get_encoder_output(self, x_time, x_spec, physics):
        """512-D pre-projection (the eval layer)."""
        return self._fuse(self._deep(x_time, x_spec), physics)

    def forward(self, x_time, x_spec, physics):
        """128-D L2-normalized embedding (training head)."""
        h = self._fuse(self._deep(x_time, x_spec), physics)
        return F.normalize(self.projection_head(h), dim=1, eps=1e-6)


def param_count(model):
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, trn
