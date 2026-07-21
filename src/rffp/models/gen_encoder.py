"""rffp.models.gen_encoder — GenRFEncoder: physics-residual front end + physics injection.

Reuses the proven dual-branch family (1D ResNet time + 2D ResNet spectral + cross-attention,
GroupNorm, GAP). Full recipe:
  * TIME branch input = 4 channels [I, Q, I_res, Q_res]
  * SPECTRAL branch input = 4 channels [|STFT(iq)| 2ch, |STFT(res)| 2ch]
  * PHYSICS token = 19-D -> LayerNorm -> MLP(19->64->128)
  * FUSION = cross-attention(time<->spectral, parent recipe) then CONCAT+FC to inject physics
  * FORCED-USAGE branch-dropout (p each) deep-only / physics-only, mutually exclusive.

ABLATION TOGGLES (defaults reproduce the full recipe EXACTLY):
  use_residual=False -> 2-ch time/spectral stems (raw channels only; A2)
  use_physics=False  -> physics token removed, fusion = out_norm(deep), no branch-dropout (A1)

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
from rffp.models import encoder as _parent          # parent model.py — GN, ResBlock1D/2D (import-safe, class defs only)


class TimeBranch(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        GN = _parent.GN
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 63, padding=31, bias=False), GN(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1, bias=False), GN(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1, bias=False), GN(64), nn.ReLU())
        self.b1 = _parent.ResBlock1D(64, 64)
        self.b2 = _parent.ResBlock1D(64, 64)
        self.b3 = _parent.ResBlock1D(64, 128)
        self.b4 = _parent.ResBlock1D(128, 128)
        self.proj = nn.Linear(128, 256)

    def forward(self, x):                      # [B, C, L]
        x = self.stem(x)
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.proj(x.mean(2))


class SpectralBranch(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        GN = _parent.GN
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False), GN(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=(2, 1), padding=1, bias=False), GN(64), nn.ReLU())
        self.b1 = _parent.ResBlock2D(64, 64)
        self.b2 = _parent.ResBlock2D(64, 64)
        self.b3 = _parent.ResBlock2D(64, 128)
        self.b4 = _parent.ResBlock2D(128, 128)
        self.proj = nn.Linear(128, 256)

    def forward(self, x):                      # [B, C, F, T]
        x = self.stem(x)
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.proj(x.mean([2, 3]))


class GenRFEncoder(nn.Module):
    def __init__(self, n_physics=19, branch_dropout=True, p=0.15,
                 use_physics=True, use_residual=True):
        super().__init__()
        self.use_physics = use_physics
        self.use_residual = use_residual
        in_ch = 4 if use_residual else 2
        self.time_branch = TimeBranch(in_ch)
        self.spectral_branch = SpectralBranch(in_ch)
        self.cross_attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.deep_norm = nn.LayerNorm(512)
        if use_physics:
            self.phys_norm = nn.LayerNorm(n_physics)
            self.phys_mlp = nn.Sequential(nn.Linear(n_physics, 64), nn.ReLU(), nn.Linear(64, 128))
            self.fuse = nn.Linear(512 + 128, 512)
        self.out_norm = nn.LayerNorm(512)
        self.projection_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
        # branch-dropout only meaningful when both pathways exist
        self.branch_dropout = branch_dropout and use_physics
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
        t = self.time_branch(x_time)
        s = self.spectral_branch(x_spec)
        attn_out, _ = self.cross_attn(t.unsqueeze(1), s.unsqueeze(1), s.unsqueeze(1))
        attn_out = attn_out.squeeze(1) + t
        return self.deep_norm(torch.cat([attn_out, s], dim=1))          # [B,512]

    def _fuse(self, deep, physics):
        if not self.use_physics:
            return self.out_norm(deep)
        pt = self.phys_mlp(self.phys_norm(physics))                     # [B,128]
        if self.training and self.branch_dropout:
            B = deep.shape[0]
            u = torch.rand(B, device=deep.device)
            drop_deep = (u < self.p).float().unsqueeze(1)
            drop_phys = ((u >= self.p) & (u < 2 * self.p)).float().unsqueeze(1)
            deep = deep * (1.0 - drop_deep)
            pt = pt * (1.0 - drop_phys)
        return self.out_norm(self.fuse(torch.cat([deep, pt], dim=1)))   # [B,512]

    def get_encoder_output(self, x_time, x_spec, physics=None):
        return self._fuse(self._deep(x_time, x_spec), physics)

    def forward(self, x_time, x_spec, physics=None):
        h = self._fuse(self._deep(x_time, x_spec), physics)
        return F.normalize(self.projection_head(h), dim=1, eps=1e-6)


def param_count(model):
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, trn
