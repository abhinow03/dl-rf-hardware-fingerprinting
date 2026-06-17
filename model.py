import torch
import torch.nn as nn
import torch.nn.functional as F

# Window size: 4096 samples
# STFT: FFT=256, hop=64, Hann -> [129, 61] per channel -> input [2, 129, 61]

def GN(c):
    g = min(8, c)
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)

class SE1D(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, max(1,c//r), bias=False), nn.ReLU(),
            nn.Linear(max(1,c//r), c, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x.mean(2)).unsqueeze(2)

class SE2D(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, max(1,c//r), bias=False), nn.ReLU(),
            nn.Linear(max(1,c//r), c, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x.mean([2,3])).unsqueeze(2).unsqueeze(3)

class ResBlock1D(nn.Module):
    def __init__(self, ic, oc, k=3):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv1d(ic, oc, k, padding=p, bias=False), GN(oc), nn.ReLU(),
            nn.Conv1d(oc, oc, k, padding=p, bias=False), GN(oc)
        )
        self.se = SE1D(oc)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential(
            nn.Conv1d(ic, oc, 1, bias=False), GN(oc)
        ) if ic != oc else nn.Identity()
    def forward(self, x):
        return self.relu(self.shortcut(x) + self.se(self.block(x)))

class ResBlock2D(nn.Module):
    def __init__(self, ic, oc, k=3):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(ic, oc, k, padding=p, bias=False), GN(oc), nn.ReLU(),
            nn.Conv2d(oc, oc, k, padding=p, bias=False), GN(oc)
        )
        self.se = SE2D(oc)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential(
            nn.Conv2d(ic, oc, 1, bias=False), GN(oc)
        ) if ic != oc else nn.Identity()
    def forward(self, x):
        return self.relu(self.shortcut(x) + self.se(self.block(x)))

class TimeBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Two strided stems to downsample 4096 -> 1024
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, 63, padding=31, bias=False), GN(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1, bias=False), GN(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1, bias=False), GN(64), nn.ReLU()
        )
        self.b1 = ResBlock1D(64, 64)
        self.b2 = ResBlock1D(64, 64)
        self.b3 = ResBlock1D(64, 128)
        self.b4 = ResBlock1D(128, 128)
        self.proj = nn.Linear(128, 256)
    def forward(self, x):          # [B, 2, 4096]
        x = self.stem(x)           # [B, 64, 1024]
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.proj(x.mean(2))

class SpectralBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [B, 2, 129, 61]  (FFT=256, hop=64, 4096 samples)
        self.stem = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1, bias=False), GN(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=(2,1), padding=1, bias=False), GN(64), nn.ReLU()
        )
        self.b1 = ResBlock2D(64, 64)
        self.b2 = ResBlock2D(64, 64)
        self.b3 = ResBlock2D(64, 128)
        self.b4 = ResBlock2D(128, 128)
        self.proj = nn.Linear(128, 256)
    def forward(self, x):          # [B, 2, 129, 61]
        x = self.stem(x)           # [B, 64, 65, 61]
        x = self.b4(self.b3(self.b2(self.b1(x))))
        return self.proj(x.mean([2,3]))

class RFEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_branch = TimeBranch()
        self.spectral_branch = SpectralBranch()
        self.cross_attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.norm = nn.LayerNorm(512)
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=0.01)
        nn.init.constant_(self.cross_attn.out_proj.bias, 0.0)
        nn.init.normal_(self.projection_head[-1].weight, std=0.01)
        nn.init.constant_(self.projection_head[-1].bias, 0.0)

    def forward(self, x_time, x_stft):
        t = self.time_branch(x_time)
        s = self.spectral_branch(x_stft)
        attn_out, _ = self.cross_attn(t.unsqueeze(1), s.unsqueeze(1), s.unsqueeze(1))
        attn_out = attn_out.squeeze(1) + t  # residual — prevents init collapse
        combined = self.norm(torch.cat([attn_out, s], dim=1))
        return F.normalize(self.projection_head(combined), dim=1, eps=1e-6)

    def get_encoder_output(self, x_time, x_stft):
        t = self.time_branch(x_time)
        s = self.spectral_branch(x_stft)
        attn_out, _ = self.cross_attn(t.unsqueeze(1), s.unsqueeze(1), s.unsqueeze(1))
        attn_out = attn_out.squeeze(1) + t
        return self.norm(torch.cat([attn_out, s], dim=1))
