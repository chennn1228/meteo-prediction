"""正式深度结构：Autoformer / Informer / FEDformer / PatchTST / TimesNet / PINN。

输入统一为 (B, L, C)，L=168（正式），输出 (B, 7) 分位数。
实现保留各模型核心机制：序列分解、Auto-Correlation、ProbSparse、频率增强、
patch 编码、周期二维卷积和物理约束。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomp(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        pad = self.kernel // 2
        trend = F.avg_pool1d(F.pad(x.transpose(1, 2), (pad, pad), mode="replicate"),
                             self.kernel, 1).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    def __init__(self, d_model, n_heads, topk=5):
        super().__init__()
        self.h = n_heads
        self.topk = topk
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, l, d = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        qf = torch.fft.rfft(q, dim=2)
        kf = torch.fft.rfft(k, dim=2)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=l, dim=2).mean(dim=(1, 3))
        kk = min(self.topk, l)
        weights, lags = torch.topk(corr, kk, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        # 向量化移位：按样本取头平均 lag，用 gather 做循环移位，避免逐样本 python 循环与 GPU 同步。
        lag = lags.float().mean(dim=1).round().long()                 # (b,)
        ar = torch.arange(l, device=x.device).view(1, l)
        idx = (ar - lag.view(b, 1)) % l                              # (b, l)
        gather_idx = idx.view(b, 1, l, 1).expand(b, self.h, l, d // self.h)
        out = torch.zeros_like(v)
        for i in range(kk):
            shifted = torch.gather(v, 2, gather_idx)
            out = out + weights[:, i].view(b, 1, 1, 1) * shifted
        out = out.permute(0, 2, 1, 3).reshape(b, l, d)
        return self.out(out)


class AutoformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn=128, dropout=0.1, kernel=25):
        super().__init__()
        self.decomp1 = SeriesDecomp(kernel)
        self.decomp2 = SeriesDecomp(kernel)
        self.attn = AutoCorrelation(d_model, n_heads)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ffn, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        x, _ = self.decomp1(x)
        x = self.norm2(x + self.drop(self.ff(x)))
        x, _ = self.decomp2(x)
        return x


class AutoformerFormal(nn.Module):
    def __init__(self, c_in, out_dim=7, d_model=64, layers=2, n_heads=4, seq_len=168):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.layers = nn.ModuleList([AutoformerLayer(d_model, n_heads) for _ in range(layers)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        z = self.proj(x) + self.pos
        for layer in self.layers:
            z = layer(z)
        return self.head(z.mean(1))


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.h = n_heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, l, d = x.shape
        qkv = self.qkv(x).reshape(b, l, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        u = min(l, max(1, int(self.factor * math.log(max(l, 2)))))
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d // self.h)
        sparsity = score.max(-1).values - score.mean(-1)
        idx = sparsity.topk(u, dim=-1).indices
        q_sel = torch.gather(q, 2, idx.unsqueeze(-1).expand(-1, -1, -1, q.size(-1)))
        attn = torch.softmax(torch.matmul(q_sel, k.transpose(-2, -1)) / math.sqrt(d // self.h), dim=-1)
        out = torch.matmul(attn, v)
        full = torch.zeros_like(q)
        full.scatter_(2, idx.unsqueeze(-1).expand(-1, -1, -1, q.size(-1)), out)
        full = full.permute(0, 2, 1, 3).reshape(b, l, d)
        return self.out(full)


class Distill(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        z = F.gelu(self.norm(self.conv(x.transpose(1, 2))))
        z = F.max_pool1d(z, 2, 2)
        return z.transpose(1, 2)


class InformerFormal(nn.Module):
    def __init__(self, c_in, out_dim=7, d_model=64, layers=2, n_heads=4, seq_len=168):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.attns = nn.ModuleList([ProbSparseAttention(d_model, n_heads) for _ in range(layers)])
        self.ffs = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, d_model)) for _ in range(layers)])
        self.distills = nn.ModuleList([Distill(d_model) for _ in range(max(0, layers - 1))])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        z = self.proj(x) + self.pos
        for i, (attn, ff) in enumerate(zip(self.attns, self.ffs)):
            z = self.norm(z + attn(z))
            z = self.norm(z + ff(z))
            if i < len(self.distills):
                z = self.distills[i](z)
        return self.head(z.mean(1))


class FreqBlock(nn.Module):
    def __init__(self, d_model, modes=8):
        super().__init__()
        self.modes = modes
        self.scale = nn.Parameter(torch.randn(modes, d_model, 2) * 0.02)

    def forward(self, x):
        f = torch.fft.rfft(x, dim=1)
        m = min(self.modes, f.size(1))
        out = torch.zeros_like(f)
        out[:, :m] = torch.view_as_complex(self.scale[:m].unsqueeze(0)) * f[:, :m]
        return torch.fft.irfft(out, n=x.size(1), dim=1)


class FEDformerFormal(nn.Module):
    def __init__(self, c_in, out_dim=7, d_model=64, layers=2, seq_len=168):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([FreqBlock(d_model) for _ in range(layers)])
        self.decomp = SeriesDecomp()
        self.ff = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        z = self.proj(x) + self.pos
        for block in self.blocks:
            z = self.norm(z + block(z))
            z, _ = self.decomp(z)
            z = self.norm(z + self.ff(z))
        return self.head(z.mean(1))


class PatchTSTFormal(nn.Module):
    def __init__(self, c_in, out_dim=7, patch=16, stride=8, d_model=128,
                 layers=3, n_heads=8, seq_len=168):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.proj = nn.Linear(patch, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2,
                                         dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model * c_in, out_dim)

    def forward(self, x):
        b, l, c = x.shape
        n = (l - self.patch) // self.stride + 1
        z = x.unfold(1, self.patch, self.stride).permute(0, 2, 1, 3)
        z = self.proj(z).reshape(b * c, n, -1)
        z = self.enc(z).mean(1).reshape(b, c * self.proj.out_features)
        return self.head(z)


class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(c_in, c_out, k, padding=k // 2) for k in (1, 3, 5)])
        self.out = nn.Conv2d(c_out * 3, c_out, 1)

    def forward(self, x):
        return self.out(torch.cat([b(x) for b in self.branches], dim=1))


class TimesNetFormal(nn.Module):
    def __init__(self, c_in, out_dim=7, k_periods=3, d_model=32, seq_len=168):
        super().__init__()
        self.k = k_periods
        self.c = c_in
        self.inception = InceptionBlock(1, 16)
        self.head = nn.Linear(c_in, out_dim)

    def forward(self, x):
        b, l, c = x.shape
        f = torch.fft.rfft(x.mean(-1), dim=1).abs()
        f[:, 0] = 0
        k = min(self.k, f.size(1) - 1)
        periods = torch.topk(f, k, dim=1).indices + 1
        outs = []
        for p in range(k):
            period = int(periods[:, p].float().mean().item())
            period = max(2, period)
            n = l // period
            if n < 2:
                continue
            z = x[:, :n * period].reshape(b, n, period, c).permute(0, 3, 2, 1)
            z = self.inception(z.reshape(b * c, 1, period, n))
            outs.append(z.mean((-1, -2)).reshape(b, c, -1))
        if outs:
            z = torch.stack(outs, dim=0).mean(0).mean(-1)
        else:
            z = x.mean(1)
        return self.head(z)


class PINNFormal(nn.Module):
    def __init__(self, c_in, out_dim=7, hidden=256, seq_len=168):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * c_in, hidden), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, 128), nn.GELU(),
                                 nn.Linear(128, out_dim))
    def forward(self, x):
        pred = self.net(x)
        # Experimental constrained model: non-negativity only. The previous
        # clear-sky ceiling was dimensionally invalid after feature scaling and
        # scientifically unsupported where observed GHI exceeds Ineichen.
        pen = torch.relu(-pred).mean()
        return pred, pen


def build_formal_model(name, c_in, seq_len=168, out_dim=7):
    if name == "autoformer":
        return AutoformerFormal(c_in, out_dim, seq_len=seq_len)
    if name == "informer":
        return InformerFormal(c_in, out_dim, seq_len=seq_len)
    if name == "fedformer":
        return FEDformerFormal(c_in, out_dim, seq_len=seq_len)
    if name == "patchtst":
        return PatchTSTFormal(c_in, out_dim, seq_len=seq_len)
    if name == "timesnet":
        return TimesNetFormal(c_in, out_dim, seq_len=seq_len)
    if name == "pinn":
        return PINNFormal(c_in, out_dim, seq_len=seq_len)
    raise ValueError(name)
