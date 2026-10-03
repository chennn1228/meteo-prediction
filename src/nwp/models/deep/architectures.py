"""Registered deep architectures for sequence-to-seven-quantile prediction."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, c_in, out_dim=7, seq_len=168):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_len * c_in, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim))

    def forward(self, x):
        return self.net(x)


class LSTM(nn.Module):
    def __init__(self, c_in, out_dim=7, seq_len=168):
        super().__init__()
        del seq_len
        self.rnn = nn.LSTM(c_in, 64, num_layers=2, batch_first=True,
                           dropout=0.1)
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        hidden, _ = self.rnn(x)
        return self.head(hidden[:, -1])


class CNN(nn.Module):
    def __init__(self, c_in, out_dim=7, seq_len=168):
        super().__init__()
        del seq_len
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(128, out_dim)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2)).squeeze(-1))


class TCN(nn.Module):
    def __init__(self, c_in, out_dim=7, seq_len=168):
        super().__init__()
        del seq_len
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 64, 2, padding=1), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(64, 64, 2, padding=2, dilation=2), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 64, 2, padding=4, dilation=4), nn.ReLU())
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2))[:, :, -1])


class Transformer(nn.Module):
    def __init__(self, c_in, out_dim=7, seq_len=168, n_heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(c_in, 64)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, 64))
        layer = nn.TransformerEncoderLayer(
            64, n_heads, 128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.head(self.encoder(self.proj(x) + self.pos).mean(1))


class DLinear(nn.Module):
    def __init__(self, c_in, out_dim=7, seq_len=168):
        super().__init__()
        self.trend = nn.Linear(seq_len, 1)
        self.season = nn.Linear(seq_len, 1)
        self.mix = nn.Linear(c_in, out_dim)

    def forward(self, x):
        values = x.transpose(1, 2)
        residual = values - F.avg_pool1d(values, 5, 1, 2)
        return self.mix(self.trend(residual).squeeze(-1)
                        + self.season(values - residual).squeeze(-1))


class TSMixer(nn.Module):
    def __init__(self, c_in, out_dim=7, seq_len=168):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(seq_len, 32), nn.ReLU(),
                                  nn.Linear(32, seq_len))
        self.feature = nn.Sequential(nn.Linear(c_in, 64), nn.ReLU(),
                                     nn.Linear(64, c_in))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_len * c_in, 128), nn.ReLU(),
            nn.Linear(128, out_dim))

    def forward(self, x):
        x = x + self.time(x.transpose(1, 2)).transpose(1, 2)
        return self.head(x + self.feature(x))


class ITransformer(nn.Module):
    def __init__(self, c_in, out_dim=7, seq_len=168):
        super().__init__()
        self.proj = nn.Linear(seq_len, 64)
        layer = nn.TransformerEncoderLayer(
            64, 4, 128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.head(self.encoder(self.proj(x.transpose(1, 2))).mean(1))


class SeriesDecomp(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        pad = self.kernel // 2
        trend = F.avg_pool1d(
            F.pad(x.transpose(1, 2), (pad, pad), mode="replicate"),
            self.kernel, 1).transpose(1, 2)
        return x - trend, trend


def aggregate_lags(values, lags, weights):
    """Aggregate independent top-k circular lags without collapsing them."""
    batch, heads, length, channels = values.shape
    if lags.shape != weights.shape or lags.shape[0] != batch:
        raise ValueError("lag/weight shape mismatch")
    positions = torch.arange(length, device=values.device).view(1, length)
    result = torch.zeros_like(values)
    for index in range(lags.shape[1]):
        shifted = (positions - lags[:, index].view(batch, 1)) % length
        gather = shifted.view(batch, 1, length, 1).expand(
            batch, heads, length, channels)
        result = result + weights[:, index].view(
            batch, 1, 1, 1) * torch.gather(values, 2, gather)
    return result


class AutoCorrelation(nn.Module):
    def __init__(self, d_model, n_heads, topk=5):
        super().__init__()
        self.heads, self.topk = n_heads, topk
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        query, key, values = qkv[0], qkv[1], qkv[2]
        correlation = torch.fft.irfft(
            torch.fft.rfft(query, dim=2)
            * torch.conj(torch.fft.rfft(key, dim=2)), n=length,
            dim=2).mean(dim=(1, 3))
        weights, lags = torch.topk(correlation, min(self.topk, length), dim=-1)
        result = aggregate_lags(values, lags, torch.softmax(weights, dim=-1))
        return self.out(result.permute(0, 2, 1, 3).reshape(
            batch, length, width))


class AutoformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn=128, dropout=0.1, kernel=25):
        super().__init__()
        self.decomp1, self.decomp2 = SeriesDecomp(kernel), SeriesDecomp(kernel)
        self.attention = AutoCorrelation(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, ffn), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x):
        x, _ = self.decomp1(
            self.norm1(x + self.dropout(self.attention(x))))
        x, _ = self.decomp2(
            self.norm2(x + self.dropout(self.feedforward(x))))
        return x


class Autoformer(nn.Module):
    def __init__(self, c_in, out_dim=7, d_model=64, layers=2,
                 n_heads=4, seq_len=168):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.layers = nn.ModuleList(
            [AutoformerLayer(d_model, n_heads) for _ in range(layers)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        encoded = self.proj(x) + self.pos
        for layer in self.layers:
            encoded = layer(encoded)
        return self.head(encoded.mean(1))


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.heads, self.factor = n_heads, factor
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        query, key, values = qkv[0], qkv[1], qkv[2]
        count = min(length, max(1, int(self.factor * math.log(max(length, 2)))))
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            width // self.heads)
        indices = (score.max(-1).values - score.mean(-1)).topk(
            count, dim=-1).indices
        selected = torch.gather(
            query, 2, indices.unsqueeze(-1).expand(-1, -1, -1, query.size(-1)))
        attention = torch.softmax(
            torch.matmul(selected, key.transpose(-2, -1))
            / math.sqrt(width // self.heads), dim=-1)
        sparse = torch.matmul(attention, values)
        full = torch.zeros_like(query)
        full.scatter_(2, indices.unsqueeze(-1).expand_as(sparse), sparse)
        return self.out(full.permute(0, 2, 1, 3).reshape(batch, length, width))


class Distill(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        return F.max_pool1d(
            F.gelu(self.norm(self.conv(x.transpose(1, 2)))), 2, 2
        ).transpose(1, 2)


class Informer(nn.Module):
    def __init__(self, c_in, out_dim=7, d_model=64, layers=2,
                 n_heads=4, seq_len=168):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.attentions = nn.ModuleList(
            [ProbSparseAttention(d_model, n_heads) for _ in range(layers)])
        self.feedforwards = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, d_model))
            for _ in range(layers)])
        self.distills = nn.ModuleList(
            [Distill(d_model) for _ in range(max(0, layers - 1))])
        self.norm, self.head = nn.LayerNorm(d_model), nn.Linear(d_model, out_dim)

    def forward(self, x):
        encoded = self.proj(x) + self.pos
        for index, (attention, feedforward) in enumerate(
                zip(self.attentions, self.feedforwards)):
            encoded = self.norm(encoded + attention(encoded))
            encoded = self.norm(encoded + feedforward(encoded))
            if index < len(self.distills):
                encoded = self.distills[index](encoded)
        return self.head(encoded.mean(1))


class FrequencyBlock(nn.Module):
    def __init__(self, d_model, modes=8):
        super().__init__()
        self.modes = modes
        self.scale = nn.Parameter(torch.randn(modes, d_model, 2) * 0.02)

    def forward(self, x):
        frequency = torch.fft.rfft(x, dim=1)
        count = min(self.modes, frequency.size(1))
        result = torch.zeros_like(frequency)
        result[:, :count] = (torch.view_as_complex(
            self.scale[:count].unsqueeze(0)) * frequency[:, :count])
        return torch.fft.irfft(result, n=x.size(1), dim=1)


class FEDformer(nn.Module):
    def __init__(self, c_in, out_dim=7, d_model=64, layers=2, seq_len=168):
        super().__init__()
        self.proj = nn.Linear(c_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList(
            [FrequencyBlock(d_model) for _ in range(layers)])
        self.decomp = SeriesDecomp()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, d_model))
        self.norm, self.head = nn.LayerNorm(d_model), nn.Linear(d_model, out_dim)

    def forward(self, x):
        encoded = self.proj(x) + self.pos
        for block in self.blocks:
            encoded, _ = self.decomp(self.norm(encoded + block(encoded)))
            encoded = self.norm(encoded + self.feedforward(encoded))
        return self.head(encoded.mean(1))


class PatchTST(nn.Module):
    def __init__(self, c_in, out_dim=7, patch=16, stride=8, d_model=128,
                 layers=3, n_heads=8, seq_len=168):
        super().__init__()
        del seq_len
        self.patch, self.stride = patch, stride
        self.proj = nn.Linear(patch, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model * c_in, out_dim)

    def forward(self, x):
        batch, length, channels = x.shape
        count = (length - self.patch) // self.stride + 1
        patches = x.unfold(1, self.patch, self.stride).permute(0, 2, 1, 3)
        encoded = self.encoder(
            self.proj(patches).reshape(batch * channels, count, -1))
        return self.head(encoded.mean(1).reshape(batch, -1))


class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(c_in, c_out, kernel, padding=kernel // 2)
            for kernel in (1, 3, 5)])
        self.out = nn.Conv2d(c_out * 3, c_out, 1)

    def forward(self, x):
        return self.out(torch.cat([branch(x) for branch in self.branches], dim=1))


class TimesNet(nn.Module):
    def __init__(self, c_in, out_dim=7, k_periods=3, d_model=32,
                 seq_len=168):
        super().__init__()
        del d_model, seq_len
        self.periods, self.channels = k_periods, c_in
        self.inception = InceptionBlock(1, 16)
        self.head = nn.Linear(c_in, out_dim)

    def forward(self, x):
        batch, length, channels = x.shape
        frequency = torch.fft.rfft(x.mean(-1), dim=1).abs()
        frequency[:, 0] = 0
        count = min(self.periods, frequency.size(1) - 1)
        periods = torch.topk(frequency, count, dim=1).indices + 1
        outputs = []
        for index in range(count):
            period = max(2, int(periods[:, index].float().mean().item()))
            blocks = length // period
            if blocks < 2:
                continue
            grid = x[:, :blocks * period].reshape(
                batch, blocks, period, channels).permute(0, 3, 2, 1)
            encoded = self.inception(
                grid.reshape(batch * channels, 1, period, blocks))
            outputs.append(encoded.mean((-1, -2)).reshape(batch, channels, -1))
        pooled = (torch.stack(outputs).mean(0).mean(-1)
                  if outputs else x.mean(1))
        return self.head(pooled)


class PINN(nn.Module):
    def __init__(self, c_in, out_dim=7, hidden=256, seq_len=168):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_len * c_in, hidden), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden, 128), nn.GELU(),
            nn.Linear(128, out_dim))

    def forward(self, x):
        prediction = self.net(x)
        # Physical-unit constraints remain disabled until inverse scaling and
        # site-adaptation tolerances are registered and validated.
        return prediction, prediction.new_zeros(())


ARCHITECTURES = {
    "mlp": MLP, "cnn": CNN, "tcn": TCN, "lstm": LSTM,
    "transformer": Transformer, "autoformer": Autoformer,
    "informer": Informer, "fedformer": FEDformer,
    "itransformer": ITransformer, "patchtst": PatchTST,
    "dlinear": DLinear, "timesnet": TimesNet, "tsmixer": TSMixer,
    "pinn": PINN,
}


def build_deep_model(name: str, c_in: int, *, seq_len: int = 168,
                     out_dim: int = 7):
    try:
        architecture = ARCHITECTURES[name]
    except KeyError as exc:
        raise ValueError(f"unknown deep architecture: {name}") from exc
    return architecture(c_in, out_dim=out_dim, seq_len=seq_len)


# Backward-neutral name for the formal architecture regression test.
build_formal_model = build_deep_model
