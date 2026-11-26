"""Compact FuXi-style U-Transformer for a 32x64 (5.625 deg) global grid.

Architecture (follows FuXi's design, scaled down for 2 CPU cores):
  * cube embedding: the two input frames (2 x 70 channels) are embedded jointly with a
    space-time "cube" convolution (kernel 2 x p x p, implemented as a Conv2d over the
    stacked frames), plus forcing/static channels -> tokens on a (32/p) x (64/p) grid
  * U-Transformer: down-sampling residual conv block (stride 2), a stack of Swin-V2
    blocks (post-norm, scaled-cosine attention, log-spaced continuous relative position
    bias, longitude-periodic shifted windows), an up-sampling block with a skip connection
  * head: transposed-conv up-sampling back to full resolution, concatenated with the last
    input frame, followed by two 3x3 convolutions (longitude-circular padding) -> 70
    channel residual increment.  (A conv head instead of a linear pixel-shuffle head avoids
    blocky patch artefacts.)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CircConv(nn.Module):
    """3x3 conv with circular padding in longitude and replicate padding in latitude."""
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.p, self.p, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, self.p, self.p), mode='replicate')
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c); self.c1 = CircConv(c, c)
        self.n2 = nn.GroupNorm(8, c); self.c2 = CircConv(c, c)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class WindowAttentionV2(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        self.heads, self.window = heads, window
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(heads, 1, 1)))
        self.cpb = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, heads, bias=False))
        wh, ww = window
        rh = torch.arange(-(wh - 1), wh, dtype=torch.float32)
        rw = torch.arange(-(ww - 1), ww, dtype=torch.float32)
        tbl = torch.stack(torch.meshgrid(rh, rw, indexing='ij'), -1)
        tbl[..., 0] /= max(wh - 1, 1); tbl[..., 1] /= max(ww - 1, 1)
        tbl *= 8
        tbl = torch.sign(tbl) * torch.log2(tbl.abs() + 1.0) / math.log2(8)
        self.register_buffer('tbl', tbl.reshape(-1, 2), persistent=False)
        coords = torch.stack(torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing='ij')).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
        idx = (rel[..., 0] + wh - 1) * (2 * ww - 1) + (rel[..., 1] + ww - 1)
        self.register_buffer('idx', idx, persistent=False)

    def forward(self, x):  # x: (B*nW, N, C)
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn * torch.clamp(self.logit_scale, max=math.log(100.)).exp()
        bias = self.cpb(self.tbl)[self.idx.reshape(-1)].reshape(N, N, -1).permute(2, 0, 1)
        attn = attn + 16 * torch.sigmoid(bias)[None]
        x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SwinV2Block(nn.Module):
    def __init__(self, dim, heads, window, shift):
        super().__init__()
        self.window, self.shift = window, shift
        self.attn = WindowAttentionV2(dim, heads, window)
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):  # x: (B, H, W, C)
        B, H, W, C = x.shape
        wh, ww = self.window
        h = torch.roll(x, -self.shift, dims=2) if self.shift else x   # longitude-periodic shift
        h = h.reshape(B, H // wh, wh, W // ww, ww, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, wh * ww, C)
        h = self.attn(h)
        h = h.reshape(B, H // wh, W // ww, wh, ww, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        if self.shift: h = torch.roll(h, self.shift, dims=2)
        x = x + self.n1(h)                        # Swin-V2 res-post-norm
        x = x + self.n2(self.mlp(x))
        return x


class UTransformer(nn.Module):
    def __init__(self, nch=70, nforce=7, dim=192, depth=8, heads=6, patch=2, window=(8, 8), H=32, W=64, hw=96):
        super().__init__()
        self.nch = nch
        self.embed = nn.Conv2d(2 * nch + nforce, dim, patch, stride=patch)   # cube embedding (2 x p x p)
        self.enorm = nn.LayerNorm(dim)
        self.down = nn.Sequential(CircConv(dim, dim, 3, stride=1), nn.GroupNorm(8, dim), nn.SiLU())
        self.down_s = nn.Conv2d(dim, dim, 2, stride=2)
        self.down_res = ResBlock(dim)
        self.blocks = nn.ModuleList([SwinV2Block(dim, heads, window, 0 if i % 2 == 0 else window[1] // 2)
                                     for i in range(depth)])
        self.up = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.up_fuse = nn.Sequential(nn.Conv2d(2 * dim, dim, 1), ResBlock(dim))
        self.head_up = nn.ConvTranspose2d(dim, hw, patch, stride=patch)
        self.head = nn.Sequential(CircConv(hw + nch + nforce, hw), nn.SiLU(), CircConv(hw, nch))
        nn.init.zeros_(self.head[-1].conv.weight); nn.init.zeros_(self.head[-1].conv.bias)

    def forward(self, x0, x1, force):
        """x0: state at t-6h, x1: state at t (normalised, B x 70 x 32 x 64); force: B x nforce x 32 x 64.
        Returns the normalised increment (to be scaled by the tendency std)."""
        z = self.embed(torch.cat([x0, x1, force], 1))
        z = self.enorm(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skip = z
        d = self.down_res(self.down_s(self.down(z)))
        t = d.permute(0, 2, 3, 1)
        for b in self.blocks:
            t = b(t)
        u = self.up(t.permute(0, 3, 1, 2))
        u = self.up_fuse(torch.cat([u, skip], 1))
        h = F.silu(self.head_up(u))
        return self.head(torch.cat([h, x1, force], 1))
