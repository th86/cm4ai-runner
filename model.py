"""U-Net with per-protein FiLM conditioning, shared by training (train.py) and the submission (predict.py)."""
import torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)

    def forward(self, x, film=None):
        x = F.silu(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        if film is not None:  # FiLM: protein embedding scales and shifts features
            g, b = film.chunk(2, 1)
            x = x * (1 + g[..., None, None]) + b[..., None, None]
        return F.silu(x)


class UNet(nn.Module):
    def __init__(self, n_prot, cin=4, w=32, emb=64, depth=4):
        super().__init__()
        chs = [w * 2 ** i for i in range(depth)]  # e.g. 32,64,128,256
        self.emb = nn.Embedding(n_prot, emb)
        self.dom = nn.Embedding(2, emb)  # 0 = KOLF2.1J (target domain), 1 = HPA
        self.enc = nn.ModuleList()
        c = cin
        for ch in chs:
            self.enc.append(Block(c, ch)); c = ch
        self.mid = Block(c, c)
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        for ch in reversed(chs):
            self.up.append(nn.Conv2d(c, ch, 1))
            self.dec.append(Block(2 * ch, ch)); c = ch
        self.films = nn.ModuleList([nn.Linear(emb, 2 * chs[-1])] +
                                   [nn.Linear(emb, 2 * ch) for ch in reversed(chs)])
        for f in self.films:
            nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x, ab, dom=None, e=None):
        e = self.emb(ab) if e is None else e
        if dom is not None:
            e = e + self.dom(dom)
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x); skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x, self.films[0](e))
        for i, (up, blk) in enumerate(zip(self.up, self.dec)):
            x = up(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
            x = blk(torch.cat([x, skips.pop()], 1), self.films[i + 1](e))
        return self.head(x)[:, 0]


def dihedral(x, k):
    """k in 0..7: rot90 k%4 times, then optional flip. Applies to last two dims."""
    x = torch.rot90(x, k % 4, (-2, -1))
    return torch.flip(x, (-1,)) if k >= 4 else x


def inv_dihedral(x, k):
    if k >= 4:
        x = torch.flip(x, (-1,))
    return torch.rot90(x, -(k % 4), (-2, -1))
