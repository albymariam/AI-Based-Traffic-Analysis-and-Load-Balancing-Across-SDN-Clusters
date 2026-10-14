"""Transformer and temporal-CNN multi-step utilisation predictors (Methodology 2).
Both predict a RESIDUAL over the last observed utilisation: out = clamp(u_last + delta, 0, 1)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBase(nn.Module):
    def __init__(self, u_idx):
        super().__init__()
        self.u_idx = u_idx
        self.register_buffer("lo_u", torch.zeros(1))
        self.register_buffer("rng_u", torch.ones(1))

    def set_norm(self, norm):
        self.lo_u.fill_(norm["lo"][self.u_idx])
        self.rng_u.fill_(norm["rng"][self.u_idx])

    def _finish(self, x, delta):
        u_last = x[:, -1, self.u_idx] * self.rng_u + self.lo_u      # de-normalised last utilisation
        return (u_last.unsqueeze(1) + delta).clamp(0.0, 1.0)


class TransformerPredictor(_ResidualBase):
    def __init__(self, n_feat, window, horizon, u_idx, d_model=32, n_heads=4, n_layers=2,
                 ff_dim=64, dropout=0.1):
        super().__init__(u_idx)
        self.inp = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x):                       # x: (B, W, F) normalised
        z = self.enc(self.inp(x) + self.pos)[:, -1]
        return self._finish(x, self.head(z))


class _TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, dil, p):
        super().__init__()
        self.pad = (k - 1) * dil
        self.c1 = nn.Conv1d(cin, cout, k, dilation=dil)
        self.c2 = nn.Conv1d(cout, cout, k, dilation=dil)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        y = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))     # causal (left) padding
        y = self.drop(F.relu(self.c2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.res(x))


class TCNPredictor(_ResidualBase):
    def __init__(self, n_feat, window, horizon, u_idx, channels=32, dropout=0.1, dilations=(1, 2, 4)):
        super().__init__(u_idx)
        blocks, cin = [], n_feat
        for d in dilations:
            blocks.append(_TCNBlock(cin, channels, 3, d, dropout))
            cin = channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, horizon)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))[:, :, -1]
        return self._finish(x, self.head(h))


def build_model(arch, n_feat, window, horizon, u_idx, pc):
    if arch == "transformer":
        return TransformerPredictor(n_feat, window, horizon, u_idx, pc["d_model"], pc["n_heads"],
                                    pc["n_layers"], pc["ff_dim"], pc["dropout"])
    if arch == "tcn":
        return TCNPredictor(n_feat, window, horizon, u_idx, pc["tcn_channels"], pc["dropout"])
    raise ValueError(arch)


if __name__ == "__main__":
    for a in ("transformer", "tcn"):
        m = build_model(a, 6, 10, 5, 2, dict(d_model=32, n_heads=4, n_layers=2, ff_dim=64, dropout=0.1,
                                             tcn_channels=32))
        print(a, tuple(m(torch.rand(8, 10, 6)).shape), sum(p.numel() for p in m.parameters()), "params")