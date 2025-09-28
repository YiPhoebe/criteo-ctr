import torch
import torch.nn as nn


class DLRMmini(nn.Module):
    def __init__(
        self,
        d_dense: int,
        bucket_sizes: list[int],
        emb_dim: int = 16,
        bot: list[int] = [256, 128],
        top: list[int] = [256, 128, 1],
    ):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(b, emb_dim) for b in bucket_sizes])
        self.bottom = nn.Sequential(
            nn.Linear(d_dense, bot[0]),
            nn.ReLU(),
            nn.Linear(bot[0], bot[1]),
            nn.ReLU(),
        )
        layers = []
        in_dim = bot[-1] + emb_dim * len(bucket_sizes)
        for i in range(len(top) - 1):
            layers += [nn.Linear(in_dim if i == 0 else top[i - 1], top[i]), nn.ReLU()]
        layers += [nn.Linear(top[-2], top[-1])]
        self.top = nn.Sequential(*layers)

    def forward(self, dense: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        z = self.bottom(dense)
        e = torch.cat([emb(cats[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        x = torch.cat([z, e], dim=1)
        logit = self.top(x)
        return logit
