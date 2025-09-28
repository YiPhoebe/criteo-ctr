import torch
import torch.nn as nn


class LogisticCTR(nn.Module):
    def __init__(
        self,
        d_dense: int,
        n_cats: int,
        emb_dim: int = 8,
        bucket_sizes: list[int] | None = None,
    ):
        super().__init__()
        bucket_sizes = bucket_sizes or [1_000_003] * n_cats
        self.embs = nn.ModuleList([nn.Embedding(b, emb_dim) for b in bucket_sizes])
        in_dim = d_dense + emb_dim * len(bucket_sizes)
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, dense: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        embs = [emb(cats[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat([dense] + embs, dim=1)
        logit = self.linear(x)
        return logit
