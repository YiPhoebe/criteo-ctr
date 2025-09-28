import torch
import torch.nn as nn


class DeepFM(nn.Module):
    """
    DeepFM with explicit FM 2nd-order interactions.
    - First-order: linear over dense + 1-dim embeddings for cats
    - Second-order: factorized interaction using per-dense feature factors + cat embeddings
    - Deep part: MLP over [dense, concat(cat_embs)]
    Returns logits (no sigmoid).
    """

    def __init__(self, d_dense: int, bucket_sizes: list[int], emb_dim: int = 16, mlp: list[int] = [256, 128, 64], dropout: float = 0.0):
        super().__init__()
        self.d_dense = d_dense
        self.n_cats = len(bucket_sizes)
        self.emb_dim = emb_dim

        # Embeddings for categorical features (for FM 2nd-order and deep part)
        self.embs = nn.ModuleList([nn.Embedding(b, emb_dim) for b in bucket_sizes])

        # First-order terms: dense linear + cat 1-dim embeddings
        self.linear_dense = nn.Linear(d_dense, 1)
        self.linear_cats = nn.ModuleList([nn.Embedding(b, 1) for b in bucket_sizes])

        # Factorization vectors for dense features (for FM 2nd-order)
        self.v_dense = nn.Parameter(torch.randn(dense_dim := d_dense, emb_dim) * 0.01)

        # Deep part over concatenated [dense, cat_embs]
        deep_in = d_dense + self.n_cats * emb_dim
        layers: list[nn.Module] = []
        prev = deep_in
        for h in mlp:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, dense: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        B = dense.size(0)

        # Cat embeddings list [B, emb_dim]
        e_list = [emb(cats[:, i]) for i, emb in enumerate(self.embs)]
        if e_list:
            e_cat = torch.stack(e_list, dim=1)  # [B, n_cats, K]
        else:
            e_cat = torch.zeros((B, 0, self.emb_dim), device=dense.device, dtype=dense.dtype)

        # First-order
        y_linear = self.linear_dense(dense)  # [B,1]
        if self.n_cats:
            y_linear += sum(emb(cats[:, i]) for i, emb in enumerate(self.linear_cats))  # [B,1]

        # FM 2nd order
        # Dense contribution: x_i * v_i -> [B, d_dense, K]
        if self.d_dense:
            vx_dense = dense.unsqueeze(2) * self.v_dense.unsqueeze(0)  # [B, D, K]
            sum_vx_dense = vx_dense.sum(dim=1)  # [B,K]
            sum_sq_dense = (vx_dense.pow(2)).sum(dim=1)  # [B,K]
        else:
            sum_vx_dense = torch.zeros((B, self.emb_dim), device=dense.device, dtype=dense.dtype)
            sum_sq_dense = torch.zeros_like(sum_vx_dense)

        if self.n_cats:
            sum_vx_cat = e_cat.sum(dim=1)  # [B,K]
            sum_sq_cat = (e_cat.pow(2)).sum(dim=1)  # [B,K]
        else:
            sum_vx_cat = torch.zeros((B, self.emb_dim), device=dense.device, dtype=dense.dtype)
            sum_sq_cat = torch.zeros_like(sum_vx_cat)

        sum_vx = sum_vx_dense + sum_vx_cat  # [B,K]
        sum_sq = sum_sq_dense + sum_sq_cat   # [B,K]
        fm_second = 0.5 * (sum_vx.pow(2) - sum_sq).sum(dim=1, keepdim=True)  # [B,1]

        # Deep part on concatenated features
        x_deep = torch.cat([dense] + e_list, dim=1) if e_list else dense
        y_deep = self.mlp(x_deep)  # [B,1]

        logit = y_linear + fm_second + y_deep
        return logit
