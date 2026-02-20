import torch
import torch.nn as nn
from torch.nn import Parameter


class QFormer(nn.Module):
    """Align CF embeddings via learned queries and cross-attention."""

    def __init__(self, d_cf: int, d_model: int, num_queries: int = 8, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.queries = Parameter(torch.randn(num_queries, d_model))
        self.proj_kv = nn.Linear(d_cf, d_model)
        self.xattn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, cf_vec: torch.Tensor) -> torch.Tensor:
        batch = cf_vec.size(0)
        q = self.queries.unsqueeze(0).expand(batch, -1, -1)
        kv = self.proj_kv(cf_vec).unsqueeze(1)
        out, _ = self.xattn(q, kv, kv)
        return out
