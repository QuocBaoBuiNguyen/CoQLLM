import torch
import torch.nn as nn
from torch.nn import Parameter


class QFormer(nn.Module):
    """Align CF embeddings via learned queries and cross-attention. Instruction-conditioned Q-Former adapter."""

    def __init__(self, d_cf: int, d_model: int, num_queries: int = 8, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.q = Parameter(torch.randn(num_queries, d_model))
        self.proj_cf = nn.Linear(d_cf, d_model)

        self.attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
            for _ in range(num_layers)
        ])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, cf_vec: torch.Tensor, ins_token_emb: torch.Tensor) -> torch.Tensor:
        B = cf_vec.size(0)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        cf_tok = self.proj_cf(cf_vec).unsqueeze(1)
        kv = torch.cat([ins_token_emb, cf_tok], dim=1)

        x = q
        for attn, ffn, ln1, ln2 in zip(self.attn, self.ffn, self.ln1, self.ln2):
            y, _ = attn(x, kv, kv)
            x = ln1(x + y)
            z = ffn(x)
            x = ln2(x + z)
        return x
