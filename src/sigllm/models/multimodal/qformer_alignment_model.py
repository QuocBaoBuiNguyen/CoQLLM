import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM

from sigllm.models.rec.matrix_factorization import MatrixFactorization
from sigllm.models.q_former.q_former import QFormer


class QFormerAlignmentModel(nn.Module):
    """CF + shared Q-Former + text encoder for alignment."""

    def __init__(
        self,
        mf_config,
        d_model: int,
        num_queries: int = 8,
        num_heads: int = 8,
        llama_model: str = "",
    ) -> None:
        super().__init__()
        self.cf = MatrixFactorization(mf_config)
        d_cf = mf_config.embedding_size
        self.qformer = QFormer(d_cf=d_cf, d_model=d_model, num_queries=num_queries, num_heads=num_heads)

        self.tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text_encoder = LlamaForCausalLM.from_pretrained(llama_model)

        self.p_user = nn.Linear(d_model, d_model)
        self.p_item = nn.Linear(d_model, d_model)
        self.p_text = nn.Linear(self.text_encoder.config.hidden_size, d_model)

    def encode_user(self, u_idx: torch.Tensor) -> torch.Tensor:
        cf_vec = self.cf.user_encoder(u_idx)
        q_tokens = self.qformer(cf_vec)
        pooled = q_tokens.mean(dim=1)
        return self.p_user(pooled)

    def encode_item(self, i_idx: torch.Tensor) -> torch.Tensor:
        cf_vec = self.cf.item_encoder(i_idx)
        q_tokens = self.qformer(cf_vec)
        pooled = q_tokens.mean(dim=1)
        return self.p_item(pooled)

    def encode_text(self, text_list: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.p_text.weight.device)
        embeds = self.text_encoder.get_input_embeddings()(tokens.input_ids)
        mask = tokens.attention_mask.unsqueeze(-1)
        summed = (embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        return self.p_text(pooled)

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor, text_list: list[str]):
        user_z = self.encode_user(u_idx)
        item_z = self.encode_item(i_idx)
        text_z = self.encode_text(text_list)
        return {
            "user": user_z,
            "item": item_z,
            "text": text_z,
        }
