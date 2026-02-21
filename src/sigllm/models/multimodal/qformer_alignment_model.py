import torch
import torch.nn as nn
import torch.nn.functional as F

class QRecInstructAlignmentModel(nn.Module):
    """Instruction-conditioned alignment with injected encoders."""

    def __init__(self, mf, qformer, llama_tokenizer, llama_model) -> None:
        super().__init__()
        self.mf = mf
        self.qformer = qformer
        self.tok = llama_tokenizer
        self.llama = llama_model

        d = llama_model.config.hidden_size
        self.p_user = nn.Linear(d, d)
        self.p_item = nn.Linear(d, d)
        self.p_text = nn.Linear(d, d)

        for p in self.llama.parameters():
            p.requires_grad = False

    def pool_queries(self, q_tokens: torch.Tensor) -> torch.Tensor:
        return q_tokens.mean(dim=1)

    def llama_embed_tokens_and_pool(self, tok, llama, text_list, device, max_len: int):
        tokens = tok(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)
        embeds = llama.get_input_embeddings()(tokens.input_ids)
        mask = tokens.attention_mask.unsqueeze(-1)
        summed = (embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        return embeds, pooled

    def ins_tokens(self, ins_list, device):
        ins_tok_emb, _ = self.llama_embed_tokens_and_pool(self.tok, self.llama, ins_list, device, max_len=48)
        return ins_tok_emb

    def text_vec(self, text_list, device):
        _, pooled = self.llama_embed_tokens_and_pool(self.tok, self.llama, text_list, device, max_len=64)
        return self.p_text(pooled)

    def enc_user(self, u_ids, ins_tok_emb):
        u_cf = self.mf.user_encoder(u_ids)
        u_q = self.qformer(u_cf, ins_tok_emb)
        return self.p_user(self.pool_queries(u_q))

    def enc_item(self, i_ids, ins_tok_emb):
        i_cf = self.mf.item_encoder(i_ids)
        i_q = self.qformer(i_cf, ins_tok_emb)
        return self.p_item(self.pool_queries(i_q))

    def forward(self, u_idx: torch.Tensor, i_idx: torch.Tensor, ins_list: list[str], text_list: list[str]):
        device = u_idx.device
        ins_tok_emb = self.ins_tokens(ins_list, device)
        user_z = self.enc_user(u_idx, ins_tok_emb)
        item_z = self.enc_item(i_idx, ins_tok_emb)
        text_z = self.text_vec(text_list, device)
        return {"user": user_z, "item": item_z, "text": text_z}

    @staticmethod
    def l2norm(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def loss_user_item(u_vec: torch.Tensor, i_pos_vec: torch.Tensor, i_neg_vecs: torch.Tensor, tau: float = 0.07):
        u = QRecInstructAlignmentModel.l2norm(u_vec)
        pos = QRecInstructAlignmentModel.l2norm(i_pos_vec)
        neg = QRecInstructAlignmentModel.l2norm(i_neg_vecs)

        pos_logit = (u * pos).sum(-1, keepdim=True) / tau
        neg_logit = (u.unsqueeze(1) * neg).sum(-1) / tau

        logits = torch.cat([pos_logit, neg_logit], dim=1)
        labels = torch.zeros(u.size(0), dtype=torch.long, device=u.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def loss_item_text(i_vec: torch.Tensor, t_vec: torch.Tensor, tau: float = 0.07):
        i = QRecInstructAlignmentModel.l2norm(i_vec)
        t = QRecInstructAlignmentModel.l2norm(t_vec)
        logits = (i @ t.T) / tau
        labels = torch.arange(i.size(0), device=i.device)
        return F.cross_entropy(logits, labels)
