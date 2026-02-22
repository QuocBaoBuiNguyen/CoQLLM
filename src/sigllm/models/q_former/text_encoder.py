import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, text_list, device, max_len=64):
        tokens = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        out = self.model(**tokens, return_dict=True)
        h = out.last_hidden_state
        mask = tokens.attention_mask

        input_mask_expanded = mask.unsqueeze(-1).expand(h.size()).float()
        sum_embeddings = torch.sum(h * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        return h, pooled