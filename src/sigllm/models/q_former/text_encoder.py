import torch
import torch.nn as nn
from transformers import AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, d_model, model_name="bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.emb = nn.Embedding(self.tokenizer.vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, text_list, device, max_len=64):
        tokens = self.tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_len
        ).to(device)
        
        ids = tokens.input_ids
        mask = tokens.attention_mask # [B, L]
        
        x = self.emb(ids) # [B, L, D]
        
        # TransformerEncoder with batch_first=True
        # src_key_padding_mask: True for padded elements
        padding_mask = (mask == 0)
        
        h = self.enc(x, src_key_padding_mask=padding_mask) # [B, L, D]
        
        # Mean pooling
        input_mask_expanded = mask.unsqueeze(-1).expand(h.size()).float()
        sum_embeddings = torch.sum(h * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled = sum_embeddings / sum_mask # [B, D]
        
        return h, pooled

