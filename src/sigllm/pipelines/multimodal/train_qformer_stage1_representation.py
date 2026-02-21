import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
from transformers import LlamaTokenizer, LlamaForCausalLM

from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset
from sigllm.models.rec.matrix_factorization import MatrixFactorization
from sigllm.models.q_former.q_former import QFormer
from sigllm.models.multimodal.qformer_alignment_model import QRecInstructAlignmentModel


def collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out

def train_step(batch, model: QRecInstructAlignmentModel, w_ui: float = 1.0, w_it: float = 0.5):
    device = batch["u"].device
    u = batch["u"]
    i_pos = batch["i_pos"]
    i_negs = batch["i_negs"]
    ins_list = batch["instruction"]
    itxt_list = batch["item_text"]

    ins_tok_emb = model.ins_tokens(ins_list, device)

    u_vec = model.enc_user(u, ins_tok_emb)
    i_pos_vec = model.enc_item(i_pos, ins_tok_emb)

    B, K = i_negs.shape
    ins_rep = ins_tok_emb.repeat_interleave(K, dim=0)
    i_negs_flat = i_negs.reshape(B * K)
    i_neg_vec_flat = model.enc_item(i_negs_flat, ins_rep)
    i_neg_vecs = i_neg_vec_flat.reshape(B, K, -1)

    t_vec = model.text_vec(itxt_list, device)

    L_ui = model.loss_user_item(u_vec, i_pos_vec, i_neg_vecs)
    L_it = model.loss_item_text(i_pos_vec, t_vec)

    loss = w_ui * L_ui + w_it * L_it
    return loss, {"L_ui": L_ui, "L_it": L_it}


def train_qformer_stage1_representation(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    storage = Path(cfg.data_dir)
    dataset_cfg = type("Cfg", (), {})()
    dataset_cfg.build_info = type("BI", (), {})()
    dataset_cfg.build_info.storage = storage

    dataset = QFormerAlignmentDataset(
        config=dataset_cfg,
        filename="train_ood2",
        neg_k=cfg.neg_k,
        hard_k=cfg.hard_k,
        p_fixed=cfg.p_fixed,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=cfg.num_workers)

    mf_config = type("MF", (), {})()
    mf_config.user_num = cfg.user_num
    mf_config.item_num = cfg.item_num
    mf_config.embedding_size = cfg.embedding_size
    mf = MatrixFactorization(mf_config).to(device)

    llama_tokenizer = LlamaTokenizer.from_pretrained(cfg.llama_model, use_fast=False)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_model = LlamaForCausalLM.from_pretrained(cfg.llama_model)
    llama_hidden = llama_model.config.hidden_size

    qformer = QFormer(d_cf=cfg.embedding_size, d_model=llama_hidden, num_queries=cfg.num_queries, num_heads=cfg.num_heads, num_layers=cfg.num_layers).to(device)

    model = QRecInstructAlignmentModel(mf, qformer, llama_tokenizer, llama_model).to(device)

    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    for epoch in range(cfg.epoch):
        model.train()
        for batch in loader:
            batch["u"] = batch["u"].to(device)
            batch["i_pos"] = batch["i_pos"].to(device)
            batch["i_negs"] = batch["i_negs"].to(device)

            loss, logs = train_step(batch, model, w_ui=cfg.w_ui, w_it=cfg.w_it)
            loss.backward()
            opt.step()
            opt.zero_grad()

        if (epoch + 1) % cfg.log_epoch == 0:
            print(f"epoch {epoch+1} loss={loss.item():.4f} L_ui={logs['L_ui'].item():.4f} L_it={logs['L_it'].item():.4f}")

    return model


def main():
    train_cfg = {
        "data_dir": "/content/SigLLM/data/processed/ml-1m/",
        "batch_size": 256,
        "num_workers": 4,
        "embedding_size": 256,
        "user_num": 100000,
        "item_num": 100000,
        "num_queries": 8,
        "num_heads": 8,
        "num_layers": 2,
        "neg_k": 32,
        "hard_k": 8,
        "p_fixed": 0.8,
        "lr": 1e-4,
        "w_ui": 1.0,
        "w_it": 0.5,
        "log_epoch": 1,
        "epoch": 10,
        "llama_model": "/content/open_llama_3b",
    }

    cfg = type("Cfg", (), train_cfg)()

    train_qformer_stage1_representation(cfg)


if __name__ == "__main__":
    main()