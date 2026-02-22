import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import omegaconf
import os

from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset
from sigllm.models.rec.matrix_factorization import MatrixFactorization
from sigllm.models.q_former.q_former import QFormer
from sigllm.models.q_former.text_encoder import TextEncoder
from sigllm.models.multimodal.qformer_alignment_model import QRecInstructAlignmentModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


def _init_rec_model(cfg, device):
    """
    Initializes the recommendation model, loads pretrained weights if available,
    and freezes parameters if configured.
    """
    mf_config = omegaconf.OmegaConf.create({
        "user_num": int(cfg.user_num),
        "item_num": int(cfg.item_num),
        "embedding_size": int(cfg.embedding_size)
    })
    mf = MatrixFactorization(mf_config).to(device)

    pretrained_rec_path = cfg.get("pretrained_rec_path", "not_have")
    if mf is not None and pretrained_rec_path != "not_have" and os.path.exists(pretrained_rec_path):
        mf.load_state_dict(torch.load(pretrained_rec_path, map_location="cpu"))
        print(f"Successfully loaded the pretrained rec model from {pretrained_rec_path}")

    if cfg.get("freeze_rec", False) and mf is not None:
        for param in mf.parameters():
            param.requires_grad = False
        mf.eval()
        mf.train = disabled_train.__get__(mf, MatrixFactorization)
        print("Freeze rec encoder completed")

    return mf


def _init_dataset(cfg, filename: str, shuffle: bool = True):
    """
    Initializes the dataset and dataloader.
    """
    dataset_cfg = omegaconf.OmegaConf.create({
        "build_info": {
            "storage": Path(cfg.data_dir)
        }
    })
    dataset = QFormerAlignmentDataset(
        config=dataset_cfg,
        filename=filename,
        neg_k=cfg.neg_k,
        hard_k=cfg.hard_k,
        p_fixed=cfg.p_fixed,
    )
    loader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=shuffle, 
        collate_fn=collate, 
        num_workers=cfg.num_workers
    )
    return loader


def _init_text_encoder(cfg, device):
    """
    Initializes the TextEncoder.
    """
    # d_model = cfg.get("text_d_model", 768)
    text_model_name = cfg.get("text_model_name", "bert-base-uncased")
    text_encoder = TextEncoder(model_name=text_model_name).to(device)
    
    # Freeze if necessary
    if cfg.get("freeze_text_encoder", True):
        for p in text_encoder.parameters():
            p.requires_grad = False
            
    return text_encoder, text_encoder.model.config.hidden_size


def _init_qformer(cfg, d_model, device):
    """
    Initializes the Q-Former model.
    """
    return QFormer(
        d_cf=cfg.embedding_size, 
        d_model=d_model, 
        num_queries=cfg.num_queries, 
        num_heads=cfg.num_heads, 
        num_layers=cfg.num_layers
    ).to(device)


def _init_optimizer(model, lr):
    """
    Initializes the Adam optimizer for trainable parameters.
    """
    return Adam([p for p in model.parameters() if p.requires_grad], lr=lr)


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



def evaluate_loss(model, loader, w_ui=1.0, w_it=0.5):
    """
    Evaluates the model on a given dataloader.
    Returns average loss, L_ui, and L_it.
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_lui = 0.0
    total_lit = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            batch["u"] = batch["u"].to(device)
            batch["i_pos"] = batch["i_pos"].to(device)
            batch["i_negs"] = batch["i_negs"].to(device)
            
            loss, logs = train_step(batch, model, w_ui=w_ui, w_it=w_it)
            
            total_loss += loss.item()
            total_lui += logs["L_ui"].item()
            total_lit += logs["L_it"].item()
            steps += 1

    if steps == 0:
        return 0, 0, 0
    return total_loss / steps, total_lui / steps, total_lit / steps


def train_qformer_stage1_representation(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = _init_dataset(cfg, filename="train_ood2.pkl", shuffle=True)
    val_loader = _init_dataset(cfg, filename="valid_ood2.pkl", shuffle=False)
    test_loader = _init_dataset(cfg, filename="test_ood2.pkl", shuffle=False)
    
    mf = _init_rec_model(cfg, device)
    text_encoder, d_model = _init_text_encoder(cfg, device)
    qformer = _init_qformer(cfg, d_model, device)

    model = QRecInstructAlignmentModel(mf, qformer, text_encoder).to(device)
    opt = _init_optimizer(model, cfg.lr)

    for epoch in range(cfg.epoch):
        model.train()
        train_loss = 0
        train_steps = 0
        for batch in train_loader:
            batch["u"] = batch["u"].to(device)
            batch["i_pos"] = batch["i_pos"].to(device)
            batch["i_negs"] = batch["i_negs"].to(device)

            loss, logs = train_step(batch, model, w_ui=cfg.w_ui, w_it=cfg.w_it)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            train_loss += loss.item()
            train_steps += 1    

        if (epoch + 1) % cfg.log_epoch == 0:
            avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
            val_loss, val_lui, val_lit = evaluate_loss(model, val_loader, w_ui=cfg.w_ui, w_it=cfg.w_it)
            print(f"epoch {epoch+1} | Train Loss={avg_train_loss:.4f} | Val Loss={val_loss:.4f} L_ui={val_lui:.4f} L_it={val_lit:.4f}")

    # Final Test
    print("Evaluating on Test Set...")
    test_loss, test_lui, test_lit = evaluate_loss(model, test_loader, w_ui=cfg.w_ui, w_it=cfg.w_it)
    print(f"Test Results: Loss={test_loss:.4f} L_ui={test_lui:.4f} L_it={test_lit:.4f}")

    outdir = cfg.get("output_dir", "/content/SigLLM/ckpt/qformer_stage1/")
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.qformer.state_dict(), os.path.join(outdir, "qformer_stage1.pth"))
    
    return model


def main():
    train_cfg_dict = {
        "data_dir": "/content/SigLLM/data/processed/ml-1m/",
        "batch_size": 256,
        "num_workers": 4,
        "embedding_size": 256,
        "user_num": 839,
        "item_num": 3256,
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
        "epoch": 100,
        "text_model_name": "bert-base-uncased",
        "text_d_model": 768,
        "pretrained_rec_path": "/content/SigLLM/ckpt/mf/mf_model.pth",
        "freeze_rec": True,
        "freeze_text_encoder": True,
    }

    cfg = omegaconf.OmegaConf.create(train_cfg_dict)

    train_qformer_stage1_representation(cfg)


if __name__ == "__main__":
    main()