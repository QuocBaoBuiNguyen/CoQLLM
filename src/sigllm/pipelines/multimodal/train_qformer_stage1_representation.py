import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import omegaconf
import os

from sigllm.datasets.qformer.qformer_alignment_builder import QFormerAlignmentBuilder
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
    dataset = QFormerAlignmentDataset(filename=filename)
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
        text_encoder.eval()
        text_encoder.train = disabled_train.__get__(text_encoder, TextEncoder)

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


def _init_optimizer(model, lr, weight_decay=0.0):
    """
    Initializes the Adam optimizer for trainable parameters.
    """
    return Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )


def _log_batch_preview(batch, prefix: str = "train_step", max_neg_preview: int = 5):
    """Print a compact preview of the current batch for debugging."""
    batch_size = batch["u"].size(0)
    neg_count = batch["i_negs"].size(1) if batch["i_negs"].dim() > 1 else 0
    first_negatives = batch["i_negs"][0, :max_neg_preview].tolist() if batch_size > 0 else []

    print(
        f"[{prefix}] batch_size={batch_size} neg_k={neg_count} "
        f"u.shape={tuple(batch['u'].shape)} i_pos.shape={tuple(batch['i_pos'].shape)} "
        f"i_negs.shape={tuple(batch['i_negs'].shape)}"
    )
    if batch_size > 0:
        print(
            f"[{prefix}] sample[0] u={batch['u'][0].item()} i_pos={batch['i_pos'][0].item()} "
            f"i_negs[:{max_neg_preview}]={first_negatives}"
        )
        print(f"[{prefix}] sample[0] instruction={batch['instruction'][0]}")
        print(f"[{prefix}] sample[0] item_text={batch['item_text'][0]}")


def _compute_alignment_metrics(u_vec, i_pos_vec, i_neg_vecs, t_vec, tau_ui: float, tau_it: float):
    """Compute Top-1 accuracy metrics for user-item and item-text alignment."""
    u = QRecInstructAlignmentModel.l2norm(u_vec)
    pos = QRecInstructAlignmentModel.l2norm(i_pos_vec)
    neg = QRecInstructAlignmentModel.l2norm(i_neg_vecs)
    text = QRecInstructAlignmentModel.l2norm(t_vec)

    pos_logits = (u * pos).sum(-1, keepdim=True) / tau_ui
    neg_logits = (u.unsqueeze(1) * neg).sum(-1) / tau_ui
    ui_logits = torch.cat([pos_logits, neg_logits], dim=1)
    ui_predictions = ui_logits.argmax(dim=1)
    ui_top1 = (ui_predictions == 0).float().mean()

    it_logits = (pos @ text.T) / tau_it
    it_labels = torch.arange(pos.size(0), device=pos.device)
    it_predictions = it_logits.argmax(dim=1)
    it_top1 = (it_predictions == it_labels).float().mean()

    return {
        "ui_top1": ui_top1,
        "it_top1": it_top1,
    }


def train_step(
    batch,
    model: QRecInstructAlignmentModel,
    w_ui: float = 1.0,
    w_it: float = 0.5,
    tau_ui: float = 0.07,
    tau_it: float = 0.2,
    debug_batch: bool = False,
):
    device = batch["u"].device
    u = batch["u"]
    i_pos = batch["i_pos"]
    i_negs = batch["i_negs"]
    ins_list = batch["instruction"]
    itxt_list = batch["item_text"]

    if debug_batch:
        _log_batch_preview(batch)

    ins_tok_emb = model.ins_tokens(ins_list, device)

    u_vec = model.enc_user(u, ins_tok_emb)
    i_pos_vec = model.enc_item(i_pos, ins_tok_emb)

    B, K = i_negs.shape
    ins_rep = ins_tok_emb.repeat_interleave(K, dim=0)
    i_negs_flat = i_negs.reshape(B * K)
    i_neg_vec_flat = model.enc_item(i_negs_flat, ins_rep)
    i_neg_vecs = i_neg_vec_flat.reshape(B, K, -1)

    t_vec = model.text_vec(itxt_list, device)

    L_ui = model.loss_user_item(u_vec, i_pos_vec, i_neg_vecs, tau=tau_ui)
    L_it = model.loss_item_text(i_pos_vec, t_vec, tau=tau_it)
    metrics = _compute_alignment_metrics(u_vec, i_pos_vec, i_neg_vecs, t_vec, tau_ui, tau_it)

    loss = w_ui * L_ui + w_it * L_it
    return loss, {"L_ui": L_ui, "L_it": L_it, **metrics}



def evaluate_loss(model, loader, w_ui=1.0, w_it=0.5, tau_ui=0.07, tau_it=0.2):
    """
    Evaluates the model on a given dataloader.
    Returns average loss, L_ui, and L_it.
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_lui = 0.0
    total_lit = 0.0
    total_ui_top1 = 0.0
    total_it_top1 = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            batch["u"] = batch["u"].to(device)
            batch["i_pos"] = batch["i_pos"].to(device)
            batch["i_negs"] = batch["i_negs"].to(device)
            
            loss, logs = train_step(
                batch,
                model,
                w_ui=w_ui,
                w_it=w_it,
                tau_ui=tau_ui,
                tau_it=tau_it,
            )
            
            total_loss += loss.item()
            total_lui += logs["L_ui"].item()
            total_lit += logs["L_it"].item()
            total_ui_top1 += logs["ui_top1"].item()
            total_it_top1 += logs["it_top1"].item()
            steps += 1

    if steps == 0:
        return 0, 0, 0, 0, 0
    return (
        total_loss / steps,
        total_lui / steps,
        total_lit / steps,
        total_ui_top1 / steps,
        total_it_top1 / steps,
    )


def _save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    val_loss,
    val_lui,
    val_lit,
    val_ui_top1,
    val_it_top1,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_lui": val_lui,
            "val_lit": val_lit,
            "val_ui_top1": val_ui_top1,
            "val_it_top1": val_it_top1,
        },
        checkpoint_path,
    )

def _load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def train_qformer_stage1_representation(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = _init_dataset(cfg, filename=cfg.data_dir + "train_qformer_ood2.pkl", shuffle=True)
    val_loader = _init_dataset(cfg, filename=cfg.data_dir + "valid_qformer_ood2.pkl", shuffle=False)
    test_loader = _init_dataset(cfg, filename=cfg.data_dir + "test_qformer_ood2.pkl", shuffle=False)
    
    mf = _init_rec_model(cfg, device)
    text_encoder, d_model = _init_text_encoder(cfg, device)
    qformer = _init_qformer(cfg, d_model, device)

    model = QRecInstructAlignmentModel(mf, qformer, text_encoder).to(device)
    opt = _init_optimizer(model, cfg.lr, weight_decay=cfg.weight_decay)

    outdir = cfg.get("output_dir", "/content/SigLLM/ckpt/qformer_stage1/")
    os.makedirs(outdir, exist_ok=True)
    best_checkpoint_path = os.path.join(outdir, cfg.get("best_checkpoint_name", "best_qformer_stage1.pt"))
    best_val_loss = float("inf")
    best_epoch = -1
    early_stop_counter = 0

    for epoch in range(cfg.epoch):
        model.train()
        train_loss = 0
        train_lui = 0
        train_lit = 0
        train_ui_top1 = 0
        train_it_top1 = 0
        train_steps = 0
        for batch in train_loader:
            batch["u"] = batch["u"].to(device)
            batch["i_pos"] = batch["i_pos"].to(device)
            batch["i_negs"] = batch["i_negs"].to(device)

            loss, logs = train_step(
                batch,
                model,
                w_ui=cfg.w_ui,
                w_it=cfg.w_it,
                tau_ui=cfg.tau_ui,
                tau_it=cfg.tau_it,
                debug_batch=cfg.debug_batch and epoch == 0 and train_steps < cfg.debug_batch_max_steps,
            )
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            train_loss += loss.item()
            train_lui += logs["L_ui"].item()
            train_lit += logs["L_it"].item()
            train_ui_top1 += logs["ui_top1"].item()
            train_it_top1 += logs["it_top1"].item()
            train_steps += 1    

        if (epoch + 1) % cfg.log_epoch == 0:
            avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
            avg_train_lui = train_lui / train_steps if train_steps > 0 else 0
            avg_train_lit = train_lit / train_steps if train_steps > 0 else 0
            avg_train_ui_top1 = train_ui_top1 / train_steps if train_steps > 0 else 0
            avg_train_it_top1 = train_it_top1 / train_steps if train_steps > 0 else 0
            val_loss, val_lui, val_lit, val_ui_top1, val_it_top1 = evaluate_loss(
                model,
                val_loader,
                w_ui=cfg.w_ui,
                w_it=cfg.w_it,
                tau_ui=cfg.tau_ui,
                tau_it=cfg.tau_it,
            )
            print(
                f"epoch {epoch+1} | "
                f"Train Loss={avg_train_loss:.4f} L_ui={avg_train_lui:.4f} L_it={avg_train_lit:.4f} "
                f"UI@1={avg_train_ui_top1:.4f} IT@1={avg_train_it_top1:.4f} | "
                f"Val Loss={val_loss:.4f} L_ui={val_lui:.4f} L_it={val_lit:.4f} "
                f"UI@1={val_ui_top1:.4f} IT@1={val_it_top1:.4f} | "
                f"w_it={cfg.w_it:.3f} tau_ui={cfg.tau_ui:.3f} tau_it={cfg.tau_it:.3f}"
            )

            if val_loss < best_val_loss - cfg.early_stopping_min_delta:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                early_stop_counter = 0
                _save_checkpoint(
                    best_checkpoint_path,
                    model,
                    opt,
                    epoch + 1,
                    val_loss,
                    val_lui,
                    val_lit,
                    val_ui_top1,
                    val_it_top1,
                )
                print(f"Saved new best checkpoint at epoch {best_epoch} -> {best_checkpoint_path}")
            else:
                early_stop_counter += 1
                print(
                    f"No val improvement for {early_stop_counter} epoch(s). "
                    f"Best Val Loss={best_val_loss:.4f} at epoch {best_epoch}"
                )

            if early_stop_counter >= cfg.early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. "
                    f"Best epoch={best_epoch} Best Val Loss={best_val_loss:.4f}"
                )
                break

    if os.path.exists(best_checkpoint_path):
        best_checkpoint = _load_checkpoint(best_checkpoint_path, model)
        print(
            f"Loaded best checkpoint from epoch {best_checkpoint['epoch']} "
            f"with Val Loss={best_checkpoint['val_loss']:.4f} "
            f"UI@1={best_checkpoint.get('val_ui_top1', 0.0):.4f} "
            f"IT@1={best_checkpoint.get('val_it_top1', 0.0):.4f}"
        )

    # Final Test
    print("Evaluating on Test Set...")
    test_loss, test_lui, test_lit, test_ui_top1, test_it_top1 = evaluate_loss(
        model,
        test_loader,
        w_ui=cfg.w_ui,
        w_it=cfg.w_it,
        tau_ui=cfg.tau_ui,
        tau_it=cfg.tau_it,
    )
    print(
        f"Test Results: Loss={test_loss:.4f} L_ui={test_lui:.4f} L_it={test_lit:.4f} "
        f"UI@1={test_ui_top1:.4f} IT@1={test_it_top1:.4f}"
    )

    torch.save(model.qformer.state_dict(), os.path.join(outdir, "qformer_stage1.pth"))
    if os.path.exists(best_checkpoint_path):
        torch.save(model.qformer.state_dict(), os.path.join(outdir, "qformer_stage1_best.pth"))
    
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
        "w_it": 0.10,
        "tau_ui": 0.05,
        "tau_it": 0.10,
        "weight_decay": 1e-4,
        "debug_batch": True,
        "debug_batch_max_steps": 1,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 1e-4,
        "best_checkpoint_name": "best_qformer_stage1.pt",
        "log_epoch": 1,
        "epoch": 100,
        "text_model_name": "bert-base-uncased",
        "text_d_model": 768,
        "pretrained_rec_path": "/content/SigLLM/ckpt/mf/mf_model.pth",
        "freeze_rec": True,
        "freeze_text_encoder": True,
    }

    cfg = omegaconf.OmegaConf.create(train_cfg_dict)

    QFormerAlignmentBuilder.build_qformer_alignment_samples(        
        input_pkl_path=cfg.data_dir + "train_ood2.pkl",
        output_path=cfg.data_dir + "train_qformer_ood2.pkl"
    )
    QFormerAlignmentBuilder.build_qformer_alignment_samples(        
        input_pkl_path=cfg.data_dir + "valid_ood2.pkl",
        output_path=cfg.data_dir + "valid_qformer_ood2.pkl"
    )
    QFormerAlignmentBuilder.build_qformer_alignment_samples(        
        input_pkl_path=cfg.data_dir + "test_ood2.pkl",
        output_path=cfg.data_dir + "test_qformer_ood2.pkl"
    )

    train_qformer_stage1_representation(cfg)


if __name__ == "__main__":
    main()