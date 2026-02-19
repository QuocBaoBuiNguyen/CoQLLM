import os
import numpy as np
import time as time
import random
import omegaconf
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader

from sigllm.common import NotebookLogger, EarlyStopping
from sigllm.models.rec.matrix_factorization import MatrixFactorization

LOGGER = NotebookLogger.rich_logger("sigllm.train_rec_baseline")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

def calculate_user_auc(user_ids, y_pred, y_true):
    """Calculate User AUC (uAUC) for recommendation tasks."""
    
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze()
    user_ids = np.asarray(user_ids)

    start_time = time.time()

    users, inverse, counts = np.unique(user_ids, return_inverse=True, return_counts=True)
    sort_indices = np.argsort(inverse)

    auc_list = []
    computed_users = []
    only_one_interaction = 0
    only_one_class = 0
    current_pos = 0

    for i, user_id in enumerate(users):
        user_count = counts[i]
        user_indices = sort_indices[current_pos:current_pos + user_count]

        user_y_true = y_true[user_indices]
        user_y_pred = y_pred[user_indices]

        current_pos += user_count

        if user_count < 2:
            only_one_interaction += 1
            continue

        if np.all(user_y_true == user_y_true[0]):
            only_one_class += 1
            continue

        score = roc_auc_score(user_y_true, user_y_pred)
        auc_list.append(score)
        computed_users.append(user_id)

    auc_array = np.array(auc_list)
    avg_uauc = auc_array.mean()
    log_step("Users with only one interaction", str(only_one_interaction))
    log_step("Users with only one class", str(only_one_class))
    log_step("Computed user AUC count", str(len(auc_list)))
    log_step("Average User AUC", f"{avg_uauc:.6f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    log_step("Time taken to calculate user AUC", f"{elapsed_time:.2f} seconds")

    return avg_uauc, computed_users, auc_array

def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model_predictions(model, data_loader, device):
    model.eval()
    all_preds, all_labels, all_users = [], [], []

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            # Forward pass
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            
            all_users.append(batch_data[:, 0].cpu().numpy())
            all_preds.append(ui_matching.detach().cpu().numpy())
            all_labels.append(batch_data[:, -1].cpu().numpy())
            
    return (np.concatenate(all_users), 
            np.concatenate(all_preds), 
            np.concatenate(all_labels))

def train_baseline_model(train_config, log_file=None, save_mode=False, save_file=None, need_train=True, warm_or_cold=None):
    # 1. Setup Environment
    set_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/content/SigLLM/data/processed/ml-1m/"
    
    # 2. Load and Filter Data
    train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid','iid','label']].values
    valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid','iid','label']].values
    test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label']].values

    user_num = max(train_data[:,0].max(), valid_data[:,0].max(), test_data[:,0].max()) + 1
    item_num =  max(train_data[:,1].max(), valid_data[:,1].max(), test_data[:,1].max()) + 1
    log_step("User num:", str(user_num))
    log_step("Item num:", str(item_num))

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir+"test_warm_cold_ood2.pkl")[['uid','iid','label', 'warm']]
            test_data = test_data[test_data['warm'].isin([1])][['uid','iid','label']].values
            log_step("warm data size:", str(test_data.shape[0]))
        else:
            test_data = pd.read_pickle(data_dir+"test_warm_cold_ood2.pkl")[['uid','iid','label', 'cold']]
            test_data = test_data[test_data['cold'].isin([1])][['uid','iid','label']].values
            log_step("cold data size:", str(test_data.shape[0]))

    train_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=train_config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=train_config['batch_size'], shuffle=False)
    
    # 3. Model & Optimizer Initialization
    mf_config = omegaconf.OmegaConf.create({
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size'])
    })

    model = MatrixFactorization(mf_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    stopper = EarlyStopping(ref_metric='valid_auc', monitor_mode='max', patience=train_config['patience'])
    criterion = nn.BCEWithLogitsLoss()

    #4. Inference only
    if not need_train:
        log_step("Starting evaluation only mode.")
        model.load_state_dict(torch.load(save_file))
        
        v_users, v_preds, v_labels = get_model_predictions(model, valid_loader, device)
        valid_auc = roc_auc_score(v_labels, v_preds)
        valid_uauc, _, _ = calculate_user_auc(v_users, v_preds, v_labels)
        
        t_users, t_preds, t_labels = get_model_predictions(model, test_loader, device)
        test_auc = roc_auc_score(t_labels, t_preds)
        test_uauc, _, _ = calculate_user_auc(t_users, t_preds, t_labels)

        threshold = 0.1
        acc = ((v_preds >= threshold) == v_labels).mean()
        
        log_step(f"Valid AUC: {valid_auc:.4f}, Valid uAUC: {valid_uauc:.4f}, Test AUC: {test_auc:.4f}, Test uAUC: {test_uauc:.4f}, Acc: {acc:.4f}")
        return
    
    #5. Training loop
    for epoch in range(train_config['epoch']):
        model.train()
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            loss = criterion(ui_matching.squeeze(), batch_data[:, -1].float())

            loss.backward()
            optimizer.step()

        if epoch % train_config['eval_epoch'] == 0:
            v_users, v_preds, v_labels = get_model_predictions(model, valid_loader, device)
            valid_auc = roc_auc_score(v_labels, v_preds)
            valid_uauc, _, _ = calculate_user_auc(v_users, v_preds, v_labels)
            
            t_users, t_preds, t_labels = get_model_predictions(model, test_loader, device)
            test_auc = roc_auc_score(t_labels, t_preds)
            test_uauc, _, _ = calculate_user_auc(t_users, t_preds, t_labels)

            threshold = 0.1
            acc = ((v_preds >= threshold) == v_labels).mean()
            
            log_step(f"Epoch {epoch}: Valid AUC: {valid_auc:.4f}, Valid uAUC: {valid_uauc:.4f}, Test AUC: {test_auc:.4f}, Test uAUC: {test_uauc:.4f}, Acc: {acc:.4f}")

            metrics = {
                'valid_auc': valid_auc, 'valid_uauc': valid_uauc,
                'test_auc': test_auc, 'test_uauc': test_uauc, 'epoch': epoch
            }

            improved = stopper.update(metrics)

            if improved:
                log_step(f"New best model found at epoch {epoch} with Valid uAUC: {valid_uauc:.4f}")
                if save_mode and save_file is not None:
                    torch.save(model.state_dict(), save_file)
                    log_step(f"Model saved to {save_file}")

            if stopper.should_stop:
                log_step("Early stopping triggered. Ending training.")
                break

            if epoch > 500 and stopper.best_metric_val < 0.52:
                log_step("Training fails to converge (Valid AUC < 0.52 at epoch 500)")
                break

    # 6. Final Logging
    final_log = f"Train Config: {train_config}\nBest Results: {stopper.best_full_metrics}"
    log_step(final_log)
    if log_file:
        log_file.write(final_log + "\n")
        log_file.flush()

    return stopper.best_full_metrics

if __name__ == "__main__":
    train_config = {
        'lr': 1e-3,
        'wd': 1e-4,
        'embedding_size': 256,
        "epoch": 5000,
        "eval_epoch":1,
        "patience":50,
        "batch_size":1024
    }
    save_file = "/content/SigLLM/mf/mf_model.pth"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    train_baseline_model(train_config, save_mode=True, save_file=save_file, need_train=True, warm_or_cold='warm')