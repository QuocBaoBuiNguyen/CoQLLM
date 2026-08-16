import torch
import logging

def build_optimizer(model, config):
    """
    Build optimizer from config.
    """
    num_parameters = 0
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
        num_parameters += p.data.nelement()
        
    logging.info("number of trainable parameters: %d" % num_parameters)
    
    optim_params = [
        {
            "params": p_wd,
            "weight_decay": float(config.run_cfg.weight_decay),
        },
        {"params": p_non_wd, "weight_decay": 0},
    ]
    
    beta2 = config.run_cfg.get("beta2", 0.999)
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(config.run_cfg.init_lr),
        weight_decay=float(config.run_cfg.weight_decay),
        betas=(0.9, beta2),
    )

    return optimizer
