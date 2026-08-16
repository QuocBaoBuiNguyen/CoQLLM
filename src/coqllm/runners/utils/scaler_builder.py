import torch

def build_scaler(config):
    """
    Build scaler for mixed precision training.
    """
    amp = config.run_cfg.get("amp", False)
    scaler = None
    if amp:
        scaler = torch.amp.GradScaler('cuda')
    return scaler
