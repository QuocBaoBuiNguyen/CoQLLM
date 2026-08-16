from torch.nn.parallel import DistributedDataParallel as DDP

def wrap_model(model, config, device, use_distributed):
    """
    Wrap model for distributed training if enabled.
    """
    # move model to device
    if model.device != device:
        model = model.to(device)

    # distributed training wrapper
    if use_distributed:
        model = DDP(
            model, device_ids=[config.run_cfg.gpu], find_unused_parameters=True
        )

    return model
