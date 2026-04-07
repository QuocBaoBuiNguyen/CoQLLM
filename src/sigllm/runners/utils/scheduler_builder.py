from sigllm.common.registry import registry
from sigllm.common import optims

def build_scheduler(optimizer, config, dataloaders, max_epoch, min_lr, init_lr):
    """
    Build learning rate scheduler.
    """
    lr_sched_cls = registry.get_lr_scheduler_class(config.run_cfg.lr_sched)

    # optional parameters
    decay_rate = config.run_cfg.get("lr_decay_rate", None)
    warmup_start_lr = config.run_cfg.get("warmup_lr", -1)
    warmup_steps = config.run_cfg.get("warmup_steps", 0)
    iters_per_epoch = config.run_cfg.get("iters_per_epoch", None)

    if iters_per_epoch is None:
        try:
            iters_per_epoch = len(dataloaders['train'])
        except (AttributeError, TypeError):
            iters_per_epoch = 10000

    lr_scheduler = lr_sched_cls(
        optimizer=optimizer,
        max_epoch=max_epoch,
        iters_per_epoch=iters_per_epoch,
        min_lr=min_lr,
        init_lr=init_lr,
        decay_rate=decay_rate,
        warmup_start_lr=warmup_start_lr,
        warmup_steps=warmup_steps,
    )

    return lr_scheduler
