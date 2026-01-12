from sigllm.common import registry


class RecBaseTask:
    def __init__(self):
        pass

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()
    
    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            builder = registry.get_builder_class(name)(cfg)
            dataset = builder.build_datasets()
            datasets[name] = dataset

        return datasets