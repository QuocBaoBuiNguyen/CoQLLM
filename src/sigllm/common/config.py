import logging

from omegaconf import OmegaConf
from sigllm.common import registry

class Config:
    def __init__(self, args):
        self.args = args
        registry.register("configuration", self)

        cli_overrides = self._parse_cli_overrides(args.options)
        main_cfg = OmegaConf.load(self.args.cfg_path)

        self.config = OmegaConf.merge(
            self._build_runner_config(main_cfg),
            self._build_dataset_config(main_cfg),
            self._build_model_config(main_cfg),
            cli_overrides
        )

    def _parse_cli_overrides(self, opts):
        return OmegaConf.from_dotlist(self._convert_to_dot_list(opts))

    def _build_runner_config(self, config):
        return OmegaConf.create({"run": config.get("run", {})})

    def _build_dataset_config(self, config):
        datasets = config.get("datasets")
        if datasets is None:
            raise KeyError("Expecting 'datasets' as the root key.")
        return OmegaConf.create({"datasets": datasets})

    def _build_model_config(self, config, **kwargs):
        model = config.get("model")
        if model is None:
            raise KeyError("Missing 'model' section in configuration.")
        return OmegaConf.create({"model": model})

    def _convert_to_dot_list(opts):
        if not opts:
            return []
        if opts[0].find("=") != -1:
            return opts
        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]
    
    @property
    def run_cfg(self):
        return self.config.run

    @property
    def datasets_cfg(self):
        return self.config.datasets

    @property
    def model_cfg(self):
        return self.config.model

    
    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        logging.info("\n======  Dataset Attributes  ======")
        datasets = self.config.datasets

        for dataset in datasets:
            if dataset in self.config.datasets:
                logging.info(f"\n======== {dataset} =======")
                dataset_config = self.config.datasets[dataset]
                logging.info(self._convert_node_to_json(dataset_config))
            else:
                logging.warning(f"No dataset named '{dataset}' in config. Skipping")

        logging.info(f"\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))

