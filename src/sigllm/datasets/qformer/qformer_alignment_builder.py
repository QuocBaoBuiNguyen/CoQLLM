"""Builder for Q-Former alignment dataset."""

from __future__ import annotations

from sigllm.common.registry import registry
from sigllm.datasets.base.rec_base_dataset_builder import RecBaseDatasetBuilder
from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset


@registry.register_builder("qformer_alignment")
class QFormerAlignmentBuilder(RecBaseDatasetBuilder):
    """Construct Q-Former alignment splits."""

    train_dataset_cls = QFormerAlignmentDataset

    def build_datasets(self, evaluate_only: bool = False):
        dataset_cls = self.train_dataset_cls
        build_info = self.dataset_config.build_info
        storage_path = build_info.storage

        if storage_path is None or not storage_path.exists():
            raise ValueError(f"storage path {storage_path} does not exist.")

        datasets = dict()

        if not evaluate_only:
            datasets["train"] = dataset_cls(config=self.dataset_config, filename="train_ood2")
            datasets["valid"] = dataset_cls(config=self.dataset_config, filename="valid_ood2")
            datasets["test"] = dataset_cls(config=self.dataset_config, filename="test_ood2")
        else:
            datasets["test"] = dataset_cls(config=self.dataset_config, filename="test_ood2")

        return datasets
