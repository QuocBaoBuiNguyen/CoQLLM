"""Project-wide registry."""

class Registry:
    """Global registry for builders, tasks, processors, models, etc."""

    mapping = {
        "builder_name_mapping": {},
    }

    @classmethod
    def register_builder(cls, name):
        r"""Register a dataset builder to registry under a given name.
        
        Args:
            name (str): The name to register the builder under.

        Returns:
            A decorator that registers the builder class.
        """

        def wrap(builder_cls):

            from sigllm.datasets.base.rec_base_dataset_builder import RecBaseDatasetBuilder

            assert issubclass(
                builder_cls, RecBaseDatasetBuilder
            ), "All builders must inherit RecBaseDatasetBuilder class, found {}".format(
                builder_cls
            )
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls

        return wrap

registry = Registry()
