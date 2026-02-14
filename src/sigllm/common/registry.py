"""Project-wide registry."""

class Registry:
    """Global registry for builders, tasks, processors, models, etc."""

    mapping = {
        "builder_name_mapping": {},
        "task_name_mapping": {},
        "model_name_mapping": {},
        "runner_name_mapping": {},
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

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def register_task(cls, name):
        r"""Register a task to registry under a given name.
        
        Args:
            name (str): The name to register the task under.

        Returns:
            A decorator that registers the task class.
        """

        def wrap(task_cls):

            from sigllm.tasks.base.rec_base_task import RecBaseTask

            assert issubclass(
                task_cls, RecBaseTask
            ), "All tasks must inherit RecBaseTask class, found {}".format(
                task_cls
            )
            cls.mapping["task_name_mapping"][name] = task_cls
            return task_cls

        return wrap

    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)
    
    @classmethod
    def register_runner(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """

        def wrap(runner_cls):
            if name in cls.mapping["runner_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["runner_name_mapping"][name]
                    )
                )
            cls.mapping["runner_name_mapping"][name] = runner_cls
            return runner_cls

        return wrap

    @classmethod
    def get_runner_class(cls, name):
        return cls.mapping["runner_name_mapping"].get(name, None)

    @classmethod
    def register_model(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """

        def wrap(model_cls):
            from sigllm.models import BaseModel

            assert issubclass(
                model_cls, BaseModel
            ), "All models must inherit BaseModel class"
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)
    
    @classmethod
    def register_path(cls, name, path):
        r"""Register a path to registry with key 'name'

        Args:
            name: Key with which the path will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """
        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            raise KeyError("Name '{}' already registered.".format(name))
        cls.mapping["paths"][name] = path
        
    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)

registry = Registry()
