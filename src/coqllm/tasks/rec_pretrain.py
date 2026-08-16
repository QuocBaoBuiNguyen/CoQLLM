
from coqllm.common import registry
from coqllm.tasks.base.rec_base_task import RecBaseTask


@registry.register_task("rec_pretrain")
class RecPretrainTask(RecBaseTask):
    def __init__(self):
        super().__init__()