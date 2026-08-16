"""CoLLM Step 1 — train LoRA only on text-only prompts.

The Q-Former, projection, MF and base LLM are all frozen; only the LoRA
adapter on the LLM is updated. The text-only prompt is identical in shape to
the full Stage 3 prompt but with the soft-token placeholders (`<ItemIDList>`,
`<TargetItemID>`) removed, so the LLM learns the recommendation task without
any collaborative-filtering signal. The best checkpoint feeds Step 2.
"""

import argparse
import glob
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.distributed.elastic.multiprocessing.errors import record

from coqllm import tasks
from coqllm.common.config import Config
from coqllm.common.dist_utils import get_rank, init_distributed_mode
from coqllm.common.utils import derive_job_id_from_llm
from coqllm.runners.runner_base_rec import RecRunnerBase  # noqa: F401  (registry side-effect)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 3 Step 1 — LoRA on text-only prompts")
    parser.add_argument("--cfg-path", type=str, required=True, help="Path to the config file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config",
    )
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def apply_step1_overrides(cfg):
    step1 = cfg.run_cfg.qformer_stage3_step1
    cfg.model_cfg.tuning_step = 1
    cfg.model_cfg.prompt_path = step1.prompt_path
    cfg.model_cfg.ckpt = None
    cfg.run_cfg.output_dir = step1.output_dir
    cfg.run_cfg.init_lr = step1.init_lr
    cfg.run_cfg.max_epoch = step1.max_epoch
    # Speed: Step-1 is text-only (no CF), so per-epoch valid AUC/uAUC is ~chance
    # and best-checkpoint selection is noise. skip_eval clears valid_splits so the
    # runner trains straight through with no eval (roughly halves wall-clock). We
    # then promote the final-epoch checkpoint to best_ckpt_name in main() so Step-2
    # still finds the file it loads.
    if step1.get("skip_eval", False):
        cfg.run_cfg.valid_splits = []


def promote_final_ckpt_to_best(out_dir, best_ckpt_name):
    """When Step-1 ran with skip_eval (no valid_splits), the runner saved
    checkpoint_<epoch>.pth each epoch but never checkpoint_best.pth. Copy the
    highest-epoch checkpoint to best_ckpt_name so Step-2's ckpt path resolves.

    `out_dir` must be the runner's actual save dir (run_cfg.output_dir/<job_id>,
    e.g. .../vicuna-7b-v1.5/) -- Step-2 loads os.path.join(step1_out, slug,
    best_ckpt_name), so the file has to live in the job_id subfolder."""
    ckpts = glob.glob(os.path.join(out_dir, "checkpoint_*.pth"))
    ckpts = [c for c in ckpts if os.path.basename(c) != best_ckpt_name]

    def _epoch_of(path):
        stem = os.path.basename(path)[len("checkpoint_"):-len(".pth")]
        return int(stem) if stem.isdigit() else -1

    numbered = [c for c in ckpts if _epoch_of(c) >= 0]
    if not numbered:
        raise FileNotFoundError(
            f"skip_eval promotion: no checkpoint_<epoch>.pth found in {out_dir}"
        )
    final = max(numbered, key=_epoch_of)
    best = os.path.join(out_dir, best_ckpt_name)
    shutil.copyfile(final, best)
    print(f"[step1] skip_eval: promoted {os.path.basename(final)} -> {best_ckpt_name}")


@record
def main():
    cfg = Config(parse_args())
    apply_step1_overrides(cfg)
    job_id = derive_job_id_from_llm(cfg)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    task = tasks.setup_task(cfg=cfg)
    datasets = task.build_datasets(cfg=cfg)

    first_dataset_key = list(cfg.datasets_cfg.keys())[0]
    data_dir = cfg.datasets_cfg[first_dataset_key].path
    train_ = pd.read_pickle(os.path.join(data_dir, "train_ood2.pkl"))
    valid_ = pd.read_pickle(os.path.join(data_dir, "valid_ood2.pkl"))
    test_ = pd.read_pickle(os.path.join(data_dir, "test_ood2.pkl"))
    user_num = max(train_.uid.max(), valid_.uid.max(), test_.uid.max()) + 1
    item_num = max(train_.iid.max(), valid_.iid.max(), test_.iid.max()) + 1
    cfg.model_cfg.rec_config.user_num = int(user_num)
    cfg.model_cfg.rec_config.item_num = int(item_num)
    cfg.pretty_print()

    model = task.build_model(cfg=cfg)
    runner = task.build_runner(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    runner.train()

    if cfg.run_cfg.qformer_stage3_step1.get("skip_eval", False) and get_rank() == 0:
        # runner.output_dir is run_cfg.output_dir/<job_id> (the real save dir) --
        # this is exactly where Step-2 looks for best_ckpt_name.
        promote_final_ckpt_to_best(
            str(runner.output_dir), cfg.run_cfg.qformer_stage3_step1.best_ckpt_name
        )


if __name__ == "__main__":
    main()
