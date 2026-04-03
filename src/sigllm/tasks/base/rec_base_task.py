import json
import logging
import os
from typing import Optional
from sklearn.metrics import roc_auc_score
import torch.distributed as dist
from sigllm import datasets
from sigllm.common import registry
from sigllm.common.data_utils import move_to_cuda
from sigllm.common.dist_utils import *
from sigllm.common.logger import MetricLogger, SmoothedValue
from sigllm.common.logging_utils import NotebookLogger
from sigllm.models.multimodal.qformer_rec_llm import QRecLLM
from sigllm.pipelines.rec.train_rec_baseline import calculate_user_auc

LOGGER = NotebookLogger.rich_logger("sigllm.tasks.base.rec_base_task")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)


class RecBaseTask:
    def __init__(self):
        super().__init__()
        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_runner(self, cfg, job_id, task, model, datasets):
        runner_cls =  registry.get_runner_class(cfg.run_cfg.get("runner", "rec_runner_base"))
        return runner_cls(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    
    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)
    
    def build_datasets(self, cfg):
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        evaluate_only = cfg.run_cfg.evaluate

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name, dataset_config in datasets_config.items():
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets(evaluate_only=evaluate_only)
            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio
            datasets[name] = dataset

        return datasets
    
    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss
    
    def valid_step(self, model, samples):
        outputs = model.generate_for_samples(samples)
        return outputs
    
    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        use_amp = scaler is not None

        # Initialize logging utilities
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        header = f"Train Epoch: [{epoch}]"
        
        # Training loop
        for step, samples in enumerate(metric_logger.log_every(data_loader, log_freq, header)):
            # 1. Move data to GPU if enabled and update learning rate scheduler
            if cuda_enabled:
                samples = move_to_cuda(samples)
            lr_scheduler.step(cur_epoch=epoch, cur_step=step)

            # 1. Forward pass with Automatic Mixed Precision (AMP)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)
                loss = loss / accum_grad_iters # Chia loss để hỗ trợ Gradient Accumulation

            # 3. Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 4. Optimizer step (every accum_grad_iters)
            if (step + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # 5. Logging
            metric_logger.update(loss=loss.item() * accum_grad_iters)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Sync stats across all distributed processes (if any)
        metric_logger.synchronize_between_processes()
        log_step(f"Averaged stats: {metric_logger.global_avg()}")
        
        return {k: f"{meter.global_avg:.4f}" for k, meter in metric_logger.meters.items()}
    
    @torch.no_grad()
    def evaluate(self, model, data_loader, cuda_enabled=True):
        model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        all_results = []

        for data_loader in data_loader.loaders:
            raw_outputs = self._collect_predictions(model, data_loader, metric_logger, header, cuda_enabled)
            combined_data = self._gather_distributed_data(raw_outputs)
            metrics = self._compute_metrics(combined_data)
            
            metric_logger.synchronize_between_processes()
            logging.info(
                f"Averaged stats: {metric_logger.global_avg()} "
                f"***auc: {metrics.get('auc', 0):.4f} ***uauc: {metrics.get('uauc', 0):.4f}"
            )        
            
            all_results = {
            'agg_metrics': metrics.get('auc', -metric_logger.meters['loss'].global_avg),
            'acc': metric_logger.meters.get('acc', SmoothedValue()).global_avg,
            'loss': metric_logger.meters['loss'].global_avg,
            'uauc': metrics.get('uauc', 0)
        }
        
        return all_results

    def _collect_predictions(self, model, data_loader, logger, header, cuda_enabled):
        results = {'logits': [], 'labels': [], 'users': []}
        
        for samples in logger.log_every(data_loader, 10, header):
            if cuda_enabled:
                samples = move_to_cuda(samples, cuda_enabled=cuda_enabled)
            eval_output = self.valid_step(model=model, samples=samples)
            
            logger.update(loss=eval_output['loss'].item())

            if 'logits' in eval_output:
                logits = eval_output['logits']
                labels = samples['label']
                
                results['logits'].append(logits.detach())
                results['labels'].append(labels.detach())
                results['users'].append(samples['UserID'].detach())
                
                acc = ((logits > 0.5).float() == labels).float().mean()
                logger.update(acc=acc.item())
                
            torch.cuda.empty_cache()
            
        return {k: torch.cat(v, dim=0) if v else None for k, v in results.items()}


    def _gather_distributed_data(self, data):
        if data['logits'] is None:
            return data
        if not is_dist_avail_and_initialized():
            return {k: v.cpu().numpy() for k, v in data.items()}
        gathered = {}
        for key, tensor in data.items():
            world_size = dist.get_world_size()
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(tensor_list, tensor)
            gathered[key] = torch.cat(tensor_list, dim=0).cpu().numpy()
        return gathered

    def _compute_metrics(self, data):
        if data['logits'] is None:
            return {}
        auc = roc_auc_score(data['labels'], data['logits'])
        uauc, _, _ = calculate_user_auc(data['users'], data['logits'], data['labels'])
        return {'auc': auc, 'uauc': uauc}
    
    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        final_file = os.path.join(result_dir, f"{filename}.json")
        rank_file = os.path.join(result_dir, f"{filename}_rank{get_rank()}.json")

        with open(rank_file, "w") as f:
            json.dump(result, f)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            log_step("Merging results from all ranks...")
            combined_result = []
            for r in range(get_world_size()):
                path = os.path.join(result_dir, f"{filename}_rank{r}.json")
                with open(path, "r") as f:
                    combined_result.extend(json.load(f))

            if remove_duplicate:
                unique_res = {res[remove_duplicate]: res for res in combined_result}
                combined_result = list(unique_res.values())

            with open(final_file, "w") as f:
                json.dump(combined_result, f)
            log_step(f"Final result saved to {final_file}")

        return final_file
