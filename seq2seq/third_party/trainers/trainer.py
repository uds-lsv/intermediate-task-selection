from typing import Dict, List, Optional
import numpy as np 
import time
import torch
import collections
from packaging import version
from torch.utils.data.dataset import Dataset

from transformers import Trainer
from transformers import logging
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_utils import (
    speed_metrics,
    EvalLoopOutput,
    denumpify_detensorize
)
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_numpify,
    nested_truncate,
    nested_concat,
    IterableDatasetShard
)
from .trainer_utils import EvalPrediction


from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
#from transformers.integrations import deepspeed_init # 4.6.0
from transformers.deepspeed import deepspeed_init     # 4.7.0

import os
from tqdm import tqdm, trange

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


def compute_Fisher(args, model, input_mask, total_tokens):
    outputs = {}

    # print(len([n for n,p in model.named_parameters()]))
    # print(len([n for n,p in model.base_model.named_parameters()]))
    # do not matter use model.base_model or model
    base_model = model.base_model
    for name, parameter in base_model.named_parameters():
        # print("name", name)

        if parameter.requires_grad:
            score = parameter.grad if args.feature_type == "grads" else parameter
            if score is not None and name not in outputs:
                score = score ** args.pow
                outputs[name] = score
    # activations
    # for key in ["multihead_output", "layer_output"]:
    #     model_outputs = base_model._get_model_outputs(key=key)

    #     for i in range(base_model.config.num_hidden_layers):
    #         name = "encoder.layer.{}.{}".format(i, key)
    #         model_outputs_i = model_outputs[i].grad if args.feature_type == "grads" else model_outputs[i]

    #         if model_outputs_i is not None:
    #             score = torch.einsum(
    #                 "ijk,ij->ijk", [model_outputs_i, input_mask.float()]  # batch_size x max_seq_length x hidden_size
    #             )  # batch_size x max_seq_length
    #             if score is not None and name not in outputs:
    #                 score = score.sum(0).sum(0)
    #                 score = score ** args.pow
    #                 # normalize
    #                 score = score / total_tokens
    #                 outputs[name] = score

    # task-specific layer
    # for name, parameter in model.named_parameters():
    #     if model.config.model_type not in name:
    #         score = parameter.grad if args.feature_type == "grads" else parameter
    #         if score is not None and name not in outputs:
    #             score = score ** args.pow
    #             outputs[name] = score
    
    return outputs

def compute_Fisher_with_labels(args, model, input_mask, loss):
    total_tokens = input_mask.float().detach().sum().data if input_mask is not None else None

    model.zero_grad()
    loss.backward()

    outputs = compute_Fisher(args, model, input_mask, total_tokens)
    return outputs



class BaseTrainer(Trainer):
    def __init__(self, evaluation_metrics=[], data_info=None, *args, **kwargs):
        """When doing evaluation, it computes average of list of metrics 
        given in evaluation_metrics and adds it to the dictionary of results.
        Trainer class then use this average metric to save the best model."""
        super().__init__(*args, **kwargs)
        self.evaluation_metrics = evaluation_metrics 
        self.data_info = data_info

    def get_data_info(self, metric_key_prefix):
        """Returns the data information required to make the predictions/labels
        suitable for the evaluation."""
        if self.data_info is not None:
            return self.data_info[metric_key_prefix]
        return None     
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            ### We save prompt prefix weights. The three variables are same objects.
            # `encoder.prefix_emb` and `decoder.prefix_emb` calls the `prefix_shared`
            ###
            self.model.save_pretrained(output_dir, state_dict=state_dict)
            prefix_shared_state_dict = {"prefix_shared": self.model.prefix_shared,
                                        "encoder.prefix_emb": self.model.encoder.prefix_emb,
                                        "decoder.prefix_emb": self.model.decoder.prefix_emb}
            path = f"{output_dir}/prefix_shared.bin"
            torch.save(prefix_shared_state_dict, path)
            logger.info(f"Model weights saved in {path}")

            #logger.info(f"is PreTrainedModel")
            #logger.info(f"output_dir {output_dir}")
            #logger.info(f"state_dict {state_dict}")

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, output.num_samples))
        if len(self.evaluation_metrics) != 0:
           selected_metrics = [output.metrics[metric_key_prefix+"_"+k] for k in self.evaluation_metrics if metric_key_prefix+"_"+k in output.metrics]
           assert len(selected_metrics) >= 1, "at least one metric should be selected to compute the average_metrics."
           output.metrics.update({metric_key_prefix+'_average_metrics': np.mean(selected_metrics)})         
    
        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels, 
            data_info=self.get_data_info(metric_key_prefix)))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
   
    def run_embeddings(self, emb_args):
        """
        task embedding
        """
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Get model
        model = self._wrap_model(self.model, training=False)
        # model = self.model
        # model.eval()
        model.train()
        
        logger.info("***** Compute TaskEmb *****")
        logger.info("Num batches = %d", len(train_dataloader))
        logger.info("Batch size = %d", train_dataloader.batch_size)
        
        ### compute taskemb ###
        total_num_examples = 0
        model.zero_grad()
        train_iterator = trange(int(1), desc="Epoch", disable=False)
        global_feature_dict = {}
        # Main evaluation loop
        for _ in train_iterator:
            num_examples = 0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for step, inputs in enumerate(epoch_iterator):
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    total_num_examples += observed_batch_size

                # move to device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)

                # Prediction step
                # loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=True, ignore_keys=None)
                outputs = model(**inputs)
                loss = outputs[0]

                feature_dict = compute_Fisher_with_labels(emb_args, model, None, loss)

                if len(global_feature_dict) == 0:
                    for key in feature_dict:
                            global_feature_dict[key] = feature_dict[key].detach().cpu().numpy()
                else:
                    for key in feature_dict:
                        global_feature_dict[key] += feature_dict[key].detach().cpu().numpy()
                
            model.zero_grad()
            num_examples += inputs["input_ids"].size(0)
        total_num_examples += num_examples

        # Normalize
        for key in global_feature_dict:
            global_feature_dict[key] = global_feature_dict[key] / total_num_examples


        feature_dict = global_feature_dict
        # compute taskemb
        # feature_dict = compute_taskemb(emb_args, dataset_manager, model, run_args)
        ### compute taskemb ###

        # save all taskemb
        for key in feature_dict:
            np.save(os.path.join(self.args.output_dir, "{}.npy".format(key)), feature_dict[key].flatten())


