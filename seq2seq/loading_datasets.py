# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import wandb
import functools
import logging
import torch
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import sys
import argparse
import subprocess
from pathlib import Path
import json
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union
from typing import Optional, List

from datasets import load_dataset, load_metric, concatenate_datasets
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
import seq2seq
from seq2seq.utils import get_adapter_config
from seq2seq.data import AutoTask
from seq2seq.data import TaskDataCollatorForSeq2Seq
from seq2seq.third_party.trainers import Seq2SeqTrainer
from training_args import AdapterTrainingArguments
from seq2seq.utils import modify_model_after_init, save_training_config
import dataclasses
from dataclasses import dataclass, field, asdict
from transformers import Seq2SeqTrainingArguments
from seq2seq.third_party.models import T5Config, T5ForConditionalGeneration
from seq2seq.data import AutoPostProcessor

import numpy as np

logger = logging.getLogger(__name__)

DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


def run_command(command):
    output = subprocess.getoutput(command)
    return output


TASK_TO_METRICS = {
    "mrpc": ["accuracy", "f1"],
    "cola": ["matthews_correlation"],
    "stsb": ["pearson", "spearmanr"],
    "sst2": ["accuracy"],
    "mnli": ["accuracy"],
    "mnli_mismatched": ["accuracy"],
    "mnli_matched": ["accuracy"],
    "qnli": ["accuracy"],
    "rte": ["accuracy"],
    "wnli": ["accuracy"],
    "qqp": ["accuracy", "f1"],
    "superglue-boolq": ["accuracy"],
    "superglue-rte": ["accuracy"],
    "superglue-cb": ["f1_multiclass", "accuracy"],
    "superglue-copa": ["accuracy"],
    "superglue-multirc": ["f1", "em"],
    "superglue-wic": ["accuracy"],
    "superglue-wsc.fixed": ["accuracy"],
    "superglue-record": ["f1", "em"],
    "cr": ["accuracy"],
    "squad": ["f1", "em"],
    'cxc': [""],
    'drop': ["f1", "em"],
    'winogrande': ["accuracy"],
    'hellaswag': ["accuracy"],
    'cosmosqa': ["accuracy"],
    'race': ["f1", "em"],
}

FILE_NAME_TO_GROUP_NAME = {
    "prompt_tuning": "prompt",
    "prompt_transfer": "prompt-transfer",
    "baseline": "fine-tuning",
    "baseline_transfer": "fine-tuning-transfer",
}


# run_seq2seq parameters.
@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    print_num_parameters: Optional[bool] = field(
        default=False,
        metadata={"help": "If set, print the parameters of " "the model."},
    )
    do_test: Optional[bool] = field(
        default=False, metadata={"help": "If set, evaluates the test performance."}
    )
    split_validation_test: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If set, for the datasets which do not"
            "have the test set, we use validation set as their"
            "test set and make a validation set from either"
            "splitting the validation set into half (for smaller"
            "than 10K samples datasets), or by using 1K examples"
            "from training set as validation set (for larger"
            " datasets)."
        },
    )
    compute_time: Optional[bool] = field(
        default=False, metadata={"help": "If set measures the time."}
    )
    compute_memory: Optional[bool] = field(
        default=False, metadata={"help": "if set, measures the memory"}
    )
    prefix_length: Optional[int] = field(
        default=100, metadata={"help": "Defines the length for prefix tuning."}
    )
    ### test ###
    config: Optional[str] = field(
        default=None, metadata={"help": "Defines a path w.r.t the configuration."}
    )
    ckpt: Optional[str] = field(
        default=None,
        metadata={
            "help": "Defines a checkpoint path. If set, pre-trained weight will be loaded."
        },
    )
    ### test ###


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the evaluation dataset to use (via the datasets library)."
        },
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the evaluation dataset to use (via the datasets library)."
        },
    )
    test_dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the test dataset to use (via the datasets library)."
        },
    )
    test_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the test dataset to use (via the datasets library)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None, metadata={"help": "Number of beams to use for evaluation."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    task_adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from task adapters to the tasks."},
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from tasks to the tasks embeddings."},
    )
    data_seed: Optional[int] = field(
        default=42, metadata={"help": "seed used to shuffle the data."}
    )
    src_task_name: Optional[str] = field(
        default=None, metadata={"help": "Src task name."}
    )

    def __post_init__(self):
        if self.task_name is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        if self.test_max_target_length is None:
            self.test_max_target_length = self.max_target_length


def overwrite_arguments(
    arg_outputs: Iterable,
    known_args: dict,
) -> Tuple[DataClass, ...]:
    """
    helper method that overwrite the value in the existing arguments outputs (List).
    """
    new_output = list()
    for args in arg_outputs:
        # Convert arguments as dict
        current_dict = vars(args)
        # Overwrite the value if key in the corresponding args
        for k, v in vars(known_args).items():
            if hasattr(args, k):
                setattr(args, k, v)
        new_output.append(args)
    return (*new_output,)


def get_best_checkpoint_from_log(
    fname, metric_name="best_model_checkpoint", debugging=False
):
    with open(fname, "r") as f:
        data = json.load(f)
    return data[metric_name]


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            AdapterTrainingArguments,
        )
    )

    # We allow the arguments to overwirte the value in json file.
    # When providing these arguments, we overwrite the default value in json.
    args_parser = argparse.ArgumentParser(description="Parameter Processing")
    args_parser.add_argument("--config", type=str)
    args_parser.add_argument("--model_name_or_path", type=str)
    args_parser.add_argument("--tokenizer_name", type=str)
    args_parser.add_argument("--ckpt", type=str)
    args_parser.add_argument("--output_dir", type=str)
    args_parser.add_argument("--learning_rate", type=float)
    args_parser.add_argument("--seed", type=int)
    args_parser.add_argument("--data_seed", type=int)
    args_parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    args_parser.add_argument("--max_source_length", type=int, default=128)
    args_parser.add_argument("--max_steps", type=int, default=30000)
    args_parser.add_argument("--eval_steps", type=int, default=500)
    args_parser.add_argument("--save_steps", type=int, default=500)
    args_parser.add_argument("--save_total_limit", type=int, default=1)
    args_parser.add_argument("--src_task_name", type=str)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
        print(f"Overwriting output_dir: {sys.argv[2]}")
        training_args.output_dir = sys.argv[2]
    elif len(sys.argv) > 2 and sys.argv[1].endswith(".json"):
        # Parse the defined arguments
        known_args = args_parser.parse_known_args()[0]
        print(
            "# Following keys will be used for updated args instead of the value in json file.",
            vars(known_args),
        )
        json_config = known_args.config
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_outputs = parser.parse_json_file(json_file=os.path.abspath(json_config))
        model_args, data_args, training_args, adapter_args = overwrite_arguments(
            arg_outputs=args_outputs, known_args=known_args
        )
        print("new model_args", model_args)
        print("new data_args", data_args)
        print("new training_args", training_args)
        print("new adapter_args", adapter_args)
    else:
        (
            model_args,
            data_args,
            training_args,
            adapter_args,
        ) = parser.parse_args_into_dataclasses()

    ### setup for wandb ###
    use_pretrained_ckpt = True if "/" in model_args.model_name_or_path else False

    task_name = (
        f"{data_args.src_task_name}_{data_args.task_name}"
        if data_args.src_task_name is not None
        else data_args.task_name
    )
    lr = f"lr={training_args.learning_rate}"
    wandb_run_name = "/".join(training_args.output_dir.split("/")[-4:])
    model_name = "t5-base " if use_pretrained_ckpt else model_args.model_name_or_path
    seed = f"seed={training_args.seed}"
    tags = [model_name, task_name, lr, seed]
    group_name = [
        v for k, v in FILE_NAME_TO_GROUP_NAME.items() if k in training_args.output_dir
    ][0]

    print(f"wandb_run_name: {wandb_run_name}")
    print(f"tags: {tags}")
    print(f"group name: {group_name}")

    wandb.init(name=wandb_run_name, tags=tags, group=group_name)

    # Get the best checkpoint if `best` is checkpoint-best
    if "checkpoint-best" in model_args.model_name_or_path:
        fname = (
            "/".join(model_args.model_name_or_path.split("/")[:-1])
            + "/trainer_state.json"
        )
        model_args.model_name_or_path = get_best_checkpoint_from_log(
            fname, metric_name="best_model_checkpoint", debugging=False
        )
        print(f"### Best checkpoint {model_args.model_name_or_path}")

    # Detecting last checkpoint.
    last_checkpoint = None

    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("#### last_checkpoint ", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            """
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
            """
            pass
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()

    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    logger.info("Set seed %s", training_args.seed)
    set_seed(training_args.seed)

    # add "google/" prefix
    print(model_args.model_name_or_path)
    if "lm-adapt" in model_args.model_name_or_path:
        model_args.model_name_or_path = f"google/{model_args.model_name_or_path}"

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = T5Config.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # overwrite

    config.train_task_adapters = adapter_args.train_task_adapters
    config.prefix_tuning = adapter_args.prefix_tuning
    print("adapter_args.prefix_tuning", adapter_args.prefix_tuning)
    print("adapter_args.init_prefix_from_vocab", adapter_args.init_prefix_from_vocab)

    adapter_config = get_adapter_config(adapter_args, data_args, training_args, config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        adapter_config=adapter_config,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = modify_model_after_init(model, training_args, adapter_args)

    data_args.dataset_name = [data_args.task_name]
    data_args.eval_dataset_name = [data_args.eval_dataset_name]
    data_args.test_dataset_name = [data_args.test_dataset_name]
    data_args.dataset_config_name = [data_args.dataset_config_name]
    data_args.eval_dataset_config_name = [data_args.eval_dataset_config_name]
    data_args.test_dataset_config_name = [data_args.test_dataset_config_name]
    assert len(data_args.dataset_name) == len(data_args.dataset_config_name)
    if data_args.eval_dataset_name is not None:
        assert len(data_args.eval_dataset_name) == len(
            data_args.eval_dataset_config_name
        )
    if data_args.test_dataset_name is not None:
        assert len(data_args.test_dataset_name) == len(
            data_args.test_dataset_config_name
        )

    # Temporarily set max_target_length for training.
    # max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    # def preprocess_function(examples, max_target_length):
    #     model_inputs = tokenizer(
    #         examples["source"],
    #         max_length=data_args.max_source_length,
    #         padding=padding,
    #         truncation=True,
    #     )
    #     # Setup the tokenizer for targets
    #     with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(
    #             examples["target"],
    #             max_length=max_target_length,
    #             padding=padding,
    #             truncation=True,
    #         )
    #     # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    #     # padding in the loss.
    #     if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    #         labels["input_ids"] = [
    #             [(l if l != tokenizer.pad_token_id else -100) for l in label]
    #             for label in labels["input_ids"]
    #         ]
    #     model_inputs["labels"] = labels["input_ids"]
    #     model_inputs["extra_fields"] = examples["extra_fields"]
    #     return model_inputs

    def preprocess_function(examples, max_target_length):
        model_inputs = tokenizer(
            examples["source"],
            max_length=1024,
            padding=False,
            truncation=False,
        )
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=1024,
                padding=False,
                truncation=False,
            )
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["extra_fields"] = examples["extra_fields"]
        return model_inputs


    column_names = ["source", "target", "extra_fields"]
    performance_metrics = {}
    if training_args.do_train:
        train_datasets = [
            AutoTask.get(
                dataset_name, dataset_config_name, seed=data_args.data_seed
            ).get(
                split="train",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_train_samples,
            )
            for dataset_name, dataset_config_name in zip(
                data_args.dataset_name, data_args.dataset_config_name
            )
        ]
        max_target_lengths = [
            AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
                tokenizer=tokenizer, default_max_length=data_args.max_target_length
            )
            for dataset_name, dataset_config_name in zip(
                data_args.dataset_name, data_args.dataset_config_name
            )
        ]
        for i, train_dataset in enumerate(train_datasets):
            train_datasets[i] = train_datasets[i].map(
                functools.partial(
                    preprocess_function, max_target_length=max_target_lengths[i]
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,  # if train_dataset != "superglue-record" else column_names+["answers"],
                load_from_cache_file=not data_args.overwrite_cache,
            )
        train_dataset = concatenate_datasets(train_datasets)

    if training_args.do_eval:
        eval_datasets = {
            eval_dataset: AutoTask.get(
                eval_dataset, eval_dataset_config, seed=data_args.data_seed
            ).get(
                split="validation",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_val_samples,
            )
            for eval_dataset, eval_dataset_config in zip(
                data_args.eval_dataset_name, data_args.eval_dataset_config_name
            )
        }
        max_target_lengths = [
            AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
                tokenizer=tokenizer, default_max_length=data_args.max_target_length
            )
            for dataset_name, dataset_config_name in zip(
                data_args.eval_dataset_name, data_args.eval_dataset_config_name
            )
        ]
        for k, name in enumerate(eval_datasets):
            eval_datasets[name] = eval_datasets[name].map(
                functools.partial(
                    preprocess_function, max_target_length=max_target_lengths[k]
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,  # if name != "superglue-record" else column_names+["answers"],
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_test:
        test_datasets = {
            test_dataset: AutoTask.get(
                test_dataset, test_dataset_config, seed=data_args.data_seed
            ).get(
                split="test",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False if adapter_args.train_task_adapters else True,
                n_obs=data_args.max_test_samples,
            )
            for test_dataset, test_dataset_config in zip(
                data_args.test_dataset_name, data_args.test_dataset_config_name
            )
        }
        max_target_lengths = [
            AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
                tokenizer=tokenizer, default_max_length=data_args.max_target_length
            )
            for dataset_name, dataset_config_name in zip(
                data_args.test_dataset_name, data_args.test_dataset_config_name
            )
        ]
        for k, name in enumerate(test_datasets):
            test_datasets[name] = test_datasets[name].map(
                functools.partial(
                    preprocess_function, max_target_length=max_target_lengths[k]
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    for idx, exm in enumerate(train_dataset):
        inp, lab = exm["input_ids"], exm["labels"]
        print("idx", idx, "inp", tokenizer.batch_decode(inp),"lab", tokenizer.batch_decode(lab))
        break
    print("# of train sets", train_datasets)
    print("# of valid sets", eval_datasets)
    print("# of test sets", test_datasets)


    export_dataset_samples(train_dataset, 
                           list(eval_datasets.values())[0],
                           list(test_datasets.values())[0],
                           tokenizer,
                           task_name)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

def export_dataset_samples(train_datasets,
                           eval_datasets,
                           test_datasets,
                           tokenizer,
                           dataset_name):
    n = 0
    inp_len_dist = list()
    lab_len_dist = list()
    
    with open(f"data_stats/{dataset_name}.train", "w") as wf:
        for idx, exm in enumerate(train_datasets):
            inp, lab = exm["input_ids"], exm["labels"]

            decoded_inp = tokenizer.batch_decode(inp)
            decoded_lab = tokenizer.batch_decode(lab)
            # print("idx", idx, "inp", decoded_inp, "lab", decoded_lab)

            if (idx+1) <= 200:
                wf.write(f"idx: {idx}, inp: {decoded_inp}, lab: {decoded_lab}\n")
                wf.write(f"idx: {idx}, inp_len: {len(decoded_inp)}, lab_len: {len(decoded_lab)}\n")

            inp_len_dist.append(len(decoded_inp))
            lab_len_dist.append(len(decoded_lab))

    avg_inp_len = sum(inp_len_dist) / len(inp_len_dist)    
    

    eighty_pct  = np.percentile(inp_len_dist,80)
    ninety_pct  = np.percentile(inp_len_dist,90)
    ninety_five_pct  = np.percentile(inp_len_dist,95)
    hundred_pct = np.percentile(inp_len_dist,100)

    avg_lab_len = sum(lab_len_dist) / len(lab_len_dist)    
    
    with open(f"data_stats/{dataset_name}.pct", "w") as wf:
        wf.write(f"inp avg: {avg_inp_len}\n")
        wf.write(f"inp 80 pct: {eighty_pct}\n")
        wf.write(f"inp 90 pct: {ninety_pct}\n")
        wf.write(f"inp 95 pct: {ninety_five_pct}\n")
        wf.write(f"inp 100 pct: {hundred_pct}\n")
        wf.write(f"lab avg: {avg_lab_len}\n")


def export_dataset_stats(hf_dataset):
    pass


if __name__ == "__main__":
    main()
