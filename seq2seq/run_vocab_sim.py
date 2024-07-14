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
from collections import Counter
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

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

    tokenization_level: Optional[str] = field(
        default=None, metadata={"help": "Tokenize text in word- or token-level."}
    )
    
    ngram: Optional[str] = field(
        default=1, metadata={"help": "Use unigram or bigram."}
    )
    
    num_token: Optional[str] = field(
        default=1000, metadata={"help": "Number of token (or word) used for vocabulary overlapping."}
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


def write_token_frequency_to_file(fname, cnt, size):
    with open(fname, "w") as wf:
        for tok, freq in cnt.most_common(size):
            wf.write(f"{tok},{freq}\n")


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
    args_parser.add_argument("--max_steps", type=int, default=30000)
    args_parser.add_argument("--eval_steps", type=int, default=500)
    args_parser.add_argument("--save_steps", type=int, default=500)
    args_parser.add_argument("--save_total_limit", type=int, default=1)
    args_parser.add_argument("--src_task_name", type=str)
    args_parser.add_argument("--tokenization_level", type=str)
    args_parser.add_argument("--ngram", type=str)
    args_parser.add_argument("--num_token", type=int)

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

    # model = T5ForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    #     adapter_config=adapter_config,
    # )
    # model.resize_token_embeddings(len(tokenizer))
    # model = modify_model_after_init(model, training_args, adapter_args)

    print("all dataset", data_args.task_name)
    data_args.dataset_name = [ t.strip() for t in data_args.task_name.split(",") ] if "," in data_args.task_name else [data_args.task_name]
    data_args.dataset_name = ["mnli", "qqp", "qnli", "superglue-boolq", "mrpc", "rte", "superglue-cb"]
    print("all dataset", data_args.dataset_name)
    data_args.eval_dataset_name = [data_args.eval_dataset_name]
    data_args.test_dataset_name = [data_args.test_dataset_name]
    data_args.dataset_config_name = [data_args.dataset_config_name]
    data_args.dataset_config_name = [['en'], ['en'], ['en'], ['en'], ['en'], ['en'], ['en']]
    # data_args.dataset_config_name = ["en", "en"]
    data_args.eval_dataset_config_name = [data_args.eval_dataset_config_name]
    data_args.test_dataset_config_name = [data_args.test_dataset_config_name]
    print(data_args.dataset_config_name)
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

    
    ###
    # the function has been revisit to preprocess the textual examples
    # by chossing data_args.num_token
    # data_args.n_gram
    # data_args.tokenization_level
    ###
    SOURCE_TASKS = ["mnli", "qqp", "qnli"]
    TARGET_TASKS = ["superglue-boolq", "mrpc", "rte", "superglue-cb"]

    stop_words = set(stopwords.words('english'))
    
    ### assertations
    assert data_args.tokenization_level in ["word", "subword"]
    assert data_args.ngram              in ["unigram", "bigram"]
    
    def preprocess_function(examples, 
                            max_target_length,
                            tokenization_level,
                            ngram):
        def fn(sent):
            return sent.translate(str.maketrans('', '', string.punctuation))

        # standard preprocessing
        model_inputs = tokenizer(
            examples["source"],
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )
            
        # tokenize and remove punctuation
        if tokenization_level == "word":
            input_tokens = [ word_tokenize(fn(sent)) for sent in examples["source"]]        
        elif tokenization_level == "subword":
            input_tokens = [ tokenizer.tokenize(fn(sent)) for sent in examples["source"]]
        
        input_bigram_tokens = list()
        # Remove stop words
        for sent_idx in range(len(input_tokens)):
            sent = input_tokens[sent_idx]
            filtered_input_tokens = [ w for w in sent if not w.lower() in stop_words]

            if ngram == "unigram":
                input_tokens[sent_idx] = filtered_input_tokens
            # add bigram example
            elif ngram == "bigram":
                input_tokens[sent_idx] = [filtered_input_tokens[i]+" "+filtered_input_tokens[i+1] for i in range(len(filtered_input_tokens)-1)]

        # Get some examples
        print("origin example 0:", examples["source"][0])
        print("preprocessing unigram example 0:", input_tokens[0])
        
        print("origin example 12:", examples["source"][12])
        print("preprocessing unigram example 12:", input_tokens[12])
        
        model_inputs["input_tokens"] = input_tokens
    
        assert type(input_tokens[0] == list)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=max_target_length,
                padding=padding,
                truncation=True,
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


    print("a", data_args.dataset_name)
    print("b", data_args.dataset_config_name)

    # column_names = ["source", "target", "extra_fields"]
    column_names = []
    performance_metrics = {}
    if training_args.do_train:
        train_datasets = [
            AutoTask.get(
                dataset_name, dataset_config_name, seed=data_args.data_seed
            ).get(
                split="train",
                split_validation_test=training_args.split_validation_test,
                add_prefix=False,
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
                    preprocess_function, 
                    max_target_length=max_target_lengths[i],
                    tokenization_level=data_args.tokenization_level,
                    ngram=data_args.ngram
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,  # if train_dataset != "superglue-record" else column_names+["answers"],
                load_from_cache_file=not data_args.overwrite_cache,
            )
        # train_dataset = concatenate_datasets(train_datasets)


    print("# of train sets", train_datasets)
    for d in train_datasets:
        print(d["task"][0])


    ### create vocab set
    source_sets = list()
    target_sets = list()    

    if (data_args.tokenization_level == "subword" and
        data_args.ngram              == "unigram"):
        SOURCE_TASKS.insert(0, "PT")
        SOURCE_TASKS_EXCEPT_PTRAIN = SOURCE_TASKS[1:]
    else:
        SOURCE_TASKS_EXCEPT_PTRAIN = SOURCE_TASKS[:]

    logger.info(f"Getting source tasks: {SOURCE_TASKS_EXCEPT_PTRAIN}")
    logger.info(f"Getting target tasks: {TARGET_TASKS}")
    
    for dataset_name in SOURCE_TASKS_EXCEPT_PTRAIN:
        idx = data_args.dataset_name.index(dataset_name)
        # dataset class to set of tokens (or words)
        cnt = Counter()
        for example in train_datasets[idx]["input_tokens"]:
            # print("source tokens", example)
            cnt.update(example)

        # write to file
        fname = training_args.output_dir+f"/{dataset_name}.vocab"
        write_token_frequency_to_file(fname=fname,
                                      cnt=cnt,
                                      size=data_args.num_token)
        token_set = set([ w[0] for w in cnt.most_common(data_args.num_token)])
        source_sets.append(token_set)

    for dataset_name in TARGET_TASKS:
        idx = data_args.dataset_name.index(dataset_name)
        # dataset class to set of tokens (or words)
        cnt = Counter()
        for example in train_datasets[idx]["input_tokens"]:
            cnt.update(example)
        # write to file
        fname = training_args.output_dir+f"/{dataset_name}.vocab"
        write_token_frequency_to_file(fname=fname,
                                      cnt=cnt,
                                      size=data_args.num_token)
        token_set = set([ w[0] for w in cnt.most_common(data_args.num_token)])
        target_sets.append(token_set)
    
    # Add pretrain vocab set if token #
    special_tokens = list(tokenizer.special_tokens_map.values())
    if (data_args.tokenization_level == "subword" and
        data_args.ngram              == "unigram"):
        pretrain_vocab = list()

        sorted_vocab = list(sorted(tokenizer.vocab.items(), key=lambda item: item[1]))[:data_args.num_token+5000]
        sorted_vocab = [t for t,f in sorted_vocab]
        
        for tok in sorted_vocab[:data_args.num_token+5000]:
            preprocessed_tok = tok.lower()
            if (preprocessed_tok not in stop_words and
                preprocessed_tok not in string.punctuation and
                preprocessed_tok not in special_tokens):

                pretrain_vocab.append(tok)
    
        pretrain_vocab = pretrain_vocab[:data_args.num_token]
        print(pretrain_vocab)

        assert len(pretrain_vocab) == data_args.num_token
        vocab_set = set(pretrain_vocab)
        source_sets.insert(0, vocab_set)

    overlapping_matrix = np.zeros((len(SOURCE_TASKS), len(TARGET_TASKS)))
    
    # compute overlapping
    for i in range(len(source_sets)):
        for j in range(len(target_sets)):
            src_set, tgt_set = source_sets[i], target_sets[j]
            intersection_set = src_set.intersection(tgt_set)        
            # Computer the intersection
            overlapping_matrix[i,j] = (len(intersection_set)/data_args.num_token)*100
    # Save text
    print("result", overlapping_matrix)
    fname = training_args.output_dir+"/overlap.txt"
    with open(fname, "w") as wf:
        wf.write(str(overlapping_matrix))
    #np.savetxt(fname, overlapping_matrix)


if __name__ == "__main__":
    main()
