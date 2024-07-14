import functools
import logging
import numpy as np
import torch 
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import sys
import argparse
import subprocess
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
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments 
from seq2seq.third_party.models import T5Config, T5ForConditionalGeneration
from seq2seq.data import AutoPostProcessor

from dev.get_scores import get_best_score_from_log
from scipy.stats import rankdata
import torch.nn as nn

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(
    logging.INFO
)

import argparse
from numpy.linalg import norm


# run_seq2seq parameters.
@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    output_dir: Optional[str] = field(
        default="132",
        metadata={"help": ""},
    )
    print_num_parameters: Optional[bool] = field(default=False, metadata={"help": "If set, print the parameters of "
                                                                                 "the model."})
    do_test: Optional[bool] = field(default=False, metadata={"help": "If set, evaluates the test performance."})
    split_validation_test: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, for the datasets which do not"
                                                                    "have the test set, we use validation set as their"
                                                                    "test set and make a validation set from either"
                                                                    "splitting the validation set into half (for smaller"
                                                                    "than 10K samples datasets), or by using 1K examples"
                                                                    "from training set as validation set (for larger"
                                                                    " datasets)."})
    compute_time: Optional[bool] = field(default=False, metadata={"help": "If set measures the time."})
    compute_memory: Optional[bool] = field(default=False, metadata={"help": "if set, measures the memory"})
    prefix_length: Optional[int] = field(default=100, metadata={"help": "Defines the length for prefix tuning."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default="ABC",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
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
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    eval_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the evaluation dataset to use (via the datasets library)."}
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the evaluation dataset to use (via the datasets library)."}
    )
    test_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the test dataset to use (via the datasets library)."}
    )
    test_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the test dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
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
        metadata={"help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."}
    )
    num_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    task_adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from task adapters to the tasks."}
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from tasks to the tasks embeddings."}
    )
    data_seed: Optional[int] = field(default=42, metadata={"help": "seed used to shuffle the data."})

    eval_file: Optional[str] = field(default="eval_file")
    opt_file: Optional[str] = field(default="optimizer.pt")
    scheduler: Optional[str] = field(default="scheduler.pt")

    def __post_init__(self):
        if self.task_name is None:
            pass
            # raise ValueError("Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        if self.test_max_target_length is None:
            self.test_max_target_length = self.max_target_length


def get_prompt(model_name_or_path,
               model_args,
               adapter_args,
               data_args,
               training_args,
    ):
    if "outputs" in model_name_or_path:
        partial_path = model_name_or_path.split("outputs")[-1].replace("/", "_").lstrip("_") + "_prompt.pt"
    elif "library" in model_name_or_path:
        partial_path = model_name_or_path.split("library")[-1].replace("/", "_").lstrip("_") + "_prompt.pt"
    save_path = "/data/users/pjlin/compacter/prompt_cache"
    save_path = os.path.join(save_path, partial_path)
    
    try:
        # logger.info("loading saved prompts weights")
        prompt_embs = torch.load(save_path)
    except:
        #model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        config = T5Config.from_pretrained(
            model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        
        config.train_task_adapters = adapter_args.train_task_adapters
        # config.prefix_tuning = adapter_args.prefix_tuning
        config.prefix_tuning = True
        adapter_args.prefix_tuning = True # need to set True to get `adapter_config`
        adapter_config = get_adapter_config(adapter_args, data_args, training_args, config)
        # print("adapter_config", adapter_config)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        
        model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            adapter_config=adapter_config
        )
        model.resize_token_embeddings(len(tokenizer))
        model = modify_model_after_init(model, training_args, adapter_args)
        prompt_embs = model.prefix_shared

        logger.info("saving prompt_embs {save_path}")
        
        torch.save(prompt_embs, save_path)
    return prompt_embs

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--eval_file', help='File to evaluate on')
    # parser.add_argument('--opt_file', help='File to optimizer file')
    # parser.add_argument('--scheduler', help='File to scheduler file')
    # args = parser.parse_args()
    
    # print(args.eval_file)

    # # See all possible arguments in src/transformers/training_args.py
    # # or by passing the --help flag to this script.
    # # We now keep distinct sets of args, for a cleaner separation of concerns.
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
    #                            AdapterTrainingArguments))
    
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # elif len(sys.argv) > 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        
    #     # Overwrite arguments
    #     print(f"Overwriting model_name_or_path: {sys.argv[2]}")
    #     model_args.model_name_or_path = sys.argv[2]
    #     print(f"Overwriting output_dir: {sys.argv[3]}")
    #     training_args.output_dir = sys.argv[3]
    # else:
    #     model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )
    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    
    # # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )
    
    # # Set the verbosity to info of the Transformers logger (on main process only):
    # if is_main_process(training_args.local_rank):     
    #     transformers.utils.logging.set_verbosity_info()
    
    # logger.info("Training/evaluation parameters %s", training_args)
    
    # # Set seed before initializing model.
    # set_seed(training_args.seed)

    f_lst = []
    sim_score_dict = dict()
    src_tasks = ["mnli", "qqp", "qnli"]
    tgt_tasks = ["boolq", "mrpc", "rte", "cb"]

    # src_tmp = "/data/users/pjlin/compacter/task_selection/taskemb_prompt_tuning/t5-base/"
    # tgt_tmp = "/data/users/pjlin/compacter/task_selection/taskemb_prompt_tuning/t5-base/"

    src_tmp = "/data/users/pjlin/compacter/task_selection/taskemb_trained_prompt_tuning/t5-base/"
    tgt_tmp = "/data/users/pjlin/compacter/task_selection/taskemb_trained_prompt_tuning/t5-base/"
    emb_file = "prefix_shared.npy"

    seeds = ["28", "52" , "112"]
    task_combination = [ f"{src_t}_{tgt_t}" for src_t in src_tasks for tgt_t in tgt_tasks]
    all_tgt_tasks    =  [ f"{tgt_t}" for _ in src_tasks for tgt_t in tgt_tasks]
    logger.info(f"task combination: {task_combination}")
    

    src_taskemb_files = list()
    tgt_taskemb_files = list()

    for seed in seeds:
        for pair in task_combination:
            fname = os.path.join(src_tmp, pair, seed, emb_file)
            src_taskemb_files.append(fname)

        for tgt_task in all_tgt_tasks:
            fname = os.path.join(tgt_tmp, tgt_task, seed, emb_file)
            tgt_taskemb_files.append(fname)

    sim_scores = list()
    sim_scores_avg_tok = list()
    sim_scores_avg_dim = list()
    for src_f, tgt_f in zip(src_taskemb_files, tgt_taskemb_files):
        logger.info(f"\nsource file: {src_f}\ntarget file: {tgt_f}")

        # compute similarity
        src_emb, tgt_emb = np.load(src_f), np.load(tgt_f)
        cos_sim = (src_emb @ tgt_emb.T) / (norm(src_emb)*norm(tgt_emb))
        print(cos_sim)

        sim_scores.append(cos_sim)

        src_emb_2, tgt_emb_2 = np.copy(src_emb), np.copy(tgt_emb)
        src_emb_2, tgt_emb_2 = src_emb_2.reshape((100,-1)), tgt_emb_2.reshape((100,-1))
        # (100, )
        src_emb_avg_tok = src_emb_2.mean(-1)
        tgt_emb_avg_tok = tgt_emb_2.mean(-1)
        print(tgt_emb_avg_tok.shape)
        cos_sim = (src_emb_avg_tok @ tgt_emb_avg_tok.T) / (norm(src_emb_avg_tok)*norm(tgt_emb_avg_tok))
        sim_scores_avg_tok.append(cos_sim)

        # (768,)
        src_emb_avg_dim = src_emb_2.mean(0)
        tgt_emb_avg_dim = tgt_emb_2.mean(0)
        print(src_emb_avg_dim.shape)
        cos_sim = (src_emb_avg_dim @ tgt_emb_avg_dim.T) / (norm(src_emb_avg_dim)*norm(tgt_emb_avg_dim))
        sim_scores_avg_dim.append(cos_sim)
        
        
    mtx = np.array(sim_scores).reshape((len(seeds), len(src_tasks), len(tgt_tasks)))
    mtx2 = np.array(sim_scores_avg_tok).reshape((len(seeds), len(src_tasks), len(tgt_tasks)))
    mtx3 = np.array(sim_scores_avg_dim).reshape((len(seeds), len(src_tasks), len(tgt_tasks)))
    
    print(mtx)
    print(mtx2)
    print(mtx3)
    assert 3==22
    task_pairs = [["-" for _ in range(len(tgt_tasks))] for _ in range(len(src_tasks))]



if __name__ == "__main__":
    main()
