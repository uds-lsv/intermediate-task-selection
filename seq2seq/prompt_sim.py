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

from dev.get_prompt_scores import get_best_score_from_log
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', help='File to evaluate on')
    parser.add_argument('--opt_file', help='File to optimizer file')
    parser.add_argument('--scheduler', help='File to scheduler file')
    args = parser.parse_args()
    
    print(args.eval_file)

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               AdapterTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) > 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        
        # Overwrite arguments
        print(f"Overwriting model_name_or_path: {sys.argv[2]}")
        model_args.model_name_or_path = sys.argv[2]
        print(f"Overwriting output_dir: {sys.argv[3]}")
        training_args.output_dir = sys.argv[3]
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    
    # Detecting last checkpoint.
    # last_checkpoint = None
    
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     print("#### last_checkpoint ", last_checkpoint)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         '''
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #         '''
    #         pass 
    #     elif last_checkpoint is not None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    
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
    set_seed(training_args.seed)

    f_lst = []
    sim_score_dict = dict()
    src_tasks = ["mnli", "qqp", "qnli"]
    tgt_tasks = ["boolq", "mrpc", "rte", "cb"]
    src_tmp = "/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base"
    tgt_tmp = "/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-2/t5-base"
    checkpoint_steps = ["5000", "10000", "15000", "20000", 
                        "25000", "30000"]
    
    checkpoints = [
        "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/mnli/150/checkpoint-25500",
        "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/qqp/150/checkpoint-20500",
        "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/qnli/150/checkpoint-8000",

    ]

    task_pairs = [["-" for _ in range(len(tgt_tasks))] for _ in range(len(src_tasks))]

    seeds = ["42", "150", "386"]
    seeds = ["28", "52" , "112"]
    lr   = "2"
    model = "t5-base"
    json_file = "trainer_state.json" 
    output_dir_tmp = "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    output_dir_tmp = "/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    # optimizer = torch.load(args.opt_file)
    # scheduler = torch.load(args.scheduler)

    # print("optimizer", optimizer)
    # print("scheduler", scheduler)


    # sim = [[0.51724952 0.34110346 0.52830184 0.36148578],
    #        [0.63061506 0.45075765 0.65610671 0.3541325 ],
    #        [0.35222611 0.32605103 0.3604984  0.33677474]]

    ### get score ###
    scores = np.zeros((len(tgt_tasks), len(seeds)))
    for i, tgt_t in enumerate(tgt_tasks):
        for j, seed in enumerate(seeds):
            output_dir = output_dir_tmp.format(
                LR=lr, MODEL=model, TASK_NAME=tgt_t, SEED=seed
            )
            output_dir = os.path.join(output_dir, json_file)

            if not os.path.isfile(output_dir):
                logger.info(f"output dir not exist: {output_dir}")

            try:
                score = get_best_score_from_log(output_dir)
                print(f"score: {score}")
                scores[i, j] = score
            except:
                scores[i, j] = 0
                pass
    print(f"seeds: {seeds}")
    print(f"scores: \n{scores}")
    assert 3==2
    ### get score ###


    ### Run ### 
    for seed in seeds:
        logger.info(f"Running seed: {seed}")
        ckpt2matrix = dict()
        ckpt2rank   = dict()
        for step in checkpoint_steps:
            score_matrix = np.zeros((len(src_tasks), len(tgt_tasks)))
            score_rank = np.zeros(score_matrix.shape)
            logger.info(f"rank init \n{score_rank}")
        

            for i, src_t in enumerate(src_tasks):
                # loading 
                # src_path = os.path.join(src_tmp, src_t, seed, f"checkpoint-{step}")
                src_path = checkpoints[i]

                src_prompt = get_prompt(src_path,
                                            model_args,
                                            adapter_args,
                                            data_args,
                                            training_args).mean(axis=0)
                

                for j, tgt_t in enumerate(tgt_tasks):
                    task_pairs[i][j] = f"{src_t}_{tgt_t}"
                    tgt_path = os.path.join(tgt_tmp, tgt_t, seed, f"checkpoint-{step}")
                    f_lst.append(f"{src_path},{tgt_path}")
                    # Load pretrained model and tokenizer
                    # with open(args.eval_file, "r") as f:

                    print(f"Source task: {src_t}")
                    print(f"Model path: {src_path}")
                    print(f"Target task: {tgt_t}")
                    print(f"Model path: {tgt_path}\n")

                    ## get similarity ###
                    tgt_prompt = get_prompt(tgt_path,
                                           model_args,
                                           adapter_args,
                                           data_args,
                                           training_args).mean(axis=0)
                    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                    cosine_score = cos(src_prompt, tgt_prompt)
                    
                    score_matrix[i,j] = cosine_score
            
            # rank
            # print(score_matrix)
            for j in range(score_matrix.shape[1]):
                col = score_matrix[:,j]
                r = rankdata(col, method="dense")
                rank = (r.max()+1) - r
                score_rank[:,j] = rank

                    # compute similarity
            ckpt2matrix[str(step)] = score_matrix
            ckpt2rank[str(step)] = score_rank
        ### get similiary ###                     
        for model_name_or_path in f_lst:
            model_name_or_path = model_name_or_path.strip()
            src_model_name_or_path = model_name_or_path.split(",")[0]
            tgt_model_name_or_path = model_name_or_path.split(",")[1]
            print(f"{src_model_name_or_path}")
            print(f"{tgt_model_name_or_path}\n")

        print("task_pairs",task_pairs)

        for step, matrix in ckpt2matrix.items():
            print(f"checkpoint-{step}")
            print(f"score :\n{matrix}\n")
            print(f"rank :\n{ckpt2rank[step]}\n")

            print()

    # seed_from -> seed_to -> steps ->
    sim_dict = dict()
    for seed_from in seeds:
        sim_dict[seed_from] = dict()
        for seed_to in seeds:
            # src * tgt * steps
            sim_dict[seed_from][seed_to] = np.zeros((len(src_tasks), len(tgt_tasks), len(checkpoint_steps)))

    for seed_from in seeds:
        logger.info(f"Running seed (from): {seed_from}")
        ckpt2matrix = dict()
        for seed_to in seeds:
            logger.info(f"Running seed (to): {seed_to}")
            for i, src_t in enumerate(src_tasks):
                score_matrix = np.zeros((len(src_tasks), len(tgt_tasks)))
                score_rank = np.zeros(score_matrix.shape)
                for j, tgt_t in enumerate(tgt_tasks):
                    for k, step in enumerate(checkpoint_steps):
                        # loading 
                        src_path = os.path.join(src_tmp, src_t, seed_from, f"checkpoint-{step}")
                        src_path = checkpoints[i]
                        src_prompt = get_prompt(src_path,
                                                model_args,
                                                adapter_args,
                                                data_args,
                                                training_args).mean(axis=0)
                    
                        task_pairs[i][j] = f"{src_t}_{tgt_t}"
                        tgt_path = os.path.join(tgt_tmp, tgt_t, seed_to, f"checkpoint-{step}")
                        f_lst.append(f"{src_path},{tgt_path}")
                        # Load pretrained model and tokenizer
                        # with open(args.eval_file, "r") as f:

                        ### get similarity ###
                        tgt_prompt = get_prompt(tgt_path,
                                               model_args,
                                               adapter_args,
                                               data_args,
                                               training_args).mean(axis=0)
                        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                        cosine_score = cos(src_prompt, tgt_prompt)
                        # update dict
                        sim_dict[seed_from][seed_to][i,j,k] = cosine_score
    # iterate as you wants
    logger.info("Fix target task and cross seeds")
    

    for task_idx, task_name in enumerate(tgt_tasks):
        for seed_to in seeds:
            tgt_collect = list()
            for seed_from in seeds:
                logger.info(f"from seed {seed_from} to {seed_to} on {task_name}")
                similarity_list = sim_dict[seed_from][seed_to][:,task_idx,:]
                logger.info(f"similarity over steps (n_src_taks, n_steps)\n{similarity_list}")
                tgt_collect.append(similarity_list)

            tgt_collect = np.stack(tgt_collect)
            logger.info(f"stacked. 3 seeds src to seed {seed_to} {task_name}")
            print(tgt_collect)


    return 

if __name__ == "__main__":
    main()
