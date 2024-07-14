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
from seq2seq.utils import modify_model_after_init, save_training_config 
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments 
from seq2seq.third_party.models import T5Config, T5ForConditionalGeneration
from seq2seq.data import AutoPostProcessor

from revist.get_prompt_scores import get_best_score_from_log
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
import json
import jsbeautifier


def write_to_json(dict_obj, output_file):
    """
    dict_obj: dictionary-like object
    output_file: file ending with .json
    """
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    with open(output_file, "w") as wf:
        wf.write(
            jsbeautifier.beautify(json.dumps(dict_obj), opts)+"\n")


def get_prompt(model_name_or_path):
    path = f"{model_name_or_path}/prefix_shared.bin"
    # decoder.prefix_emb | prefix_shared | encoder.prefix_emb
    
    device = torch.device("cpu")
    prompt_embs = torch.load(path, map_location=device)["encoder.prefix_emb"]
    return prompt_embs


def create_eval_obj():
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', help='File to evaluate on')
    parser.add_argument('--opt_file', help='File to optimizer file')
    parser.add_argument('--tgt_task', default="rte")
    parser.add_argument('--norm_order', default="fro")
    parser.add_argument('--output_dir', default="/data/users/pjlin/compacter/spot_eval/prompt_similarity_flat")
    args = parser.parse_args()
    
    print(args.eval_file)

    # Opening JSON file
    f = open("test.json")
    data = json.load(f)
    print(data)

    # # test obj
    # o = {
    #     "task": "cb",
    #     "seed": "28",
    #     "src_tasks": ["mnli", "qqp", "cxc"],
    #     "performance": ["88", "83", "82"],
    #     "performance_rank": ["1", "2", "3"]
    # }
    # d = {"5000": o}
    # write_to_json(d, "test.json")

    # assert 300==200
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    
    # Set seed before initializing model.
    set_seed(42)

    f_lst = []
    sim_score_dict = dict()
    src_tasks = ["mnli", "qqp", "qnli"]
    src_tasks = ["mnli", "qqp", "qnli", 
                "superglue-record","cxc","squad",
                "drop","sst2","winogrande",
                "hellaswag", "superglue-multirc", "cosmosqa", 
                "race"]
    tgt_tasks = ["boolq", "mrpc", "rte", "cb"]
    tgt_tasks = ["cb"]
    tgt_tasks = [args.tgt_task]
    src_tmp = "/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base"
    tgt_tmp = "/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-2/t5-base"
    
    # "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    tgt_tmp = "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-2/t5-base"
    checkpoint_steps = ["5000", "10000", "15000", "20000",  "25000", "30000"]         
    
    checkpoints = [
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/mnli/42/checkpoint-25500",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/qqp/386/checkpoint-19500",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/qnli/42/checkpoint-6000",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/superglue-record/42/checkpoint-15000",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/cxc/42/checkpoint-29000",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/squad/386/checkpoint-20000",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/drop/42/checkpoint-29500",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/sst2/42/checkpoint-5500",           
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/winogrande/386/checkpoint-30000",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/hellaswag/42/checkpoint-23500",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/superglue-multirc/386/checkpoint-28500",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/cosmosqa/42/checkpoint-29000",
        "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/race/386/checkpoint-26000",        
    ]

    task_pairs = [["-" for _ in range(len(tgt_tasks))] for _ in range(len(src_tasks))]
    
    seeds = ["28", "52" , "112"]
    lr   = "2"
    model = "t5-base"
    json_file = "trainer_state.json" 
    output_dir_tmp = "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    output_dir_tmp = "/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    output_dir_tmp = "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"

    ### get score ###
    scores = np.zeros((len(tgt_tasks), len(seeds)))
    for j, seed in enumerate(seeds):
        for i, tgt_t in enumerate(tgt_tasks):
            output_dir = output_dir_tmp.format(
                LR=lr, MODEL=model, TASK_NAME=tgt_t, SEED=seed
            )
            output_dir = os.path.join(output_dir, json_file)

            print("output_dir: ", output_dir)
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
    # assert 3==2
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
                src_path = checkpoints[i] # src_path = os.path.join(src_tmp, src_t, seed, f"checkpoint-{step}")
                src_prompt = get_prompt(src_path)
                # src_prompt, _ = p.max(axis=0)
                # src_prompt, _ = torch.cat([even_rows, odd_rows], axis=-1).max(dim=-1)
                # n_row = p.shape[0]
                #src_prompt = torch.tensor([ p[i:i+3,:].mean() for i in range(0, n_row)])

                # src_prompt = p.flatten()
                # print("first row", p[0,:])
                # print("sum of first row", p[0,:].sum())
                # print("mean of first row", (p[0,:].sum())/len(p[0,:]))
                # print("src_prompt_m[0]", src_prompt_m[0])
                # print("src_prompt", src_prompt)
                # print("src_prompt.shape", src_prompt.shape)

                for j, tgt_t in enumerate(tgt_tasks):
                    task_pairs[i][j] = f"{src_t}_{tgt_t}"
                    tgt_path = os.path.join(tgt_tmp, tgt_t, seed, f"checkpoint-{step}")
                    f_lst.append(f"{src_path},{tgt_path}")
                    # Load pretrained model and tokenizer
                    print(f"Source task: {src_t}")
                    print(f"Model path: {src_path}")
                    print(f"Target task: {tgt_t}")
                    print(f"Model path: {tgt_path}\n")

                    ## get similarity ###
                    # tgt_prompt = get_prompt(tgt_path).mean(axis=-1)
                    tgt_prompt = get_prompt(tgt_path)
                    # tgt_prompt, _ = p.max(axis=0)
                    # n_row = p.shape[0]
                    # tgt_prompt = torch.tensor([ p[i:i+3,:].mean() for i in range(0, n_row)])
                    # tgt_prompt = get_prompt(tgt_path).flatten()
                    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                    cosine_score = cos(src_prompt, tgt_prompt).mean()
                    
                    # print("prompt shape", tgt_prompt.shape)
                    # print("cos", cosine_score)
                    # assert 3==2
                    score_matrix[i,j] = cosine_score
            # rank
            # print(score_matrix)
            for j in range(score_matrix.shape[1]):
                col = score_matrix[:,j]
                r = rankdata(col, method="dense")
                rank = (r.max()+1) - r
                score_rank[:,j] = rank
            
            # compute similarity
            # create obj
            ckpt2matrix[str(step)] = {"task": tgt_t,
                                      "seed": seed,
                                      "src_tasks": src_tasks,
                                      "score":score_matrix.flatten().tolist(),
                                      "ranking":score_rank.flatten().tolist()}
            ckpt2rank[str(step)] = score_rank
            eval_file = os.path.join(args.output_dir, f"eval_tgt-{tgt_t}_seed-{seed}.json")
            write_to_json(ckpt2matrix, eval_file)

        ### get similiary ###
        # pairwise checkpoints
        # /data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/qnli/42/checkpoint-6000
        # /data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-2/t5-base/rte/112/checkpoint-30000
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
    ### run ###

    print("ckpt2matrix", ckpt2matrix)
    print("ckpt2rank", ckpt2rank)
    assert 5==2

    ### sim ###
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
    
    print("abc", sim_dict)
    assert 3==22
    # iterate as you wants
    logger.info("Fix target task and cross seeds")
    ### sim ###



    return 

if __name__ == "__main__":
    main()
