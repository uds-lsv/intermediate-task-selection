"""
The script calculate the relative performance
"""

import os
import sys
import json
import random
import logging
import numpy as np
from scipy.stats import rankdata
import argparse

np.set_string_function(
    lambda x: repr(x).replace("(", "")
                     .replace(")", "")
                     .replace("array", "")
                     .replace("       ", " "),
                     repr=False,
)
# np.set_printoptions(separator=", ")
logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

SEEDS = [37, 150, 262, 386, 42]
# SEEDS = [42, 150, 386]
SEEDS = [28, 52, 112]

from local_utils import (
    get_best_score_from_log,
    write_to_json
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', help='File to evaluate on')
    parser.add_argument('--opt_file', help='File to optimizer file')
    parser.add_argument('--tgt_task', default="rte")
    parser.add_argument('--output_dir', default="/data/users/pjlin/compacter/spot_eval/prompt_similarity")
    args = parser.parse_args()
    
    print(args.eval_file)


    ### modify here accordingly ###
    lr = "2"
    model = "t5-base"
    #  "qqp", "qnli", "boolq", "mrpc", "rte", "cb"
    tasks = ["boolq", "mrpc", "rte", "cb"]
    # tasks = ["mnli", "qqp", "qnli"]

    # best checkpoints of each source tasks
    checkpoints = [
        "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/mnli/150/checkpoint-25500",
        "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/qqp/150/checkpoint-20500",
        "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/qnli/150/checkpoint-8000",

    ]
    
    # no_transfer_scores = np.array([
    #     [[62.17125382, 90.11014335, 67.50902527, 87.67745148], # seed 28
    #      [62.17125382, 90.11014335, 67.50902527, 87.67745148],
    #      [62.17125382, 90.11014335, 67.50902527, 87.67745148]],

    #     [[67.24770642, 90.50299323, 60.64981949 , 81.05513672], # seed 52
    #      [67.24770642, 90.50299323, 60.64981949 , 81.05513672],
    #      [67.24770642, 90.50299323, 60.64981949 , 81.05513672]],

    #     [[76.17737003, 85.01583377, 59.566787, 87.67745148], # seed 112
    #      [76.17737003, 85.01583377, 59.566787, 87.67745148],
    #      [76.17737003, 85.01583377, 59.566787, 87.67745148]]
    # ])

    tasks = [
        "mnli_boolq",
        "mnli_mrpc",
        "mnli_rte",
        "mnli_cb",
        "qqp_boolq",
        "qqp_mrpc",
        "qqp_rte",
        "qqp_cb",
        "qnli_boolq",
        "qnli_mrpc",
        "qnli_rte",
        "qnli_cb",
    ]

    tasks = [
        "boolq",
        "cola",
        "stsb",
        "superglue-wic",
        "cr",
        "mrpc",
        "rte",
        "superglue-wsc",
        "superglue-copa",
        "cb",   
    ]
    
    src_tasks = ["superglue-multirc"]

    tmp_list = list()
    for src_task in src_tasks:
        for tgt_task in tasks:
            tmp_list.append(f"{src_task}_{tgt_task}")
    tasks= tmp_list

    json_file = "trainer_state.json"
    ### modify here accordingly ###

    num_seed = len(SEEDS)
    # output_dir_tmp = "/data/users/pjlin/compacter/transfer_outputs/prompt_transfer_tokens_init_best_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    output_dir_tmp = "/data/users/pjlin/compacter/spot_outputs/prompt_transfer_tokens_init_best_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"

    ten_seeds = [random.sample(SEEDS, num_seed) for _ in range(1)]
    ten_seeds = [SEEDS]
    print(ten_seeds)
    
    logger.info(f"random seeds: {SEEDS}")
    # n_task * n_seeds
    scores = np.zeros((len(tasks), len(SEEDS)))

    for i, task in enumerate(tasks):
        for j, seed in enumerate(SEEDS):
            output_dir = output_dir_tmp.format(
                LR=lr, MODEL=model, TASK_NAME=task, SEED=seed
            )
            output_dir = os.path.join(output_dir, json_file)
            print("output_dir", output_dir)
            if not os.path.isfile(output_dir):
                logger.info(f"output dir not exist: {output_dir}")

            score = get_best_score_from_log(output_dir)
            print(f"get score: {score}")
            scores[i, j] = score
            
            try:
                score = get_best_score_from_log(output_dir)
                print(f"get score: {score}")
                scores[i, j] = score
            except:
                scores[i, j] = 0
                pass
        
    print(scores)

    logger.info(f"scores\n{scores}")
    score_mean = scores.mean(axis=-1)
    score_stddev = scores.std(axis=-1)
    score_in_row = list()
    for i in range(len(tasks)):
        score_in_row.append(f"{score_mean[i]:.2f}|{score_stddev[i]:.2f}")
        logger.info(
            f"task: {tasks[i]} mean: {score_mean[i]} std: {score_stddev[i]}"
        )
    print(",".join(score_in_row))

if __name__ == "__main__":
    main()
