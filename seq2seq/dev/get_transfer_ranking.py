"""
Generate transfer ranking
"""

import os
import sys
import json
import random
import logging
import numpy as np
from scipy.stats import rankdata
import jsbeautifier
import sys
import argparse

from local_utils import (
    get_best_score_from_log,
    write_to_json,
    read_json
)

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


# SEEDS = [42, 150, 386]
SEEDS = [28, 52, 112]

# (5e-1 3e-1 1e-1 5e-2 3e-2 1)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', help='File to evaluate on')
    parser.add_argument('--opt_file', help='File to optimizer file')
    parser.add_argument('--tgt_task', default="rte")
    parser.add_argument('--output_dir', default="/data/users/pjlin/compacter/spot_eval/transfer_ranking")
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
    # low, boolq on seed 28, 62.17125382
    # ok, qnli -> boolq on seed 28, 76.94
    # ok, qnli -> boolq on seed 28, 77.88
    # low, qnli -> boolq on seed 112, 63.27
    # seeds in each row [42,150,386]
    # boolq mrpc rte cb
    output_dir = "/data/users/pjlin/compacter/spot_eval/transfer_ranking"
    no_transfer_scores = np.array([
        [[62.17125382, 90.11014335, 67.50902527, 87.67745148], # seed 28
         [62.17125382, 90.11014335, 67.50902527, 87.67745148],
         [62.17125382, 90.11014335, 67.50902527, 87.67745148]],

        [[67.24770642, 90.50299323, 60.64981949 , 81.05513672], # seed 52
         [67.24770642, 90.50299323, 60.64981949 , 81.05513672],
         [67.24770642, 90.50299323, 60.64981949 , 81.05513672]],

        [[76.17737003, 85.01583377, 59.566787, 87.67745148], # seed 112
         [76.17737003, 85.01583377, 59.566787, 87.67745148],
         [76.17737003, 85.01583377, 59.566787, 87.67745148]]
    ])

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

    tgt_tasks = [args.tgt_task]
    src_tasks = ["mnli", "qqp", "qnli", 
                "superglue-record","cxc","squad",
                "drop","sst2","winogrande",
                "hellaswag", "superglue-multirc", "cosmosqa", 
                "race"]

    tmp_list = list()
    for src_task in src_tasks:
        for tgt_task in tgt_tasks:
            tmp_list.append(f"{src_task}_{tgt_task}")
    tasks = tmp_list

    json_file = "trainer_state.json"
    ### modify here accordingly ###

    num_seed = len(SEEDS)
    # get baseline socre (no transfer)
    baseline_output_dir_tmp = "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"

    output_dir_tmp = "/data/users/pjlin/compacter/transfer_outputs/prompt_transfer_tokens_init_best_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    # output_dir_tmp = "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    # output_dir_tmp = "/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    output_dir_tmp = "/data/users/pjlin/compacter/spot_outputs/prompt_transfer_tokens_init_best_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"

    ten_seeds = [random.sample(SEEDS, num_seed) for _ in range(1)]
    ten_seeds = [SEEDS]
    
    logger.info(SEEDS)
    # n_task * n_seeds
    baseline_scores= np.zeros((len(tasks), len(SEEDS))) 
    scores         = np.zeros((len(tasks), len(SEEDS)))

    for i, task in enumerate(tasks):
        for j, seed in enumerate(SEEDS):
            # baseline (no transfer)
            bl_output_dir = baseline_output_dir_tmp.format(
                LR=lr, MODEL=model, TASK_NAME=task.split("_")[-1], SEED=seed
            )
            bl_output_dir = os.path.join(bl_output_dir, json_file)


            # transfer
            output_dir = output_dir_tmp.format(
                LR=lr, MODEL=model, TASK_NAME=task, SEED=seed
            )
            output_dir = os.path.join(output_dir, json_file)

            if not os.path.isfile(output_dir):
                logger.info(f"output dir not exist: {output_dir}")

            try:
                baseline_score = get_best_score_from_log(bl_output_dir)
                print(f"get baseline score: {baseline_score}")
                baseline_scores[i, j] = baseline_score

                score = get_best_score_from_log(output_dir)
                print(f"get score: {score}")
                scores[i, j] = score
            except:
                scores[i, j] = 0
                pass
    
    relative_score = ((scores-baseline_scores)/baseline_scores)*100
    n_improvement = ( (scores - baseline_scores)>0).astype(float)
    
    score_rank = np.zeros(scores.shape)
    for j in range(scores.shape[1]):
        col = scores[:,j]
        r = rankdata(col, method="dense")
        rank = (r.max()+1) - r
        score_rank[:,j] = rank

        tgt_t = tgt_tasks[0]
        seed = SEEDS[j]
        
        d = dict()
        d["best"] = {"task": tgt_t,
                    "seed": seed,
                    "src_tasks": src_tasks,
                    "baseline_score":baseline_scores[:,j].flatten().tolist(),
                    "score":scores[:,j].flatten().tolist(),
                    "relative_score":relative_score[:,j].flatten().tolist(),
                    "ranking":score_rank[:,j].flatten().tolist()}
        eval_file = os.path.join(args.output_dir, f"eval_tgt-{tgt_t}_seed-{seed}.json")
        
        print("d",d)
        print("eval_file", eval_file)
        write_to_json(d, eval_file)
    print("ranking", score_rank)
    
    ### relative improv. / no of improv. ###

    logger.info(f"all baseline scores\n{baseline_scores}")
    logger.info(f"all transfer scores\n{scores}")
    logger.info(f"transfer - baseline \n{relative_score}")
    logger.info(f"n_improvement \n{n_improvement}")
    logger.info(f"relative_score \n{relative_score.flatten()}")
    print(n_improvement.sum(axis=0))
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

