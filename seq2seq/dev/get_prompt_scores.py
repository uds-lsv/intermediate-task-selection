"""
This script computes averages socres of prompt tuning from source tasks or target tasks
"""
import os
import sys
import json
import random
import logging
import numpy as np
from scipy.stats import rankdata


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

SEEDS = [42, 150, 386] # src, tgt (step-500, outputs       , soft prompt )
# SEEDS = [28, 52, 112]  # tgt    (step-5k,  prompt_library, transfer )


def get_best_score_from_log(fname, metric_name="best_metric", debugging=False):
    logger.info(f"Get score from log: {fname}")
    with open(fname, "r") as f:
        data = json.load(f)
    return data[metric_name]


# (5e-1 3e-1 1e-1 5e-2 3e-2 1)
def main():
    ### modify here accordingly ###
    lr = "5e-1"
    model = "t5-base"
    tasks = ["mnli", "qqp", "qnli", "superglue-record" ,"cxc", "squad", 
            "drop", "sst2", "winogrande", "hellaswag", "superglue-multirc", "cosmosqa", "race"]
    #tasks = ["boolq", "cola", "stsb", "superglue-wic", "cr", "mrpc", "rte", "superglue-wsc", "superglue-copa", "cb"]
    json_file = "trainer_state.json"
    output_dir_tmp = "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"    
    ### modify here accordingly ###

    num_seed = len(SEEDS)
    
    ten_seeds = [random.sample(SEEDS, num_seed) for _ in range(1)]
    ten_seeds = [SEEDS]

    for rdn_seeds in ten_seeds:
        logger.info(rdn_seeds)

        # n_task * n_seeds
        scores = np.zeros((len(tasks), len(rdn_seeds)))

        for i, task in enumerate(tasks):
            for j, seed in enumerate(rdn_seeds):
                output_dir = output_dir_tmp.format(
                    LR=lr, MODEL=model, TASK_NAME=task, SEED=seed
                )
                output_dir = os.path.join(output_dir, json_file)

                if not os.path.isfile(output_dir):
                    logger.info(f"output dir not exist: {output_dir}")

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
        for i in range(len(tasks)):
            logger.info(
                f"task: {tasks[i]} mean: {score_mean[i]} std: {score_stddev[i]}"
            )



if __name__ == "__main__":
    main()
