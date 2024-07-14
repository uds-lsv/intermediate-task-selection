"""
Evaluation of task selection
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
    read_json,
    calculate_dcg
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

# sort by data szie
REFERENCE_SCORE_SIZE = {
    "mnli": 13, 
    "qqp": 12, 
    "qnli": 11,
    "superglue-record": 10,
    "cxc": 8,
    "squad":9,            
    "drop":7,
    "sst2":6,
    "winogrande":5,
    "hellaswag":4, 
    "superglue-multirc": 3, 
    "cosmosqa":1, 
    "race": 2
}



def compute_ndcg_score(transfer_scores, similarity_scores, has_done_check):
    # get idx ; higher similarity, lower index
    argsort_indices = np.array(similarity_scores).argsort()[::-1]
    # similarity-ranked list; (higher similar infront)
    ranked_score_according_prompt_sim = [ transfer_scores[i] for i in argsort_indices]
    # ideal list: from high perforamnce to low
    ideal_ranked_pref_list = [ transfer_scores[i] for i in np.array(transfer_scores).argsort()[::-1]]
    if has_done_check is False:
        print("argsort_indices", argsort_indices)
        print("ranked_score_according_prompt_sim", ranked_score_according_prompt_sim)
        print("ideal_ranked_pref_list",            ideal_ranked_pref_list)
        print("transfer_perf",transfer_scores)
        print("prompt_sim", similarity_scores)
        has_done_check = True

    dcg       = calculate_dcg(ranked_score_according_prompt_sim)
    ideal_dcg = calculate_dcg(ideal_ranked_pref_list)
    ndcg      = dcg/ideal_dcg
    return ndcg, has_done_check


def compute_regret_at_k_score(transfer_scores, similarity_scores, has_done_check, k=3):
    # get idx ; higher similarity, lower index
    argsort_indices = np.array(similarity_scores).argsort()[::-1]
    # similarity-ranked list; (higher similar infront)
    ranked_score_according_prompt_sim = [ transfer_scores[i] for i in argsort_indices][:k]

    # highest performance among the k top-ranked source tasks
    max_k_selected_perf = max(ranked_score_according_prompt_sim)
    # highest performance among all
    max_transfer_perf = max(transfer_scores)

    # ideal list: from high perforamnce to low
    ideal_ranked_pref_list = [ transfer_scores[i] for i in np.array(transfer_scores).argsort()[::-1]]
    if has_done_check is False:
        print("argsort_indices", argsort_indices)
        print("ranked_score_according_prompt_sim", ranked_score_according_prompt_sim)
        print("ideal_ranked_pref_list",            ideal_ranked_pref_list)
        print("max_k_selected_perf", max_k_selected_perf)
        print("max_transfer_perf", max_transfer_perf)
        print("transfer_perf",transfer_scores)
        print("prompt_sim", similarity_scores)
        has_done_check = True

    regret_at_k = ((max_transfer_perf - max_k_selected_perf) / max_transfer_perf) * 100
    return regret_at_k, has_done_check


def compute_top_k_score(transfer_scores, similarity_scores, has_done_check, k=3):
    # get idx ; higher similarity, lower index
    argsort_indices = np.array(similarity_scores).argsort()[::-1]
    # similarity-ranked list; (higher similar infront)
    ranked_score_according_prompt_sim = [ transfer_scores[i] for i in argsort_indices][:k]

    # mean performance among the k top-ranked source tasks
    max_k_selected_perf = np.array(ranked_score_according_prompt_sim).mean()
    # highest performance among all
    max_transfer_perf = np.array(transfer_scores).mean()

    # ideal list: from high perforamnce to low
    ideal_ranked_pref_list = [ transfer_scores[i] for i in np.array(transfer_scores).argsort()[::-1]]
    if has_done_check is False:
        print("argsort_indices", argsort_indices)
        print("ranked_score_according_prompt_sim", ranked_score_according_prompt_sim)
        print("ideal_ranked_pref_list",            ideal_ranked_pref_list)
        print("max_k_selected_perf", max_k_selected_perf)
        print("max_transfer_perf", max_transfer_perf)
        print("transfer_perf",transfer_scores)
        print("prompt_sim", similarity_scores)
        has_done_check = True

    regret_at_k = (max_k_selected_perf)
    return regret_at_k, 0, has_done_check




def valid_method_type(arg_method):
    """Custom argparse type for method"""
    if arg_method in ["ndcg", "regret_at_k", "top_k_performance"]:
        return arg_method
    else:
        # msg = "Given method arg not valid! Expected ndcg and regret_at_k"
        raise argparse.ArgumentTypeError("Expected in ndcg or regret_at_k")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', help='File to evaluate on')
    parser.add_argument('--opt_file', help='File to optimizer file')
    parser.add_argument('--tgt_task', default="rte")
    parser.add_argument('--relevance', default="score")
    parser.add_argument('--method', default="ndcg", type=valid_method_type)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--ranking_by_size', action="store_true")
    parser.add_argument('--ranking_by_random', action="store_true")
    parser.add_argument('--pred_dir', default="/data/users/pjlin/compacter/spot_eval/prompt_similarity")
    parser.add_argument('--target_dir', default="/data/users/pjlin/compacter/spot_eval/transfer_ranking")
    parser.add_argument('--output_dir', default="/data/users/pjlin/compacter/spot_eval/ndcg_results")
    args = parser.parse_args()
    
    ### modify here accordingly ###
    lr = "2"
    model = "t5-base"
    #  "qqp", "qnli", "boolq", "mrpc", "rte", "cb"
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
    tasks = ["superglue-wsc"]

    json_template = "eval_tgt-{TASK_NAME}_seed-{SEED}.json"
    print(json_template.format(TASK_NAME="rte",
                               SEED=32))
    
    checkpoint_steps = ["5000", "10000", "15000", "20000", "25000", "30000"]
    # seeds in each row [42,150,386]
    output_dir = "/data/users/pjlin/compacter/spot_eval/transfer_ranking"
    tgt_tasks = [args.tgt_task]
    src_tasks = ["mnli", "qqp", "qnli", 
                "superglue-record","cxc","squad",
                "drop","sst2","winogrande",
                "hellaswag", "superglue-multirc", "cosmosqa", 
                "race"]


    json_file = "trainer_state.json"
    ### modify here accordingly ###

    num_seed = len(SEEDS)
    # output_dir_tmp = "/data/users/pjlin/compacter/transfer_outputs/prompt_transfer_tokens_init_best_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    # output_dir_tmp = "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    # output_dir_tmp = "/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    output_dir_tmp = "/data/users/pjlin/compacter/spot_outputs/prompt_transfer_tokens_init_best_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"

    random.seed(5)
    ir_metric_means = list()
    ir_metric_std    = list()
    rdn_list = list() # check random generated rankings
    for ckpt_step in checkpoint_steps:
        has_done_check = False
        # n_task * n_seeds
        scores = np.zeros((len(tasks), len(SEEDS)))
        for i, task in enumerate(tasks):
            for j, seed in enumerate(SEEDS):
                # create predict file / tgt file
                pred_file = os.path.join(args.pred_dir, json_template.format(TASK_NAME=task, SEED=seed))
                tgt_file  = os.path.join(args.target_dir, json_template.format(TASK_NAME=task, SEED=seed))

                print(f"{pred_file}\n{tgt_file}\n\n")
                output_dir = output_dir_tmp.format(
                    LR=lr, MODEL=model, TASK_NAME=task, SEED=seed
                )
                output_dir = os.path.join(output_dir, json_file)

                if not os.path.isfile(output_dir):
                    logger.info(f"output dir not exist: {output_dir}")
                # read json                
                pred_json_data = read_json(pred_file)[ckpt_step]
                tgt_json_data = read_json(tgt_file)["best"]
                # assert the orderings of src task, creating same similarity
                assert tgt_json_data["src_tasks"] == pred_json_data["src_tasks"]
                # filter score and similarity with the items in `all_src_tasks`
                all_src_tasks = tgt_json_data["src_tasks"]
                keep_idx = [ i for i in range(len(all_src_tasks)) if all_src_tasks[i] in src_tasks ]
                transfer_perf = [ tgt_json_data["score"][i] for i in keep_idx ]
                
                if args.ranking_by_random:
                    # random.seed(seed)
                    prompt_sim = list(range(1,len(keep_idx)+1))
                    random.shuffle(prompt_sim)
                    rdn_list.append(prompt_sim) # check
                # sort by dataset size
                elif args.ranking_by_size:
                    prompt_sim = [ REFERENCE_SCORE_SIZE[pred_json_data["src_tasks"][i]] for i in keep_idx ]
                    print("prompt_sim by size", prompt_sim)
                else:
                    # prompt_sim = [ pred_json_data["score"][i] for i in keep_idx ]
                    prompt_sim = [ pred_json_data[args.relevance][i] for i in keep_idx ]

                # apply ir metrics
                if args.method == "ndcg":
                    score, has_done_check = compute_ndcg_score(transfer_scores=transfer_perf,
                                                               similarity_scores=prompt_sim,
                                                               has_done_check=has_done_check)

                elif args.method == "regret_at_k":
                    score, has_done_check = compute_regret_at_k_score(transfer_scores=transfer_perf,
                                                                      similarity_scores=prompt_sim,
                                                                      k=3,
                                                                      has_done_check=has_done_check)

                elif args.method == "top_k_performance" :
                    score, std, has_done_check = compute_top_k_score(transfer_scores=transfer_perf,
                                                                     similarity_scores=prompt_sim,
                                                                     k=2,
                                                                     has_done_check=has_done_check)



                has_done_check = has_done_check
                print(f"get score: {score}")
                scores[i, j] = score
        
        print(f"Steps: {ckpt_step}")    
        print(scores)
        print("mean", scores.mean(axis=-1))
        print("std", scores.std(axis=-1))
        print("mean", scores.mean())
        print("std", scores.std())
        ir_metric_means.append(scores.mean())
        ir_metric_std.append(scores.std())
    
    print("ir_metric_means", ir_metric_means)
    print("ir_metric_std", ir_metric_std)
    print("rdn_list", rdn_list)

    if args.method == "top_k_performance":
        print(scores.mean(axis=-1))
        print(scores.std(axis=-1))

if __name__ == "__main__":
    main()

