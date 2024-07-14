import functools
import logging
import numpy as np
import torch 
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
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
import json
import jsbeautifier
from dev.local_utils import (
    compute_matrix_norm, 
    compute_three_matrix_norms,
    compute_matrice_distance_norm
)


def write_to_json(dict_obj, output_file):
    """
    dict_obj: dictionary-like object
    output_file: file ending with .json
    """
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    with open(output_file, "w") as wf:
        wf.write(
            jsbeautifier.beautify(json.dumps(dict_obj, default=int), opts)+"\n")


def get_prompt(model_name_or_path):
    path = f"{model_name_or_path}/prefix_shared.bin"
    # decoder.prefix_emb | prefix_shared | encoder.prefix_emb
    
    device = torch.device("cpu")
    prompt_embs = torch.load(path, map_location=device)["encoder.prefix_emb"]
    return prompt_embs


def create_eval_obj():
    return


def compute_prompt_similarity(src_prompt, tgt_prompt, task_embedding_type=None):
    """Compute task embedding similarity

    features 
    """
    def _check_prompt_shape(prompt):
        if prompt.shape != (768, 100):
            return prompt.t()
        return prompt

    # src_prompt = src_prompt[:2,:2]
    # tgt_prompt = tgt_prompt[:2,:2]
    assert task_embedding_type in ["feature_mean", "flatten", "unigram", "bigram", "max_pairwise"]
    _src_prompt = _check_prompt_shape(src_prompt)
    _tgt_prompt = _check_prompt_shape(tgt_prompt)
    # print("src_prompt.shape 1", src_prompt.shape)
    # print("tgt_prompt.shape 1", tgt_prompt.shape)
    # print("src_prompt 1", src_prompt)
    # print("tgt_prompt 1", tgt_prompt)
    # _src_prompt = nn.functional.relu(_src_prompt)
    # _tgt_prompt = nn.functional.relu(_tgt_prompt)
    
    def compute_kernel_bias(vecs):
        """计算kernel和bias
        vecs.shape = [num_samples, embedding_size]，
        最后的变换：y = (x + bias).dot(kernel)
        """
        if vecs.shape != (100, 768):
            vecs = vecs.t()
        # (1, 768)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = torch.cov(vecs.T)
        # (768), (768, 768)
        u, s, vh = torch.svd(cov)
        # (768, 100)
        W = torch.mm(u, torch.diag(1 / torch.sqrt(s)))
        
        # print("vecs", vecs.shape)
        # print("mu", mu.shape)
        # print("u", u.shape)
        # print("s", s.shape)
        # print("W", W.shape)
        # cov = np.cov(vecs.T)
        # u, s, vh = np.linalg.svd(cov)
        # W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W, -mu

    def transform_and_normalize(vecs, kernel=None, bias=None):
        """应用变换，然后标准化
        """
        if vecs.shape != (100, 768):
            vecs = vecs.t()
        if not (kernel is None or bias is None):
            a = vecs + bias
            vecs = torch.mm(a,kernel)
        o = vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
        return o.t()
        
    ### whitening
    # _src_prompt_kernel, _src_prompt_bias = compute_kernel_bias(_src_prompt)
    # _src_prompt = transform_and_normalize(_src_prompt, _src_prompt_kernel, _src_prompt_bias)
    _src_prompt = transform_and_normalize(_src_prompt, None, None)

    # _tgt_prompt_kernel, _tgt_prompt_bias = compute_kernel_bias(_tgt_prompt)
    # _tgt_prompt = transform_and_normalize(_tgt_prompt, _tgt_prompt_kernel, _tgt_prompt_bias)
    _tgt_prompt = transform_and_normalize(_tgt_prompt, None, None)
    # print(_src_prompt.shape)
    # print(_tgt_prompt.shape)
    
    # (768)
    if task_embedding_type == "feature_mean":
        _src_prompt = _src_prompt.mean(axis=-1)
        _tgt_prompt = _tgt_prompt.mean(axis=-1)
        # _src_prompt = nn.functional.relu(_src_prompt)
        # _tgt_prompt = nn.functional.relu(_tgt_prompt)
        assert _tgt_prompt.shape[0] == 768
    # (768,100)
    elif task_embedding_type == "flatten":
        _src_prompt = _src_prompt.flatten()
        _tgt_prompt = _tgt_prompt.flatten()
    # (768,100)
    elif task_embedding_type == "unigram":
        _src_prompt = _src_prompt.mean(axis=0)
        _tgt_prompt = _tgt_prompt.mean(axis=0)
    # (768,50)
    elif task_embedding_type == "bigram":
        n_row = _src_prompt.shape[-1]
        # (50,768) -> (768, 50)
        _src_prompt = torch.stack([ _src_prompt[:, i:i+2].flatten() for i in range(0, n_row, 2)], dim=0).t()
        n_row = tgt_prompt.shape[-1]
        _tgt_prompt = torch.stack([ _tgt_prompt[:, i:i+2].flatten() for i in range(0, n_row, 2)], dim=0).t()
    # (768, 100*100), (768, 100*100)
    elif task_embedding_type == "max_pairwise":
        _src_prompt = _src_prompt.repeat_interleave(100, dim=-1)
        _tgt_prompt = _tgt_prompt.repeat(1, 100)

    # print("src_prompt.shape 2", src_prompt.shape)
    # print("tgt_prompt.shape 2", tgt_prompt.shape)
    # print("src_prompt.shape 2", src_prompt)
    # print("tgt_prompt.shape 2", tgt_prompt)

    # similarity between column vector
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_sim = cos(_src_prompt, _tgt_prompt)
    #  print("cos_sim", cos_sim)
    if task_embedding_type == "max_pairwise":
        cos_sim = cos_sim.reshape(100, -1)
        cos_sim = cos_sim.max(dim=-1)[0]
        
    # print("cos_sim", cos_sim)
    # print("mean cos sim. (dim=0)", cos_sim.shape)
    # print("mean cos sim. (dim=0)", cos_sim.mean())
    # print("n_sim_tokens (dim=0)", n_sim_tokens)
    return cos_sim.mean().item(), cos_sim, (_src_prompt, _tgt_prompt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', help='File to evaluate on')
    parser.add_argument('--opt_file', help='File to optimizer file')
    parser.add_argument('--tgt_task', default="rte")
    parser.add_argument('--task_embedding_type', default="feature_mean")
    # parser.add_argument('--output_dir', default="/data/users/pjlin/compacter/spot_eval/prompt_similarity_relu_task_emb")
    # parser.add_argument('--output_dir', default="/data/users/pjlin/compacter/spot_eval/prompt_similarity_whitening")
    parser.add_argument('--output_dir', default="/data/users/pjlin/compacter/spot_eval/prompt_similarity_max_pairwise_normalization")
    # parser.add_argument('--output_dir', default="/data/users/pjlin/compacter/spot_eval/prompt_similarity_avg_unigram")
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
    # checkpoint_steps = ["10000"]
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

    seeds = ["42", "150", "386"]
    seeds = ["28", "52" , "112"]
    lr   = "2"
    model = "t5-base"
    json_file = "trainer_state.json" 
    output_dir_tmp = "/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    output_dir_tmp = "/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"
    output_dir_tmp = "/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-{LR}/{MODEL}/{TASK_NAME}/{SEED}"

    # sim = [[0.51724952 0.34110346 0.52830184 0.36148578],
    #        [0.63061506 0.45075765 0.65610671 0.3541325 ],
    #        [0.35222611 0.32605103 0.3604984  0.33677474]]

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
    
    # norms of high resource prompt weights
    l1_norm_lst, l2_norm_lst, frobenius_norm_lst = list(), list(), list()
    
    ### Run ### 
    for seed in seeds:
        logger.info(f"Running seed: {seed}")
        ckpt2matrix = dict()
        ckpt2rank   = dict()
        for step in checkpoint_steps:
            score_matrix    = np.zeros((len(src_tasks), len(tgt_tasks)))
            l1_norm_matrix  = np.zeros((len(src_tasks), len(tgt_tasks)))
            l2_norm_matrix  = np.zeros((len(src_tasks), len(tgt_tasks)))
            fro_norm_matrix = np.zeros((len(src_tasks), len(tgt_tasks)))

            score_rank = np.zeros(score_matrix.shape)
            logger.info(f"rank init \n{score_rank}")
            for i, src_t in enumerate(src_tasks):
                # loading 
                src_path = checkpoints[i] # src_path = os.path.join(src_tmp, src_t, seed, f"checkpoint-{step}")
                src_prompt = get_prompt(src_path) # (768, 100)
                if len(l1_norm_lst) != len(src_tasks):
                    l1_norm, l2_norm, frobenius_norm = compute_three_matrix_norms(mtx=src_prompt, axis="token_norm")
                    l1_norm_lst.append(l1_norm)
                    l2_norm_lst.append(l2_norm)
                    frobenius_norm_lst.append(frobenius_norm)


                similarity_vector_collection = list()
                cnt_num_larger_similarities  = list()
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
                    tgt_prompt = get_prompt(tgt_path)
                    cosine_score, cosine_vector, (_, _) = compute_prompt_similarity(src_prompt, tgt_prompt, task_embedding_type=args.task_embedding_type)
                    
                    n_sim_tokens = (cosine_vector > 0.3).sum()
                    similarity_vector_collection.append(cosine_vector.tolist())
                    cnt_num_larger_similarities.append(n_sim_tokens)
                    
                    l1_norm_pair = compute_matrice_distance_norm(src_prompt, tgt_prompt, order="1")
                    l2_norm_pair = compute_matrice_distance_norm(src_prompt, tgt_prompt, order="2")
                    fro_norm_pair = compute_matrice_distance_norm(src_prompt, tgt_prompt, order="fro")

                    # print("prompt shape", tgt_prompt.shape)
                    # print("cos", cosine_score)
                    # assert 3==2
                    score_matrix[i,j]    = cosine_score
                    l1_norm_matrix[i,j]  = l1_norm_pair
                    l2_norm_matrix[i,j]  = l2_norm_pair
                    fro_norm_matrix[i,j] = fro_norm_pair
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
                                      "ranking":score_rank.flatten().tolist(),
                                      "similarity_list": similarity_vector_collection,
                                      "num_sim_greater_than_point_three": cnt_num_larger_similarities,
                                      "src_l1_norm": l1_norm_lst,
                                      "src_l2_norm": l2_norm_lst,
                                      "src_frobenius_norm": frobenius_norm_lst,
                                      "l1_norm": l1_norm_matrix.flatten().tolist(),
                                      "l2_norm": l2_norm_matrix.flatten().tolist(),
                                      "frobenius_norm": fro_norm_matrix.flatten().tolist()
                                      }
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
    
    return 

if __name__ == "__main__":
    main()
