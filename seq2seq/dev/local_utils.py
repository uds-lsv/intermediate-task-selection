import sys
import json
import jsbeautifier
import logging
import numpy as np
import torch
from torch.linalg import matrix_norm


logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def compute_matrix_norm(mtx, order="fro", axis="token_norm"):
    """Compute matrix norm
    """
    # column-sum norm; select largest column sum
    mtx_ = mtx
    # norm of each token
    if axis == "token_norm" and mtx.shape[0] != 768:
        mtx_ = mtx.t()
    # norm of feature
    elif axis == "feature_norm" and mtx.shape[0] != 100:
        mtx_ = mtx.t()
    return matrix_norm(mtx, ord=order).item()

def compute_three_matrix_norms(mtx, axis="token_norm"):
    """Compute l1, l2, and frobenius matrix norm
    """
    l1 = compute_matrix_norm(mtx, order=1, axis=axis)
    l2 = compute_matrix_norm(mtx, order=2, axis=axis)
    fro = compute_matrix_norm(mtx, order="fro", axis=axis)
    return l1, l2, fro



def compute_matrice_distance_norm(mtx_a, mtx_b, order="fro"):
    """Kullback-Leibler divergence for probability vectors

    D(A,B) = sum A_ij * (log(A/B) + log(sum(B)) - log(sum(A)))

    """
    order = str(order)
    mtx_diff = mtx_a - mtx_b
    if order == "fro":
        norm = (torch.square(torch.abs(mtx_diff)).sum()) ** 0.5
    elif order == "1":
        norm = torch.abs(mtx_diff).sum()
    elif order == "2":
        norm = (torch.square(mtx_diff)).sum() ** 0.5
    return norm.item()


def kl_divergence(mtx_a, mtx_b):
    """Kullback-Leibler divergence for probability vectors

    D(A,B) = sum A_ij * (log(A/B) + log(sum(B)) - log(sum(A)))

    """
    S = torch.log(mtx_a/mtx_b) + torch.log(mtx_b) - torch.log(mtx_a)
    return (A*S).sum()

def calculate_dcg(scores):
    dcg = 0
    for i, score in enumerate(scores):
        dcg += (2**score - 1) / np.log2(i + 2)  # 添加 2^相關性 - 1 / log2(位置 + 2)
    return dcg

def read_json(filepath):
   f = open(filepath,)
   return json.load(f)

def get_best_score_from_log(fname, metric_name="best_metric", debugging=False):
    logger.info(f"Get score from log: {fname}")
    with open(fname, "r") as f:
        data = json.load(f)
    return data[metric_name]


def write_to_json(dict_obj, output_file):
    """
    dict_obj: dictionary-like object
    output_file: file ending with .json
    """
    wf = open(output_file, "w")
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    wf.write(
        jsbeautifier.beautify(json.dumps(dict_obj), opts)+"\n")