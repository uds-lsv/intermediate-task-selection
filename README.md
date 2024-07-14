# Exploring Task Selection for Intermediate-Task Transfer Learning

| [Installation](#installation) | [Datasets](#datasets) | [Prompt Transfer](#prompt-transfer) | [Task Selection](#task-selection) |

This repository contains the implementation of "Exploring the Effectiveness and Consistency of Task Selection in Intermediate-Task Transfer Learning: A Systematic Study" with prompt tuning, and prompt transfer.

<!--
In this repository, we explore the effectiveness of task selection approaches based on prompt transfer. We first train soft prompts with a frozen T5 model, then continually train the prompt weight on resource-constrained tasks. 
We investigate task selection methods including data size, text embedding, prompt-based task embedding, and further extend other direction to contruct the task embedding, such as max-pairwise similarity (MAX), flatten, unigram task embedding.

Our code is run by submitting a job via HTCondor. Please specify the following scripts in your submit file. Otherwise, you can comment out the lines for job submission setup in those bash scripts. For instructions on HTCondor, please read the [documents](https://repos.lsv.uni-saarland.de/mmosbach/htcondor-test/-/tree/master).
-->


## Installation

### Python Version

* Python >= 3.8

### Environment

Create an environment from file and activate the environment.

```
conda create -n intermediate-task-selection
conda activate intermediate-task-selection
```

Install denpendencies first.

```bash
. install_dependencies.sh
```

We suggest to install editable mode. This works for most of case. 

```bash
pip install -e .
```


## Datasets

We train 23 tasks across NLI, paraphrase detection, semantic similarity, question answering, reading comprehension, grammatical acceptability, word sense disambiguation, sentiment analysis, and coreference resolution. 

We split these datasets into those with less than 1K training samples as target tasks and the those with more than 1K samples as source tasks.

In total, we select 13 source tasks and 10 target tasks. To run the following tasks, please use the provided names below:

### Source Tasks

Here, source tasks paired with their respective evaluation metrics are the dataset with richer annotation numbers. 
We train 13 source tasks with prompt tuning, then initialize these pre-trained prompt weights for continual training on target tasks. We apply a learning rate of 5e-1 for all source tasks.


| Dataset            | Metrics                        |
| ------------------ | ------------------------------ |
| mnli               | accuracy                       |
| mnli_mismatched    | accuracy                       |
| mnli_matched       | accuracy                       |
| qqp                | accuracy, f1                   |
| qnli               | accuracy                       |
| superglue-record   | f1, em                         |
| cxc                | pearson, spearmanr             |
| squad              | f1, em                         |
| drop               | f1, em                         |
| sst2               | accuracy                       |
| winogrande         | accuracy                       |
| hellaswag          | accuracy                       |
| superglue-multirc  | f1, em                         |
| cosmosqa           | accuracy                       |
| race               | accuracy                       |


### Target Tasks

We train on 10 target task as baseline and transfer 13 source prompt to each target task. We apply learning rate of 2.


| Dataset            | Metrics                        |
| ------------------ | ------------------------------ |
| boolq              | accuracy                       |
| cola               | matthews_correlation           |
| stsb               | pearson, spearmanr             |
| superglue-wic      | accuracy                       |
| cr                 | accuracy                       |
| mrpc               | accuracy, f1                   |
| rte                | accuracy                       |
| superglue-wsc      | accuracy                       |
| superglue-copa     | accuracy                       |
| cb                 | f1_multiclass, accuracy        |



## Prompt Transfer

This section performs intermediate task transfer with prompt tuning. This involves:

1. Prompt tuning that initialized from the sampled embedding's token.
2. Prompt transfer that initialized from pretrained prompt trained on the source task.

To reproduce the results of Table 5.2 (Effect of prompt transfer), you need to execute both scripts for prompt tuning and the prompt transfer one.

We applied the same configuration for both prompt tuning and prompt transfer.

You can find our example scripts under `seq2seq/scripts`.
These scripts demonstrate prompt tuning, prompt transfer, and task selection using the configuration files located in `seq2seq/configs`.
To execute the models, please first do:

```bash
cd intermediate-task-selection/seq2seq
```


### Configuration File and Arguments

When training the model with prompt tuning, all code for training requires a configuration file defined in `configs` folder. Our implementation manages files according to the fine-tuning methods and model type, e.g.  `configs/prompt_tuning_tokens_config/t5-base`. Feel free to create your own directory.

Note that we offer partial arguments in our main Python script (`run_seq2seq.py`) to enable flexible configuration of hyperparameters. 
These partial arguments facilitate sweeping over different testing values, overriding the arguments specified in the configuration files.


### Run Prompt Tuning

To perform prompt tuning on both the source and target tasks for a single task, execute the following command. This script trains prompt tuning with initialization from the language model's vocabulary method. 

When dealing with target tasks, we set the learning rate to 2, while for the source tasks, a learning rate of 5e-1 is employed.

To run prompt tuning, please run the command:

```bash
. script/prompt_tuning_tokens_init.sh
```

To get the average performance, please run:


```bash
python dev/get_prompt_scores.py
```


### Run Prompt Transfer

The commands train prompt tuning with initialization from pretrained prompt weights. Please specify your prompt checkpoints to `CKPTS`.

```bash
. script/tf_prompt_tuning.sh
```

To get the relative performance, please run:

```bash
python dev/get_transfer_scores.py
```


### Creating Ranking Files

After training all models on target tasks, you can create a ranking of empirical prompt transfer as ground-truth reference for evaluating the prediction of task embedding. For evaluating all 13 source tasks transferring to RTE, please do:

```bash
python dev/get_transfer_ranking.py \
	--tgt_task=rte \
	--output_dir=PATH_TO_YOUR_DIR/spot_eval/transfer_ranking
```

We save the result file in the `--output_dir` and require it for evaluation ranking. Each file is named as `eval_tgt-TASKNAME_seed-VALUE`, which contains prompt tuning's and prompt transfer's performances and ranking of intermediate tasks sorted by prompt transfer.

You can also run the script once you have done all the training jobs.

```bash
. scripts/run_transfer_ranking.sh
```



## Task Selection

In order to evaluate the transferability of a source task to a given target task, one would need to run prompt tuning on all tasks.
We provides code for estimating the transferability via vocab similarity and task embedding upon prompt weight.

- vocab similarity

vocab similarity estimates the overlapping of two vocaublaries.
Please run:

```bash
. scripts/run_vocab_sim.sh
```

- task embeddings

With our training scripts, we save the prompt weight along in the file prefix_shared.bin.
Through all task embedding experiments, the weights are calculated for the task embeddings.

Except for prompt similarity (`feature_mean`), we provide additional constructions for task embeddings.
The following values are supported for `--task_embedding_type` argument: `feature_mean`, `flatten`, `unigram`, `bigram`, `max_pairwise`.


```bash
. scripts/get_prompt_similarity.sh
```

We save the predicted ranking file `eval_tgt-TASKNAME_seed-VALUE.json` in the `--output_dir`.


### Evaluation on Task Selection

We evaluate using metrics such as `ndcg` and `regret_at_k`, which are supported by the `--method` argument.
To assess the task selection methods (random, size-based, text embedding-based, or task embedding-based), both a prediction file and a reference file are necessary.

```bash
python dev/get_ndcg.py \
	--pred_dir=PATH_TO_YOUR_DIR \
	--target_dir=PATH_TO_YOUR_DIR \
	--output_dir=PATH_TO_YOUR_DIR \
	--method=ndcg
```

You can replace `ndcg` with `regret_at_k` and `top_k_performance`.

For evaluation prediction of data size and random approaches, you can directly pass boolean arguments as follows:

```bash
python dev/get_ndcg.py 	--ranking_by_random
```

For data size method, please do:  

```bash
python dev/get_ndcg.py 	--ranking_by_size
```

## Fine-tuning methods and adding tasks

This repo is developed based on [COMPACTER](https://github.com/rabeehk/compacter) and contains the implementation of recent parameter-efficient fine-tuning methods. For full fine-tuning, please run:


```bash
. scripts/baseline.sh
```  

Other parameter fine-tuning methods can be found in the scripts folder (Adapter, AdapterDrop, Low-Rank, BitFit, Compacter, Compacter++, PHM-Adapters, Intrinsic-SAID). Please check `scripts` for detail

If you wish to add a new task, you will need to create a new dataset class in `/data/{tasks,postprocessors}.py `and its corresponding configuration file. For example, when running a task, a configuration file such as `prompt_tuning_tokens_config/t5-base/prompt_tuning_tokens_init_boolq.json` is required.


<!--## Bibliography

If you find this repo useful, please cite our work:

```bash
@inproceedings{lin2024taskselection,
  title={Exploring Task Selection For Intermediate Task Transfer},
  author={Pin-Jie Lin},
  year={2024}
}
```
-->

## Contact Information

For help or issues using our code, please submit a issue or contact to pjlin@lsv.uni-saarland.de
