#!/usr/bin/env bash

# This scripts calcualte the prompt similarity (task embeddings).

GLUE_TASKS=(boolq cola stsb superglue-wic cr mrpc rte superglue-wsc superglue-copa cb) 

METHODS=(feature_mean flatten unigram bigram max_pairwise)
declare -A OUTPUT_DIRS=(
    ["feature_mean"]=/data/users/pjlin/compacter/spot_eval/prompt_similarity
    ["flatten"]=/data/users/pjlin/compacter/spot_eval/prompt_similarity_flat
    ["unigram"]=/data/users/pjlin/compacter/spot_eval/prompt_similarity_avg_unigram
    ["bigram"]=/data/users/pjlin/compacter/spot_eval/prompt_similarity_avg_bigram
    ["max_pairwise"]=/data/users/pjlin/compacter/spot_eval/prompt_similarity_max_pairwise
)

# RUN
echo "Running prompt similarity"
for m in ${METHODS[*]}
do
    for tgt_task in ${GLUE_TASKS[*]}
    do
        # source checkpoint
        # ckpt_dir=/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/${MODEL}/${SOURCE_TASK}/150/checkpoint-${STEP} # fix to 150
        output_dir="${OUTPUT_DIRS[${m}]}"
        echo $m
        echo $output_dir

        python prompt_sim_np.py \
            --tgt_task=$tgt_task \
            --task_embedding_type=$m \
            --output_dir=$output_dir 2>&1 | tee prompt_similarity_best_cross_steps.log
    done
done








