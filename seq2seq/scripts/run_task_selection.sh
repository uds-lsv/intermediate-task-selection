#!/usr/bin/env bash

# This scripts run task selection methods 

###### Job setup stuff ######
# run setup
source /nethome/pjlin/pythonProjects/htcondor_test_master/scripts/setup.sh

# run misc. stuff
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
echo $HOSTNAME
which python
python -m pip list
###### Job setup stuff ######


##### Project setup stuff #####
export HOME=/nethome/pjlin
cd $HOME
source ~/.bashrc
conda activate compacter
cd /nethome/pjlin/pythonProjects/compacter/seq2seq/
##### Project setup stuff #####


##### data & training setup stuff #####
# boolq mrpc rte cb
SEEDS=(28 52 112)
LEARNING_RATES=(2)
GLUE_TASKS=(boolq mrpc rte cb)
SOURCE_TASKS=(mnli qqp qnli)

# t5-base, t5-large or t5-base-lm-adapt
MODEL=t5-base
# change TUNING accordingly
CONFIG_PREFIX=configs/prompt_tuning_tokens_config
CKPT=t5-base
LR_SCHEDULER_TYPE=linear
##### data & training setup stuff #####


# taskemb (single-task)
for SEED in ${SEEDS[*]}
do
    for TASK_NAME in ${GLUE_TASKS[*]}
    do
        # organized as follow: OUTPUT_DIR_PREFIX/TUNING/MODEL/TASK/SEED
        # output_dir="/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-${LR}/${MODEL}/${TASK_NAME}/${SEED}_run-2"
        # output_dir="/data/users/pjlin/compacter/task_selection/vocab_similarity_${TOK_LEVEL}_${NGRAM}"
        output_dir="/data/users/pjlin/compacter/task_selection/taskemb_trained_prompt_tuning/${MODEL}/${TASK_NAME}/${SEED}" # taskem

        rm -fr $output_dir
        mkdir -p $output_dir
        echo "output_dir: ${output_dir}"

        CONFIG=$CONFIG_PREFIX/$MODEL/prompt_tuning_tokens_init_${TASK_NAME}.json
        python run_embeddings.py \
            --config=$CONFIG \
            --model_name_or_path=$MODEL \
            --seed=$SEED \
            --data_seed=$SEED \
            --do_taskemb \
            --output_dir=$output_dir 2>&1 | tee ${output_dir}/train.log
    done
done


# taskemb (cross-task)
# for SEED in ${SEEDS[*]}
# do
#     for SOURCE_TASK in ${SOURCE_TASKS[*]}
#     do
#         for TASK_NAME in ${GLUE_TASKS[*]}
#         do
#             # source checkpoint
#             ckpt_dir=/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/${MODEL}/${SOURCE_TASK}/150/checkpoint-best # fix to 150
#             echo "CKPT_DIR: ${ckpt_dir}"
#             if [ ! -d "$ckpt_dir" ]
#             then
#                 echo "Checkpoint $ckpt_dir does not exist"
#             fi

#             # no train
#             # output_dir="/data/users/pjlin/compacter/task_selection/taskemb_prompt_tuning/${MODEL}/${SOURCE_TASK}_${TASK_NAME}/${SEED}" # taskem
#             output_dir="/data/users/pjlin/compacter/task_selection/taskemb_trained_prompt_tuning/${MODEL}/${SOURCE_TASK}_${TASK_NAME}/${SEED}" # taskem
            
#             rm -fr $output_dir
#             mkdir -p $output_dir
#             echo "output_dir: ${output_dir}"

#             CONFIG=$CONFIG_PREFIX/$MODEL/prompt_tuning_tokens_init_${TASK_NAME}.json
#             python run_embeddings.py \
#                 --config=$CONFIG \
#                 --model_name_or_path=$ckpt_dir \
#                 --src_task_name=$SOURCE_TASK \
#                 --seed=$SEED \
#                 --data_seed=$SEED \
#                 --do_taskemb \
#                 --output_dir=$output_dir 2>&1 | tee ${output_dir}/train.log
#         done
#     done
# done


