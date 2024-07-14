#!/usr/bin/env bash

# This scripts trains prompt tuning with initialization from language model's vocabulary method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3.
# python run_seq2seq.py  configs/prompt_tuning_tokens_init.json

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
# variable `SOURCE_TASK` has to assign to `--src_task_name`
#   total experiements is (src_t * ckpt_step * seeds * tgt_t)
export WANDB_PROJECT=prompt-transfer
export WANDB_WATCH=all
export WANDB_SILENT=true
export HOME=/nethome/pjlin
cd $HOME
source ~/.bashrc
conda activate compacter
cd /nethome/pjlin/pythonProjects/compacter/seq2seq/
##### Project setup stuff #####

# export CUDA_VISIBLE_DEVICES=3

##### data & training setup stuff #####
# boolq mrpc rte cb
GLUE_TASKS=(stsb superglue-copa superglue-wic superglue-wsc cr cola)
GLUE_TASKS=(stsb)
GLUE_TASKS=(mnli qqp qnli sst2 superglue-multirc)

# t5-base, t5-large or t5-base-lm-adapt
MODEL=t5-base
# change TUNING accordingly
CONFIG_PREFIX=configs/prompt_tuning_tokens_config
# Fix one source tasks. Varying in steps and seeds

SOURCE_TASKS=(mnli qqp qnli sst2 superglue-multirc)


STEPS=(best)
SEEDS=(28 52 112) # src tasks
SEEDS=(42) # src tasks
LR=5e-1
LR_SCHEDULER_TYPE=linear
TO_LRS=()
##### data & training setup stuff #####


for STEP in ${STEPS[*]}
do
    for SEED in ${SEEDS[*]}
    do
        for SOURCE_TASK in ${SOURCE_TASKS[*]}
        do
            # source checkpoint
            #ckpt_dir=/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/${MODEL}/${SOURCE_TASK}/150/checkpoint-${STEP} # fix to 150

            ckpt_dir=/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/${SOURCE_TASK}/42/checkpoint-30000/
            #ckpt_dir=/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/${SOURCE_TASK}/42/checkpoint-27500/
            ckpt_dir=/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/race/42/checkpoint-28500
            ckpt_dir=/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1_addQ/t5-base/winogrande/42/checkpoint-30000
            #ckpt_dir=/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/sst2/42/checkpoint-14500
            ckpt_dir=/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-2/t5-base/cb/42/checkpoint-1500
            
            echo "CKPT_DIR: ${ckpt_dir}"
            if [ ! -d "$ckpt_dir" ]
            then
                echo "Checkpoint $ckpt_dir does not exist"
            fi

            # Run
            for TASK_NAME in ${GLUE_TASKS[*]}
            do
                # organized as follow: OUTPUT_DIR_PREFIX/TUNING/MODEL/TASK/SEED
                # output_dir="/data/users/pjlin/compacter/transfer_outputs/prompt_transfer_test"
                output_dir="/data/users/pjlin/compacter/test_outputs/prompt_transfer_tokens_init_best_adafactor_lr-${LR}/${MODEL}/${SOURCE_TASK}_${TASK_NAME}/${SEED}"

                rm -fr $output_dir
                mkdir -p $output_dir

                echo "output_dir: ${output_dir}"
                CONFIG=$CONFIG_PREFIX/$MODEL/prompt_tuning_tokens_init_${TASK_NAME}.json
                echo $CONFIG
                python run_eval.py \
                --config=$CONFIG \
                --model_name_or_path=t5-base \
                --src_task_name=$SOURCE_TASK \
                --learning_rate=$LR \
                --compute_time \
                --max_steps=100 \
                --eval_steps=50 \
                --max_source_length=128 \
                --compute_memory \
                --lr_scheduler_type=$LR_SCHEDULER_TYPE \
                --seed=$SEED \
                --data_seed=$SEED \
                --output_dir=$output_dir 2>&1 | tee ${output_dir}/train.log
            done
        done
    done
done
