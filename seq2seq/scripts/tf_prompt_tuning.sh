#!/usr/bin/env bash

# This scripts trains prompt tuning with initialization from pretrained prompt weight.
# Please specify the checkpoints to CKPTS.

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

##### data & training setup stuff #####
GLUE_TASKS=(boolq cola stsb superglue-wic cr mrpc rte superglue-wsc superglue-copa cb) # tgt task

# t5-base, t5-large or t5-base-lm-adapt
MODEL=t5-base
# change TUNING accordingly
CONFIG_PREFIX=configs/prompt_tuning_tokens_config
# Fix one source tasks. Varying in steps and seeds
SOURCE_TASKS=(mnli qqp qnli superglue-record cxc squad drop sst2 winogrande hellaswag cosmosqa race superglue-multirc)

STEPS=(best)
SEEDS=(28 52 112) # src-to-tgt
LR=2
LR_SCHEDULER_TYPE=linear
##### data & training setup stuff #####

# define lenght for each task
declare -A LENGTHS=(
    ["mnli"]=128             ["qnli"]=128              ["qqp"]=128
    ["superglue-record"]=512 ["cxc"]=128               ["squad"]=512
    ["drop"]=512             ["sst2"]=128              ["winogrande"]=128
    ["hellaswag"]=512        ["superglue-multirc"]=512 ["cosmosqa"]=512
    ["race"]=512

    ["boolq"]=128            ["cola"]=128              ["stsb"]=128
    ["superglue-wic"]=128    ["cr"]=128                ["mrpc"]=128
    ["rte"]=128              ["superglue-wsc"]=128     ["superglue-copa"]=128
    ["cb"]=128
    )

# Path to pretrained prompt weight
declare -A CKPTS=(
    ["mnli"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/mnli/42/checkpoint-25500
    ["qqp"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/qqp/386/checkpoint-19500
    ["qnli"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/qnli/42/checkpoint-6000
    ["superglue-record"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/superglue-record/42/checkpoint-15000
    ["cxc"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/cxc/42/checkpoint-29000
    ["squad"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/squad/386/checkpoint-20000
    ["drop"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/drop/42/checkpoint-29500
    ["sst2"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/sst2/42/checkpoint-5500              
    ["winogrande"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/winogrande/386/checkpoint-30000
    ["hellaswag"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/hellaswag/42/checkpoint-23500
    ["cosmosqa"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/cosmosqa/42/checkpoint-29000
    ["race"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/race/386/checkpoint-26000
	["superglue-multirc"]=/data/users/pjlin/compacter/prompt_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/t5-base/superglue-multirc/386/checkpoint-28500
)


for STEP in ${STEPS[*]}
do
	for SEED in ${SEEDS[*]}
	do
		for SOURCE_TASK in ${SOURCE_TASKS[*]}
		do
			# source checkpoint
			# ckpt_dir=/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-5e-1/${MODEL}/${SOURCE_TASK}/150/checkpoint-${STEP} # fix to 150
			ckpt_dir="${CKPTS[${SOURCE_TASK}]}"
			echo "source task: ${SOURCE_TASK}"
			echo "CKPT_DIR: ${ckpt_dir}"
			if [ ! -d "$ckpt_dir" ]
			then
				echo "Checkpoint $ckpt_dir does not exist"
			fi

			# Run
		    for TASK_NAME in ${GLUE_TASKS[*]}
		    do
		        # organized as follow: OUTPUT_DIR_PREFIX/TUNING/MODEL/TASK/SEED
		        # output_dir="/data/users/pjlin/compacter/spot_outputs/prompt_transfer_tokens_init_best_adafactor_lr-${LR}/${MODEL}/${SOURCE_TASK}_${TASK_NAME}/${SEED}"
		        output_dir="/data/users/pjlin/compacter/spot_test_outputs/prompt_transfer_tokens_init_best_adafactor_lr-${LR}/${MODEL}/${SOURCE_TASK}_${TASK_NAME}/${SEED}"
		        rm -r $output_dir
		        mkdir -p $output_dir

		        L="${LENGTHS[${TASK_NAME}]}"

		        echo "output_dir: ${output_dir}"
		        CONFIG=$CONFIG_PREFIX/$MODEL/prompt_tuning_tokens_init_${TASK_NAME}.json
		        python run_seq2seq.py \
		        --config=$CONFIG \
		        --model_name_or_path=$MODEL \
		        --prefix_shared=${ckpt_dir}/prefix_shared.bin \
		        --src_task_name=$SOURCE_TASK \
		        --learning_rate=$LR \
		        --compute_time \
		        --compute_memory \
		        --max_source_length=$L \
		        --lr_scheduler_type=$LR_SCHEDULER_TYPE \
		        --seed=$SEED \
		        --data_seed=$SEED \
		        --clean_checkpoint \
		        --output_dir=$output_dir 2>&1 | tee ${output_dir}/train.log
		    done
		done
	done
done

