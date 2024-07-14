# This scripts trains full finetuning method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. 
# python run_seq2seq.py  configs/baseline.json

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
SEEDS=(42 150 386)

# 3e-1 1e-2
LEARNING_RATES=(3e-1)
GLUE_TASKS=(rte)

# define lenght for each task
declare -A LENGTHS=(
    ["mnli"]=128             ["qnli"]=128              ["qnli"]=128
    ["superglue-record"]=512 ["cxc"]=128               ["squad"]=512
    ["drop"]=512             ["sst2"]=128              ["winogrande"]=128
    ["hellaswag"]=512        ["superglue-multirc"]=512 ["cosmosqa"]=512
    ["race"]=512             ["qqp"]=128

    ["boolq"]=128            ["cola"]=128              ["stsb"]=128
    ["superglue-wic"]=128    ["cr"]=128                ["mrpc"]=128
    ["rte"]=128              ["superglue-wsc"]=128     ["superglue-copa"]=128
    ["cb"]=128
    )

# cache: /nethome/pjlin/.cache/huggingface/datasets/super_glue/record
# t5-base, t5-large or t5-base-lm-adapt
MODEL=t5-base
# change TUNING accordingly
TUNING=baseline
CONFIG_PREFIX=configs/${TUNING}_config
CKPT=t5-base
LR_SCHEDULER_TYPE=linear
##### data & training setup stuff #####

## new
for SEED in ${SEEDS[*]}
do
    for LR in ${LEARNING_RATES[*]}
    do
        for TASK_NAME in ${GLUE_TASKS[*]}
        do
            # organized as follow: OUTPUT_DIR_PREFIX/TUNING/MODEL/TASK/SEED
            output_dir="/data/users/pjlin/compacter/outputs/prompt_tuning_tokens_init_30k_adafactor_lr-${LR}/${MODEL}/${TASK_NAME}/${SEED}"
            
            rm -fr $output_dir
            mkdir -p $output_dir
            echo "output_dir: ${output_dir}"

            L="${LENGTHS[${TASK_NAME}]}"
            echo "context length: ${L}"
            
            CONFIG=$CONFIG_PREFIX/$MODEL/${TUNING}_${TASK_NAME}.json
            # Run
            python run_seq2seq.py \
            --config=$CONFIG \
            --model_name_or_path=$MODEL \
            --learning_rate=$LR \
            --compute_time \
            --compute_memory \
            --max_source_length=$L \
            --clean_checkpoint \
            --lr_scheduler_type=$LR_SCHEDULER_TYPE \
            --seed=$SEED \
            --data_seed=$SEED \
            --output_dir=$output_dir 2>&1 | tee ${output_dir}/train.log
        done
    done
done


# t5-base or t5-large
MODEL=t5-base
# change TUNING accordingly

CONFIG_PREFIX=configs/${TUNING}_config

for TASK_NAME in ${GLUE_TASKS[*]}
do
    # organized as follow: OUTPUT_DIR_PREFIX/TUNING/MODEL/TASK/SEED
    #output_dir="/data/users/pjlin/compacter/outputs/${TUNING}/${MODEL}/${TASK_NAME}/42"
    output_dir="/data/users/pjlin/compacter/outputs/test"
	rm -r $output_dir
    mkdir -p $output_dir
    echo "output_dir: ${output_dir}"
    
    CONFIG=$CONFIG_PREFIX/$MODEL/baseline_${TASK_NAME}.json
    echo "config: ${CONFIG}"
    # configs/prompt_tuning_tokens_init.json
    python run_seq2seq.py $CONFIG 2>&1 | tee ${output_dir}/train.log
done
