# This scripts load the dataset from Huggingface 

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

GLUE_TASKS=(squad drop sst2 winogrande hellaswag superglue-multirc cosmosqa race superglue-record)
#GLUE_TASKS=(superglue-multirc)
# t5-base, t5-large or t5-base-lm-adapt
MODEL=t5-base
# change TUNING accordingly
CONFIG_PREFIX=configs/prompt_tuning_tokens_config
SEED=42

LEARNING_RATES=(5e-1)
for LR in ${LEARNING_RATES[*]}
do
    for TASK_NAME in ${GLUE_TASKS[*]}
    do
        # output_dir="/data/users/pjlin/compacter/test"
        output_dir="/data/users/pjlin/compacter/test_outputs/prompt_tuning_tokens_init_30k_adafactor_lr-${LR}_seq-512/${MODEL}/${TASK_NAME}/${SEED}"
        rm -r $output_dir
        mkdir -p $output_dir
        echo "output_dir: ${output_dir}"
        
        CONFIG=$CONFIG_PREFIX/$MODEL/prompt_tuning_tokens_init_${TASK_NAME}.json
        # python run_seq2seq.py $CONFIG $CKPT $output_dir 2>&1 | tee ${output_dir}/train.log
        python loading_datasets.py \
        --config=$CONFIG \
        --model_name_or_path=$MODEL \
        --learning_rate=$LR \
        --compute_time \
        --max_source_length=512 \
        --compute_memory \
        --lr_scheduler_type=$LR_SCHEDULER_TYPE \
        --seed=$SEED \
        --data_seed=$SEED \
        --output_dir=$output_dir 2>&1 | tee ${output_dir}/train.log
    done
done

