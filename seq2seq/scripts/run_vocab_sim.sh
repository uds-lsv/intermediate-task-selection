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
SEEDS=(28)
LEARNING_RATES=(2)
GLUE_TASKS=(boolq mrpc rte cb)

TOKENIZATIONS=(word) # subword
NGRAMS=(unigram) # bigram

# t5-base, t5-large or t5-base-lm-adapt
MODEL=t5-base
# change TUNING accordingly
CONFIG_PREFIX=configs/prompt_tuning_tokens_config
CKPT=t5-base
LR_SCHEDULER_TYPE=linear
##### data & training setup stuff #####


# ### Get vocabulary overlapping (unigram, bigram)
for SEED in ${SEEDS[*]}
do
    for TOK_LEVEL in ${TOKENIZATIONS[*]}
    do
        for NGRAM in ${NGRAMS[*]}
        do
            # organized as follow: OUTPUT_DIR_PREFIX/TUNING/MODEL/TASK/SEED
            # output_dir="/data/users/pjlin/compacter/prompt_library/prompt_tuning_tokens_init_30k_adafactor_lr-${LR}/${MODEL}/${TASK_NAME}/${SEED}_run-2"
            # output_dir="/data/users/pjlin/compacter/task_selection/vocab_similarity_${TOK_LEVEL}_${NGRAM}"
            output_dir="/data/users/pjlin/compacter/task_selection/taskemb_test"

            #rm -fr $output_dir
            #mkdir -p $output_dir
            echo "output_dir: ${output_dir}"

            CONFIG=$CONFIG_PREFIX/$MODEL/prompt_tuning_tokens_init_${TASK_NAME}.json
            # CONFIG=configs/task_selection/t5-base/vocab_similarity_all.json
            
            # Run vocabulary similarity
            python run_vocab_sim.py \
                --config=$CONFIG \
                --model_name_or_path=$MODEL \
                --tokenization_level=$TOK_LEVEL \
                --num_token=1000 \
                --ngram=$NGRAM \
                --seed=$SEED \
                --data_seed=$SEED \
                --output_dir=$output_dir 2>&1 | tee ${output_dir}/train.log
        done
    done
done

