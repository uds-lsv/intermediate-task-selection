# This scripts trains prompt tuning with random initialization method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3.


# cb boolq rte mrpc cola qnli qqp mnli
# qnli qqp mnli
# boolq rte mrpc cola
GLUE_TASKS=(cb)

# t5-base or t5-large
MODEL=t5-base
# change TUNING accordingly
CONFIG_PREFIX=configs/prompt_tuning_config

#for SRC_TASK in ${GLUE_TASKS[*]}
#do
# SOURCE_TASK=(qqp mnli)

# mnli: "/data/users/pjlin/compacter/outputs/prompt_tuning_random_init/t5-base/mnli/42/checkpoint-9204"
# qqp:"/data/users/pjlin/compacter/outputs/prompt_tuning_random_init/t5-base/qqp/42/checkpoint-8502" 
CKPT="/data/users/pjlin/compacter/outputs/prompt_tuning_random_init/t5-base/mnli/42/checkpoint-9204"
# --model_name_or_path $CKPT
for TASK_NAME in ${GLUE_TASKS[*]}
do
	# organized as follow: OUTPUT_DIR_PREFIX/TUNING/MODEL/TASK/SEED
	# output_dir="/data/users/pjlin/compacter/transfer_outputs/prompt_tuning_random_init/${MODEL}/mnli_to_${TASK_NAME}/42"
	output_dir="/data/users/pjlin/compacter/test"
	rm -r $output_dir
	mkdir -p $output_dir
	echo "output_dir: ${output_dir}"
	
	CONFIG=$CONFIG_PREFIX/$MODEL/prompt_tuning_random_init_${TASK_NAME}.json
	# configs/prompt_tuning_tokens_init.json
	# python run_seq2seq.py $CONFIG $CKPT $output_dir 2>&1 | tee ${output_dir}/train.log
	python run_seq2seq.py $CONFIG $CKPT $output_dir 2>&1 | tee ${output_dir}/train.log
done


